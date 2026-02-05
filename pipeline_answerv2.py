# pipeline_answerv2.py (stable v2.1)
# 作用：
#   读取 normalized_questions_*.json
#   - 对每个候选问题：优先使用 split_questions（多问拆分）逐条做 PPT-RAG
#   - 若无 split_questions，则用 normalized_question
#   - 每条子问题单独检索+回答，避免“多问混在一个 query”导致引用错乱
#   输出 qa_with_citations.json（结构尽量兼容你现有 pipeline_answer.py，但新增 rag_list）
#
# 用法（保持你原来的参数风格，不强依赖 notes）：
#   python pipeline_answerv2.py --in normalized_questions_v2.4.json --out qa_with_citations.json
#   python pipeline_answerv2.py --in normalized_questions_v2.4.json --out qa_with_citations.json --only_file "F:\...\自学报告-第1周-刘思扬组.docx"
#   python pipeline_answerv2.py --in normalized_questions_v2.4.json --out qa_with_citations.json --dry_run 1
#
import os
import sys
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

# ========= 路径与 import（适配 rag/database/rag_demo.py）=========
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DEMO_DIR = os.path.join(THIS_DIR, "database")
if RAG_DEMO_DIR not in sys.path:
    sys.path.insert(0, RAG_DEMO_DIR)

try:
    from database.rag_demo import (
        Hit,
        load_embedder,
        embed_query,
        load_faiss_and_db,
        search_ppt_hits,
        build_messages,
        deepseek_chat,
        PPT_INDEX_DIR,
        TOP_K,
    )
except Exception as e:
    raise RuntimeError(
        f"无法导入 rag_demo.py。请确认路径存在：{RAG_DEMO_DIR}\n"
        f"以及 rag_demo.py 在该目录下。\n原始错误：{e}"
    )

SLEEP_BETWEEN_QA = 0.2

# ======== v2 的关键策略：split_questions 最大处理条数（写死，避免太多问导致乱）========
MAX_SPLIT_Q_PER_CAND = 3  # 建议 2~3；先写 3，后面你可再调成 2


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def __strip_leading_numbering(s: str) -> str:
    import re
    return re.sub(r"^\(?\d{1,2}\)?[\.、\)]\s*", "", s).strip()


def norm_question_text(s: str) -> str:
    """清洗规范化问题：去编号、去多余空白、统一问号。"""
    if not s:
        return ""
    s = s.strip()
    s = s.replace("\u3000", " ").strip()
    s = __strip_leading_numbering(s)
    s = " ".join(s.split())
    if not s:
        return ""
    # 统一问号
    s = s.replace("?", "？")
    if not s.endswith("？"):
        s = s.rstrip("？") + "？"
    return s


def uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x = (x or "").strip()
        if not x:
            continue
        if x in seen:
            continue
        out.append(x)
        seen.add(x)
    return out


def dedupe_subsumed_questions(qs: List[str]) -> List[str]:
    """
    粗去冗余：如果一个问题文本几乎完全包含另一个问题（子串），优先保留更“具体/更长”的那个。
    这不是完美算法，但很稳，不依赖 LLM，也不需要 notes。
    """
    qs = uniq_keep_order([norm_question_text(q) for q in qs if q])
    if len(qs) <= 1:
        return qs

    # 按长度从长到短：先保留长的（通常更具体）
    qs_sorted = sorted(qs, key=lambda x: len(x), reverse=True)
    kept: List[str] = []
    for q in qs_sorted:
        is_subsumed = False
        for k in kept:
            # 若 q 是 k 的子串（去掉问号对比），认为 q 更泛/更短 -> 丢掉
            q0 = q.rstrip("？")
            k0 = k.rstrip("？")
            if q0 and (q0 in k0) and (len(k0) - len(q0) >= 2):
                is_subsumed = True
                break
        if not is_subsumed:
            kept.append(q)

    # 恢复一个稳定顺序：按原始出现顺序过滤
    kept_set = set(kept)
    return [q for q in qs if q in kept_set]


def pick_queries(
    rec: Dict[str, Any],
    meta: Dict[str, Any],
    use_topic_prefix: bool = False,
) -> Tuple[List[str], str]:
    """
    v2：优先用 split_questions（逐条检索）；否则用 normalized_question。
    返回：(queries, reason_if_empty)
    """
    llm = rec.get("llm") or rec.get("normalization") or {}
    is_q = bool(llm.get("is_course_question", False))
    if not is_q:
        return [], "not_course_question"

    # 1) 优先 split_questions
    split_qs = llm.get("split_questions")
    queries: List[str] = []

    if isinstance(split_qs, list) and split_qs:
        for x in split_qs:
            if isinstance(x, str) and x.strip():
                queries.append(x)

    # 2) 若 split_questions 不可用，用 normalized_question
    if not queries:
        nq = llm.get("normalized_question")
        if not (isinstance(nq, str) and nq.strip()):
            return [], "empty_normalized_question"
        queries = [nq]

    # 清洗 + 去冗余
    queries = [norm_question_text(q) for q in queries if norm_question_text(q)]
    queries = uniq_keep_order(queries)
    queries = dedupe_subsumed_questions(queries)

    # 限制最多处理 N 条
    queries = queries[:MAX_SPLIT_Q_PER_CAND]

    if not queries:
        return [], "empty_after_cleaning"

    # 可选：topic 前缀（仍然轻量；默认你不开）
    if use_topic_prefix:
        topic_hint = llm.get("topic_hint") or []
        topic = topic_hint[0] if isinstance(topic_hint, list) and topic_hint else None
        if not topic:
            topics = meta.get("topics") or []
            if isinstance(topics, list) and topics:
                topic = topics[0]
        if isinstance(topic, str) and topic.strip():
            queries = [f"{topic}：{q}" for q in queries]

    return queries, ""


def hit_to_dict(h: "Hit", snippet_chars: int = 500) -> Dict[str, Any]:
    meta = h.meta or {}
    file_name = meta.get("file_name") or meta.get("filename") or meta.get("source")
    page_no = meta.get("page_no") or meta.get("page") or meta.get("pageno")
    src = f"PPT:{file_name}#page={page_no}"

    text = (h.text or "").strip()
    snippet = text.replace("\n", " ")
    if len(snippet) > snippet_chars:
        snippet = snippet[:snippet_chars] + "..."

    return {
        "score": float(h.score),
        "idx": int(h.idx),
        "chunk_id": h.chunk_id,
        "doc_id": h.doc_id,
        "meta": meta,
        "source": src,
        "text_snippet": snippet,
    }


def run_rag_once(embedder, ppt_index, ppt_conn, question: str, top_k: int) -> Dict[str, Any]:
    qv = embed_query(embedder, question)
    hits = search_ppt_hits(ppt_index, ppt_conn, qv, top_k)
    messages = build_messages(question, hits)
    answer = deepseek_chat(messages)

    return {
        "question": question,
        "top_k": top_k,
        "hits": [hit_to_dict(h) for h in hits],
        "answer": answer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="normalized_questions_v2.4.json")
    parser.add_argument("--out", dest="out_path", default="qa_with_citations.json")
    parser.add_argument("--only_file", default="", help="只处理某一个学生文档（需与输入JSON里的 file 完全一致）")
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--use_topic_prefix", type=int, default=0, help="是否在query前加主题前缀（0/1），默认0更稳")
    parser.add_argument("--dry_run", type=int, default=0, help="只打印将要处理的query，不调用RAG（0/1）")
    args = parser.parse_args()

    in_abspath = os.path.abspath(args.in_path)
    if not os.path.exists(in_abspath):
        raise FileNotFoundError(f"输入文件不存在：{in_abspath}")

    data = load_json(in_abspath)

    # 兼容 normalize 输出结构：dict(results=...) 或 list
    docs = data.get("results", []) if isinstance(data, dict) else data
    if not isinstance(docs, list):
        raise ValueError("输入JSON结构不符合预期：需要 list，或 dict 中包含 results(list)。")

    # 预统计
    will_process: List[Tuple[str, str, int, int]] = []  # (file, query, source_line_no, sent_no)
    skipped_preview: List[Tuple[str, str]] = []  # (file, reason)
    total_candidates = 0

    for d in docs:
        fpath = d.get("file")
        if not fpath:
            continue
        if args.only_file and fpath != args.only_file:
            continue
        meta = d.get("meta", {}) or {}
        norm_list = d.get("normalized", []) or []
        total_candidates += len(norm_list)

        for rec in norm_list:
            qs, reason = pick_queries(rec, meta, use_topic_prefix=bool(args.use_topic_prefix))
            if qs:
                for q in qs:
                    will_process.append((fpath, q, rec.get("source_line_no"), rec.get("sent_no")))
            else:
                skipped_preview.append((fpath, reason))

    print(f"[INFO] input={in_abspath}")
    print(f"[INFO] docs={len(docs)} only_file={args.only_file!r}")
    print(f"[INFO] candidates_total={total_candidates}")
    print(f"[INFO] will_process_queries={len(will_process)} skipped_candidates={len(skipped_preview)}")
    print(f"[INFO] max_split_q_per_candidate={MAX_SPLIT_Q_PER_CAND}")

    if len(will_process) == 0:
        print("[WARN] 没有可处理的问题（is_course_question=false 或 normalized_question/split_questions 为空）。仍会输出文件用于检查。")

    if args.dry_run:
        for i, (f, q, ln, sn) in enumerate(will_process, 1):
            print(f"[DRYRUN] {i}. file={os.path.basename(f)} L{ln}:{sn} query={q}")
        save_json(args.out_path, {
            "summary": {"answered_questions": 0, "reason": "dry_run"},
            "items": []
        })
        print(f"[DRYRUN] Saved: {args.out_path}")
        return

    # 共享资源（只加载一次）
    embedder = load_embedder()
    ppt_index, ppt_conn = load_faiss_and_db(PPT_INDEX_DIR)

    out: Dict[str, Any] = {
        "input_file": in_abspath,
        "rag_demo_dir": RAG_DEMO_DIR,
        "ppt_index_dir": str(PPT_INDEX_DIR),
        "top_k": args.top_k,
        "use_topic_prefix": bool(args.use_topic_prefix),
        "max_split_q_per_candidate": MAX_SPLIT_Q_PER_CAND,
        "items": [],
    }

    answered = 0
    total_considered = 0

    try:
        for d in docs:
            fpath = d.get("file")
            if not fpath:
                continue
            if args.only_file and fpath != args.only_file:
                continue

            meta = d.get("meta", {}) or {}
            norm_list = d.get("normalized", []) or []

            doc_out = {
                "file": fpath,
                "meta": meta,
                "qa": []
            }

            for rec in norm_list:
                total_considered += 1

                qs, reason = pick_queries(rec, meta, use_topic_prefix=bool(args.use_topic_prefix))

                entry = {
                    "source_line_no": rec.get("source_line_no"),
                    "sent_no": rec.get("sent_no"),
                    "raw_text": rec.get("raw_text"),
                    "context_prev": rec.get("context_prev"),
                    "context_next": rec.get("context_next"),
                    "triggers": rec.get("triggers", []),
                    "in_q_section": rec.get("in_q_section", False),
                    # 不刻意使用 notes；原样保留 normalization 结构即可
                    "normalization": (rec.get("llm") or rec.get("normalization") or {}),
                    # v2：一个候选句可能拆成多条问题，因此输出 rag_list
                    "chosen_queries": qs,
                    "rag_list": [],
                    # 为兼容你旧逻辑：rag = rag_list[0]（如果存在）
                    "rag": None,
                }

                if not qs:
                    entry["rag"] = {"skipped": True, "reason": reason}
                    doc_out["qa"].append(entry)
                    continue

                # 对每条子问题分别 RAG（避免多问混在一个 query）
                for qi, q in enumerate(qs, 1):
                    print(f"[RUN] file={os.path.basename(fpath)} L{entry['source_line_no']}:{entry['sent_no']} q{qi}/{len(qs)} query={q}")
                    rag_res = run_rag_once(embedder, ppt_index, ppt_conn, q, args.top_k)
                    entry["rag_list"].append(rag_res)
                    answered += 1
                    time.sleep(SLEEP_BETWEEN_QA)

                # 兼容字段：取第一条作为 rag
                if entry["rag_list"]:
                    entry["rag"] = entry["rag_list"][0]

                doc_out["qa"].append(entry)

            out["items"].append(doc_out)
            doc_answered = sum(1 for x in doc_out["qa"] if x.get("rag_list"))
            print(f"[OK] {os.path.basename(fpath)} candidates={len(doc_out['qa'])} answered_candidates={doc_answered}")

    finally:
        try:
            ppt_conn.close()
        except Exception:
            pass

    out["summary"] = {
        "total_candidates": total_considered,
        "answered_queries": answered,  # 注意：这里统计的是“子问题条数”，不是候选句数
        "avg_queries_per_candidate": (answered / total_considered) if total_considered else 0.0,
    }

    save_json(args.out_path, out)
    print(f"\nSaved: {args.out_path}")
    print("Summary:", out["summary"])


if __name__ == "__main__":
    main()
