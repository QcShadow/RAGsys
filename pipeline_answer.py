# pipeline_answer.py (stable v2)
# 作用：读取 normalized_questions_a.json -> 仅用 normalized_question 做 PPT-RAG -> 保存 qa_with_citations.json
#
# 用法：
#   python pipeline_answer.py --in normalized_questions_a.json --out qa_with_citations.json
#   python pipeline_answer.py --in normalized_questions_a.json --out qa_with_citations.json --only_file "F:\...\自学报告-第1周-刘思扬组.docx"
#   python pipeline_answer.py --in normalized_questions_a.json --out qa_with_citations.json --dry_run 1   (只打印将会处理哪些问题，不调用 RAG)
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
    from database.rag_demo import  (
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


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def norm_question_text(s: str) -> str:
    """清洗规范化问题：去编号、去多余空白、统一问号。"""
    if not s:
        return ""
    s = s.strip()
    # 去掉开头编号：1. / 1、 / (1) / 1)
    s = s.lstrip()
    s = s.replace("\u3000", " ")
    s = s.strip()
    s = __strip_leading_numbering(s)
    s = " ".join(s.split())
    if s and not s.endswith("？"):
        # 如果本来是英文问号也转一下
        s = s.rstrip("?") + "？"
    return s


def __strip_leading_numbering(s: str) -> str:
    import re
    return re.sub(r"^\(?\d{1,2}\)?[\.、\)]\s*", "", s).strip()


def pick_query(rec: Dict[str, Any], meta: Dict[str, Any], use_topic_prefix: bool = False) -> Tuple[Optional[str], str]:
    """
    只取 normalized_question 作为 query，避免 JSON 噪声污染召回。
    可选：use_topic_prefix=True 时，会把 topic_hint 的第1个作为前缀（轻微约束检索）。
    返回：(query or None, reason_if_none)
    """
    llm = rec.get("llm") or rec.get("normalization") or {}
    is_q = bool(llm.get("is_course_question", False))
    nq = llm.get("normalized_question")

    if (not is_q) or (not nq) or (not isinstance(nq, str)) or (not nq.strip()):
        return None, "not_course_question_or_empty_normalized_question"

    q = norm_question_text(nq)

    if use_topic_prefix:
        # topic_hint 优先，其次用 meta.topics
        topic_hint = llm.get("topic_hint") or []
        topic = topic_hint[0] if isinstance(topic_hint, list) and topic_hint else None
        if not topic:
            topics = meta.get("topics") or []
            if isinstance(topics, list) and topics:
                topic = topics[0]
        if topic and isinstance(topic, str) and topic.strip():
            # 轻量前缀，不把一堆元数据塞进去
            q = f"{topic}：{q}"

    return q, ""


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
    parser.add_argument("--in", dest="in_path", default="normalized_questions_a.json")
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

    # 兼容 normalize_questions_a.py 的输出结构
    docs = data.get("results", []) if isinstance(data, dict) else data
    if not isinstance(docs, list):
        raise ValueError("输入JSON结构不符合预期：需要 list，或 dict 中包含 results(list)。")

    # 提前统计可处理的问题数量（避免“什么都没打印”的困惑）
    will_process: List[Tuple[str, str]] = []  # (file, query)
    skipped_preview: List[Tuple[str, str]] = []  # (file, reason)
    for d in docs:
        fpath = d.get("file")
        if not fpath:
            continue
        if args.only_file and fpath != args.only_file:
            continue
        meta = d.get("meta", {}) or {}
        norm_list = d.get("normalized", []) or []
        for rec in norm_list:
            q, reason = pick_query(rec, meta, use_topic_prefix=bool(args.use_topic_prefix))
            if q:
                will_process.append((fpath, q))
            else:
                skipped_preview.append((fpath, reason))

    print(f"[INFO] input={in_abspath}")
    print(f"[INFO] docs={len(docs)} only_file={args.only_file!r}")
    print(f"[INFO] candidates_total={sum(len((d.get('normalized') or [])) for d in docs)}")
    print(f"[INFO] will_process_questions={len(will_process)} skipped={len(skipped_preview)}")
    if len(will_process) == 0:
        print("[WARN] 没有可处理的问题（is_course_question=false 或 normalized_question为空）。仍会输出文件用于检查。")

    if args.dry_run:
        for i, (f, q) in enumerate(will_process, 1):
            print(f"[DRYRUN] {i}. file={os.path.basename(f)} query={q}")
        # 仍然输出空结果文件
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
                q, reason = pick_query(rec, meta, use_topic_prefix=bool(args.use_topic_prefix))

                entry = {
                    "source_line_no": rec.get("source_line_no"),
                    "sent_no": rec.get("sent_no"),
                    "raw_text": rec.get("raw_text"),
                    "context_prev": rec.get("context_prev"),
                    "context_next": rec.get("context_next"),
                    "triggers": rec.get("triggers", []),
                    "in_q_section": rec.get("in_q_section", False),
                    "normalization": (rec.get("llm") or rec.get("normalization") or {}),
                    "rag": None,
                }

                if not q:
                    entry["rag"] = {"skipped": True, "reason": reason}
                    doc_out["qa"].append(entry)
                    continue

                print(f"[RUN] file={os.path.basename(fpath)} query={q}")
                rag_res = run_rag_once(embedder, ppt_index, ppt_conn, q, args.top_k)
                entry["rag"] = rag_res
                doc_out["qa"].append(entry)
                answered += 1

                time.sleep(SLEEP_BETWEEN_QA)

            out["items"].append(doc_out)
            print(f"[OK] {os.path.basename(fpath)} answered={sum(1 for x in doc_out['qa'] if x.get('rag') and not x['rag'].get('skipped', False))}")

    finally:
        try:
            ppt_conn.close()
        except Exception:
            pass

    out["summary"] = {
        "total_candidates": total_considered,
        "answered_questions": answered,
        "answer_rate": (answered / total_considered) if total_considered else 0.0
    }

    save_json(args.out_path, out)
    print(f"\nSaved: {args.out_path}")
    print("Summary:", out["summary"])


if __name__ == "__main__":
    main()
