#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pipeline_answer_llmonly.py

作用：
  读取 llm_only / 规则抽取 的 candidates JSON（顶层为 list）
  对每个候选：直接使用 candidates[].text 作为 query 做 PPT-RAG
  不做规范化、不做拆分、不做 is_course_question 门控，便于公正评估。

用法：
  python pipeline_answer_llmonly.py --in llm_only_candidates.json --out qa_with_citations_llmonly.json
  python pipeline_answer_llmonly.py --in llm_only_candidates.json --out qa.json --only_file "F:\\...\\某docx"
  python pipeline_answer_llmonly.py --in llm_only_candidates.json --out qa.json --dry_run 1
"""

import os
import sys
import json
import time
import argparse
from typing import Any, Dict, List, Tuple

# ========= 路径与 import（完全仿照 pipeline_answerv2.py）=========
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


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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
    parser.add_argument("--in", dest="in_path", required=True, help="llm_only / candidates JSON（顶层 list）")
    parser.add_argument("--out", dest="out_path", default="qa_with_citations_llmonly.json")
    parser.add_argument("--only_file", default="", help="只处理某一个学生文档（需与输入JSON里的 file 完全一致）")
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--dry_run", type=int, default=0, help="只打印将要处理的query，不调用RAG（0/1）")
    args = parser.parse_args()

    in_abspath = os.path.abspath(args.in_path)
    if not os.path.exists(in_abspath):
        raise FileNotFoundError(f"输入文件不存在：{in_abspath}")

    data = load_json(in_abspath)
    if not isinstance(data, list):
        raise ValueError("输入JSON结构不符合预期：顶层需要是 list（每个元素包含 file/candidates）。")

    # 预统计
    will_process: List[Tuple[str, str, int, int]] = []  # (file, query, source_line_no, sent_no)
    total_candidates = 0

    for d in data:
        fpath = d.get("file")
        if not fpath:
            continue
        if args.only_file and fpath != args.only_file:
            continue

        cands = d.get("candidates") or []
        if not isinstance(cands, list):
            continue
        total_candidates += len(cands)

        for c in cands:
            q = (c.get("text") or "").strip()
            if not q:
                continue
            will_process.append((fpath, q, int(c.get("source_line_no", -1)), int(c.get("sent_no", 0))))

    print(f"[INFO] input={in_abspath}")
    print(f"[INFO] docs={len(data)} only_file={args.only_file!r}")
    print(f"[INFO] candidates_total={total_candidates}")
    print(f"[INFO] will_process_queries={len(will_process)}")
    print(f"[INFO] top_k={args.top_k}")

    if args.dry_run:
        for i, (f, q, ln, sn) in enumerate(will_process, 1):
            print(f"[DRYRUN] {i}. file={os.path.basename(f)} L{ln}:{sn} query={q}")
        save_json(args.out_path, {
            "input_file": in_abspath,
            "summary": {"answered_queries": 0, "reason": "dry_run"},
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
        "items": [],
    }

    answered_queries = 0
    total_considered = 0

    try:
        for d in data:
            fpath = d.get("file")
            if not fpath:
                continue
            if args.only_file and fpath != args.only_file:
                continue

            cands = d.get("candidates") or []
            if not isinstance(cands, list):
                cands = []

            doc_out = {
                "file": fpath,
                "qa": []
            }

            for c in cands:
                total_considered += 1

                q = (c.get("text") or "").strip()
                entry = {
                    "source_line_no": c.get("source_line_no", -1),
                    "sent_no": c.get("sent_no", 0),
                    "raw_text": q,
                    "context_prev": c.get("context_prev"),
                    "context_next": c.get("context_next"),
                    "triggers": c.get("triggers", []),
                    "in_q_section": c.get("in_q_section", False),
                    "llm": c.get("llm"),  # 原样保留（如果有）
                    "chosen_queries": [q] if q else [],
                    "rag_list": [],
                    "rag": None,
                }

                if not q:
                    entry["rag"] = {"skipped": True, "reason": "empty_text"}
                    doc_out["qa"].append(entry)
                    continue

                print(f"[RUN] file={os.path.basename(fpath)} L{entry['source_line_no']}:{entry['sent_no']} query={q}")
                rag_res = run_rag_once(embedder, ppt_index, ppt_conn, q, args.top_k)
                entry["rag_list"].append(rag_res)
                entry["rag"] = entry["rag_list"][0]
                answered_queries += 1
                time.sleep(SLEEP_BETWEEN_QA)

                doc_out["qa"].append(entry)

            out["items"].append(doc_out)
            print(f"[OK] {os.path.basename(fpath)} candidates={len(doc_out['qa'])} answered={sum(1 for x in doc_out['qa'] if x.get('rag_list'))}")

    finally:
        try:
            ppt_conn.close()
        except Exception:
            pass

    out["summary"] = {
        "total_candidates": total_considered,
        "answered_queries": answered_queries,
        "answer_rate": (answered_queries / total_considered) if total_considered else 0.0,
    }

    save_json(args.out_path, out)
    print(f"\nSaved: {args.out_path}")
    print("Summary:", out["summary"])


if __name__ == "__main__":
    main()
