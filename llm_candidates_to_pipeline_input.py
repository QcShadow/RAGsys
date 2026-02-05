#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
candidates_to_pipeline_input.py

用途：
- 读取 question_candidates_*.json（你的抽取结果，list[ {file, candidates:[...]} ]）
- 生成 pipeline_answerv2.py 可直接使用的“normalized_questions”格式 JSON
- （可选）直接调用 pipeline_answerv2.py 跑 RAG 并输出 qa_with_citations_v2.json

运行示例：
1) 只做转换：
python candidates_to_pipeline_input.py \
  --candidates runs/norm_dataset/20260130_105000/question_candidates_v2_3_fallback_llm_markercontv3.json \
  --meta runs/norm_dataset/20260130_105000/metadata_results.json \
  --out runs/norm_dataset/20260130_105000/normalized_questions_from_candidates.json

2) 转换后直接跑 pipeline：
python candidates_to_pipeline_input.py \
  --candidates runs/norm_dataset/20260130_105000/question_candidates_v2_3_fallback_llm_markercontv3.json \
  --meta runs/norm_dataset/20260130_105000/metadata_results.json \
  --out runs/norm_dataset/20260130_105000/normalized_questions_from_candidates.json \
  --run_pipeline \
  --pipeline_py pipeline_answerv2.py \
  --pipeline_out runs/norm_dataset/20260130_105000/qa_with_citations_v2_from_candidates.json
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional


def make_nonconflicting_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    if not ext:
        ext = ".json"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{ts}{ext}"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_meta_map(meta_items: Any) -> Dict[str, Dict[str, Any]]:
    """
    metadata_results.json 一般是 list[{file:..., ...}]
    建一个 file(abs_path) -> meta 的映射，方便合并。
    """
    meta_map: Dict[str, Dict[str, Any]] = {}
    if not meta_items:
        return meta_map

    if isinstance(meta_items, dict) and "items" in meta_items:
        meta_items = meta_items["items"]

    if not isinstance(meta_items, list):
        return meta_map

    for it in meta_items:
        if not isinstance(it, dict):
            continue
        fp = it.get("file")
        if not fp:
            continue
        meta_map[os.path.abspath(fp)] = it
    return meta_map


def guess_question_type(triggers: List[str], raw_text: str) -> str:
    """
    粗分 QUESTION / CONFUSION（只是附加信息，不影响 pipeline 跑）
    """
    raw = (raw_text or "")
    trig = triggers or []
    if "HAS_QUESTION_MARK" in trig or "？" in raw or "?" in raw:
        return "QUESTION"
    if any(t.startswith("QWORD:") for t in trig):
        return "QUESTION"
    if any(t.startswith("CONFUSE:") for t in trig):
        return "CONFUSION"
    return "UNKNOWN"


def convert_candidates_to_normalized(
    candidates_data: Any,
    meta_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    输出结构尽量贴近 normalized_questions_run.py 的产物（pipeline_answerv2.py 可读）：
    {
      "items": [
        {
          "file": "...",
          "meta": {...},  # 可选
          "candidates": [
            {
              "source_line_no": ...,
              "sent_no": ...,
              "raw_text": "...",
              "context_prev": "...",
              "context_next": "...",
              "triggers": [...],
              "in_q_section": true/false,
              "llm": {...} or null,
              "normalization": {
                 "original": "...",
                 "normalized": "...",
                 "split_questions": ["..."],
                 "question_type": "QUESTION/CONFUSION/UNKNOWN",
                 "confidence": 0.5,
                 "policy": "NO_LLM_NORMALIZATION"
              }
            }
          ]
        }
      ],
      "summary": {...}
    }
    """
    if isinstance(candidates_data, dict) and "items" in candidates_data:
        # 有些人会把它包一层 items
        candidates_data = candidates_data["items"]

    if not isinstance(candidates_data, list):
        raise ValueError("candidates json 顶层应为 list（或 dict 包含 items:list）")

    out_items: List[Dict[str, Any]] = []
    total_candidates = 0

    for file_item in candidates_data:
        if not isinstance(file_item, dict):
            continue

        file_path = file_item.get("file", "")
        abs_file = os.path.abspath(file_path) if file_path else file_path

        cands = file_item.get("candidates", []) or []
        norm_cands: List[Dict[str, Any]] = []

        for c in cands:
            if not isinstance(c, dict):
                continue

            raw_text = (c.get("text") or c.get("raw_text") or "").strip()
            if not raw_text:
                continue

            triggers = c.get("triggers") or []
            if not isinstance(triggers, list):
                triggers = [str(triggers)]

            qtype = guess_question_type(triggers, raw_text)

            norm_cands.append({
                "source_line_no": c.get("source_line_no", -1),
                "sent_no": c.get("sent_no", 0),
                "raw_text": raw_text,
                "context_prev": c.get("context_prev"),
                "context_next": c.get("context_next"),
                "triggers": triggers,
                "in_q_section": bool(c.get("in_q_section", False)),
                "llm": c.get("llm", None),
                "normalization": {
                    "original": raw_text,
                    "normalized": raw_text,              # 最小可用：不改写
                    "split_questions": [raw_text],       # 最小可用：不拆分
                    "question_type": qtype,
                    "confidence": 0.5,
                    "policy": "NO_LLM_NORMALIZATION",
                }
            })

        total_candidates += len(norm_cands)

        out_items.append({
            "file": abs_file,
            "meta": meta_map.get(abs_file, None),
            "candidates": norm_cands,
        })

    out = {
        "items": out_items,
        "summary": {
            "total_files": len(out_items),
            "total_candidates": total_candidates,
            "note": "Converted from question_candidates json without LLM normalization.",
        }
    }
    return out


def run_cmd(cmd: List[str], cwd: Optional[str] = None):
    print("\n[CMD]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="question_candidates_*.json 路径")
    ap.add_argument("--meta", default="", help="metadata_results.json 路径（可选）")
    ap.add_argument("--out", default="normalized_questions_from_candidates.json", help="输出 normalized json")
    ap.add_argument("--no_timestamp", action="store_true", help="不自动避开重名覆盖（默认会加时间戳）")

    # 可选：直接跑 pipeline
    ap.add_argument("--run_pipeline", action="store_true", help="转换后直接调用 pipeline_answerv2.py")
    ap.add_argument("--pipeline_py", default="pipeline_answerv2.py", help="pipeline_answerv2.py 的路径")
    ap.add_argument("--pipeline_out", default="qa_with_citations_v2_from_candidates.json", help="pipeline 输出 json 路径")

    args = ap.parse_args()

    candidates_path = args.candidates
    meta_path = args.meta.strip()

    out_path = args.out
    if not args.no_timestamp:
        out_path = make_nonconflicting_path(out_path)

    candidates_data = load_json(candidates_path)
    meta_map: Dict[str, Dict[str, Any]] = {}
    if meta_path:
        meta_data = load_json(meta_path)
        meta_map = build_meta_map(meta_data)

    converted = convert_candidates_to_normalized(candidates_data, meta_map)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved normalized input: {out_path}")
    print(f"[INFO] summary: {converted.get('summary')}")

    if args.run_pipeline:
        pipeline_out = args.pipeline_out
        if not args.no_timestamp:
            pipeline_out = make_nonconflicting_path(pipeline_out)

        py = sys.executable
        run_cmd([py, args.pipeline_py, "--in", out_path, "--out", pipeline_out])
        print(f"[DONE] pipeline output: {pipeline_out}")


if __name__ == "__main__":
    main()
