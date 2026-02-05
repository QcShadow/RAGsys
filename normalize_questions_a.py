import os
import json
import time
import re
from typing import Any, Dict, List, Optional

import requests


# ============ 配置区 ============
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
# DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")  # 若你用的是别的网关，改这里
# DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY = "sk-66df7e992c384ffd8d352fb948465d2e"

DEFAULT_TIMEOUT = 60
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 0.2  # 防止打太快，可按需调大


# ============ Prompt（v3风格：严格 JSON + 受控输出） ============
SYSTEM_PROMPT = """你是“程序设计基础（C语言）”课程的助教。
你的任务是把学生报告中的“疑问候选句”做两件事：
1) 判断它是否是与课程学习相关的真实问题（is_course_question）。
2) 如果是，改写为更规范、可检索的一句话问题（normalized_question），保持原意，不要引入新知识点。
要求：
- 只输出 JSON，不要输出任何额外文字、解释或 markdown。
- normalized_question 必须是一个问句，使用中文，末尾加“？”。
- 如果候选句只是心得/陈述/无具体疑问，则 is_course_question=false，normalized_question 设为 null。
- confidence 给 0~1 的小数，越像真实课程问题越高。
- topic_hint 从给定 topics 中选择 0~2 个最相关的；若不确定可为空数组。
JSON Schema：
{
  "is_course_question": true/false,
  "normalized_question": "…？" or null,
  "confidence": 0.0~1.0,
  "topic_hint": ["..."] ,
  "notes": "可选的极简备注，不超过20字"
}
"""

USER_TEMPLATE = """文档元数据（可能为空）：
- doc_type: {doc_type}
- week: {week}
- topics: {topics}
- people: {people}

候选疑问：
- raw_text: {raw_text}
- context_prev: {context_prev}
- context_next: {context_next}

请按 schema 输出 JSON："""


# ============ 工具函数 ============
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_file_meta_map(metadata_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    metadata_results.json 是一个数组，每个元素包含 file/ week/ topics/ people/ doc_type 等。
    建立 file -> meta 的映射，便于按文件合并。
    """
    mp = {}
    for it in metadata_results:
        fpath = it.get("file")
        if fpath:
            mp[fpath] = it
    return mp


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    DeepSeek 有时会返回额外文字，这里尽量从回复里抠出第一个 JSON object。
    """
    text = text.strip()
    # 最理想情况：纯 JSON
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    # 从文本中提取 {...}
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return None


def call_deepseek(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY env var.")

    url = f"{DEEPSEEK_BASE_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def normalize_one_candidate(
    raw_text: str,
    context_prev: Optional[str],
    context_next: Optional[str],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    doc_type = meta.get("doc_type")
    week = meta.get("week")
    topics = meta.get("topics") or []
    people = meta.get("people") or {}

    user_prompt = USER_TEMPLATE.format(
        doc_type=doc_type,
        week=week,
        topics=topics,
        people=people,
        raw_text=raw_text,
        context_prev=context_prev,
        context_next=context_next,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            content = call_deepseek(messages, temperature=0.2)
            obj = extract_json_object(content)
            if obj is None:
                raise ValueError(f"JSON parse failed. Raw content: {content[:200]}")

            # 基本字段兜底
            if "is_course_question" not in obj:
                obj["is_course_question"] = False
            if "confidence" not in obj:
                obj["confidence"] = 0.0
            if "topic_hint" not in obj or obj["topic_hint"] is None:
                obj["topic_hint"] = []
            if obj.get("is_course_question") is False:
                obj["normalized_question"] = None

            # 约束：topic_hint 只能从 topics 中选（否则清空）
            topics_set = set(map(str, topics))
            obj["topic_hint"] = [t for t in obj.get("topic_hint", []) if str(t) in topics_set][:2]

            # 约束：normalized_question 必须以 ？ 结尾
            nq = obj.get("normalized_question")
            if nq and isinstance(nq, str):
                nq = nq.strip()
                if not nq.endswith("？"):
                    nq = nq + "？"
                obj["normalized_question"] = nq

            return obj

        except Exception as e:
            last_err = str(e)
            # 给模型一个“纠错指令”再重试
            messages.append({
                "role": "user",
                "content": f"上一次输出不符合要求（{last_err}）。请你这次只输出严格 JSON 对象，不能包含任何额外文本。",
            })
            time.sleep(0.4)

    return {
        "is_course_question": False,
        "normalized_question": None,
        "confidence": 0.0,
        "topic_hint": [],
        "notes": f"failed:{last_err}"[:20],
    }


# ============ 主流程 ============
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="question_candidates_v2.json", help="input candidates json")
    parser.add_argument("--metadata", default="metadata_results.json", help="optional metadata json (same folder)")
    parser.add_argument("--out", default="normalized_questions_a.json", help="output json")
    parser.add_argument("--only_file", default="", help="only process a specific file path (exact match)")
    args = parser.parse_args()

    candidates_data = load_json(args.candidates)

    meta_map: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(args.metadata):
        metadata_data = load_json(args.metadata)
        if isinstance(metadata_data, list):
            meta_map = build_file_meta_map(metadata_data)

    results = []
    total = 0
    kept = 0

    for file_item in candidates_data:
        fpath = file_item.get("file")
        if not fpath:
            continue
        if args.only_file and fpath != args.only_file:
            continue

        meta = meta_map.get(fpath, {})
        out_item = {
            "file": fpath,
            "meta": {
                "doc_type": meta.get("doc_type"),
                "week": meta.get("week"),
                "topics": meta.get("topics", []),
                "people": meta.get("people", {}),
            },
            "normalized": []
        }

        for cand in file_item.get("candidates", []):
            total += 1
            raw_text = cand.get("text", "")
            context_prev = cand.get("context_prev")
            context_next = cand.get("context_next")

            norm = normalize_one_candidate(raw_text, context_prev, context_next, meta)
            record = {
                "source_line_no": cand.get("source_line_no"),
                "sent_no": cand.get("sent_no"),
                "raw_text": raw_text,
                "context_prev": context_prev,
                "context_next": context_next,
                "triggers": cand.get("triggers", []),
                "in_q_section": cand.get("in_q_section", False),
                "llm": norm
            }
            out_item["normalized"].append(record)

            if norm.get("is_course_question") and norm.get("normalized_question"):
                kept += 1

            time.sleep(SLEEP_BETWEEN_CALLS)

        results.append(out_item)
        print(f"[OK] {os.path.basename(fpath)} candidates={len(file_item.get('candidates', []))}")

    summary = {
        "total_candidates": total,
        "kept_questions": kept,
        "kept_rate": (kept / total) if total else 0.0
    }

    final_out = {"summary": summary, "results": results}
    save_json(args.out, final_out)
    print(f"\nSaved: {args.out}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
