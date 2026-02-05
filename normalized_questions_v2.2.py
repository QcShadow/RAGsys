import os
import json
import time
import re
import hashlib
from typing import Any, Dict, List, Optional

import requests


# ============ 配置区（写死） ============
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY = "sk-66df7e992c384ffd8d352fb948465d2e"  # 建议你本地替换；不要在公开场合暴露

DEFAULT_TIMEOUT = 60
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 0.2  # 防止打太快，可按需调大

# 默认输入输出（写死）
DEFAULT_CANDIDATES = "question_candidates_v2_2.json"
DEFAULT_METADATA = "metadata_results.json"
DEFAULT_OUT = "normalized_questions_v2.2.json"
DEFAULT_RESUME = True


# ============ Prompt（更稳：严格 JSON + 受控输出） ============
SYSTEM_PROMPT = """你是“程序设计基础（C语言）”课程的助教。
你的任务是把学生报告中的“疑问候选句”做两件事：
1) 判断它是否是与课程学习相关的真实问题（is_course_question）。
2) 如果是，改写为更规范、可检索的一句话问题（normalized_question），保持原意，不要引入新知识点。

重要要求：
- 只输出 JSON，不要输出任何额外文字、解释或 markdown。
- normalized_question 必须是中文问句，末尾必须是“？”。
- 允许做轻量纠错：如将常见误写“ASCLL码”纠正为“ASCII码”，但不得改变问题意图。
- 若候选句混杂陈述 + 问句：只提取其中最明确的“问句部分”进行规范化（不要把陈述带进问题）。
- 如果候选句只是心得/陈述/无具体疑问，则 is_course_question=false，normalized_question=null。
- confidence 给 0~1 的小数，越像真实课程问题越高。
- topic_hint 从给定 topics 中选择 0~2 个最相关的；若不确定可为空数组。

JSON Schema（必须严格匹配字段名）：
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
    """metadata_results.json 是数组：每个元素包含 file/week/topics/people/doc_type 等。"""
    mp = {}
    for it in metadata_results:
        fpath = it.get("file")
        if fpath:
            mp[fpath] = it
    return mp


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """尽量从回复里抠出第一个 JSON object。"""
    text = (text or "").strip()

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


def sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


# ============ 预清洗（减少姓名等噪声对规范化的干扰） ============
SPEAKER_PREFIX_RE = re.compile(r"^\s*[\u4e00-\u9fa5]{2,6}\s*[：:]\s*")  # 张三：/李四:

def preprocess_raw_text(raw_text: str) -> str:
    t = (raw_text or "").strip()
    t = t.replace("?", "？")
    t = re.sub(r"\s+", " ", t).strip()
    # 去掉“人名：”
    t = SPEAKER_PREFIX_RE.sub("", t).strip()
    return t


def call_deepseek(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY in code.")

    url = f"{DEEPSEEK_BASE_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
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

    cleaned = preprocess_raw_text(raw_text)

    user_prompt = USER_TEMPLATE.format(
        doc_type=doc_type,
        week=week,
        topics=topics,
        people=people,
        raw_text=cleaned,
        context_prev=(context_prev or ""),
        context_next=(context_next or ""),
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
                raise ValueError(f"JSON parse failed. Raw head: {content[:200]}")

            # 字段兜底
            if "is_course_question" not in obj:
                obj["is_course_question"] = False
            if "confidence" not in obj:
                obj["confidence"] = 0.0
            if "topic_hint" not in obj or obj["topic_hint"] is None:
                obj["topic_hint"] = []
            if "notes" not in obj or obj["notes"] is None:
                obj["notes"] = ""

            # is_course_question=false => normalized_question=null
            if obj.get("is_course_question") is False:
                obj["normalized_question"] = None

            # topic_hint 只能从 topics 中选
            topics_set = set(map(str, topics))
            obj["topic_hint"] = [t for t in obj.get("topic_hint", []) if str(t) in topics_set][:2]

            # normalized_question 必须以 ？ 结尾
            nq = obj.get("normalized_question")
            if nq and isinstance(nq, str):
                nq = nq.strip().replace("?", "？")
                if not nq.endswith("？"):
                    nq = nq + "？"
                obj["normalized_question"] = nq

            # confidence 规整到 [0,1]
            try:
                c = float(obj.get("confidence", 0.0))
            except Exception:
                c = 0.0
            c = max(0.0, min(1.0, c))
            obj["confidence"] = c

            # notes 限长
            obj["notes"] = (obj.get("notes") or "")[:20]

            return obj

        except Exception as e:
            last_err = str(e)
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
        "notes": (f"failed:{last_err}" if last_err else "failed")[:20],
    }


# ============ 主流程 ============
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES, help="input candidates json")
    parser.add_argument("--metadata", default=DEFAULT_METADATA, help="optional metadata json (same folder)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output json")
    parser.add_argument("--only_file", default="", help="only process a specific file path (exact match)")
    args = parser.parse_args()

    candidates_data = load_json(args.candidates)

    meta_map: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(args.metadata):
        metadata_data = load_json(args.metadata)
        if isinstance(metadata_data, list):
            meta_map = build_file_meta_map(metadata_data)

    # resume：如果输出已存在，读取并跳过已做的 candidate（默认开启）
    done_keys = set()
    existing_results: List[Dict[str, Any]] = []
    if DEFAULT_RESUME and os.path.exists(args.out):
        try:
            old = load_json(args.out)
            existing_results = old.get("results", []) or []
            for item in existing_results:
                f = item.get("file")
                for rec in item.get("normalized", []):
                    key = sha1(f"{f}|{rec.get('source_line_no')}|{rec.get('sent_no')}|{rec.get('raw_text')}")
                    done_keys.add(key)
            print(f"[RESUME] found existing: {args.out}, skip_done={len(done_keys)}")
        except Exception:
            existing_results = []
            done_keys = set()

    results = existing_results[:]  # 继续追加
    total = 0
    kept = 0

    def get_or_create_out_item(fpath: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        for it in results:
            if it.get("file") == fpath:
                return it
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
        results.append(out_item)
        return out_item

    for file_item in candidates_data:
        fpath = file_item.get("file")
        if not fpath:
            continue
        if args.only_file and fpath != args.only_file:
            continue

        meta = meta_map.get(fpath, {})
        out_item = get_or_create_out_item(fpath, meta)

        cands = file_item.get("candidates", []) or []
        print(f"[INFO] {os.path.basename(fpath)} candidates={len(cands)}")

        for cand in cands:
            total += 1
            raw_text = cand.get("text", "")
            key = sha1(f"{fpath}|{cand.get('source_line_no')}|{cand.get('sent_no')}|{raw_text}")
            if key in done_keys:
                continue

            context_prev = cand.get("context_prev")
            context_next = cand.get("context_next")

            norm = normalize_one_candidate(raw_text, context_prev, context_next, meta)

            record = {
                "source_line_no": cand.get("source_line_no"),
                "sent_no": cand.get("sent_no"),
                "raw_text": raw_text,  # 原样保留
                "context_prev": context_prev,
                "context_next": context_next,
                "triggers": cand.get("triggers", []),
                "in_q_section": cand.get("in_q_section", False),
                "llm": norm
            }
            out_item["normalized"].append(record)

            if norm.get("is_course_question") and norm.get("normalized_question"):
                kept += 1

            done_keys.add(key)
            time.sleep(SLEEP_BETWEEN_CALLS)

        print(f"[OK] {os.path.basename(fpath)} done={len(out_item['normalized'])}")

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
