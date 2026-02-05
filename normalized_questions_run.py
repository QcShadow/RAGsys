# normalized_questions_run.py
# 基于 normalized_questions_v2.4.py 的“可传参复制版”
# - 支持 --candidates --meta --out
# - 强烈建议把 DEEPSEEK_API_KEY 放到环境变量里：DEEPSEEK_API_KEY=...
import os
import json
import time
import re
import argparse
from typing import Any, Dict, List, Optional

import requests

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY = "sk-66df7e992c384ffd8d352fb948465d2e"

DEFAULT_TIMEOUT = 60
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 0.2

SYSTEM_PROMPT = """你是“程序设计基础（C语言）”课程的助教。
你的任务是把学生报告中的“疑问候选句”做两件事：
1) 判断它是否是与课程学习相关的真实问题（is_course_question）。
2) 如果是，改写为更规范、可检索的一句话问题（normalized_question），保持原意，不要引入新知识点。

额外要求（多问拆分）：
- 如果候选句里包含多个独立问题，请拆分成多条标准问句，放到 split_questions 数组中。
- 若只有一个问题，split_questions 也必须返回一个长度为1的数组，内容与 normalized_question 一致。
- 若不是课程问题：is_course_question=false，normalized_question=null，split_questions=[]。

输出约束：
- 只输出 JSON，不要输出任何额外文字、解释或 markdown。
- normalized_question 必须是中文问句，末尾加“？”（全角）。
- split_questions 中每条也必须是中文问句，末尾加“？”（全角）。
- 允许轻量纠错：例如将常见误写“ASCLL码”纠正为“ASCII码”，但不得改变问题意图。
- 如果候选句混杂陈述 + 问句，只提取其中最明确的“问句部分”进行规范化（不要把陈述内容带进问题）。
- confidence 给 0~1 的小数，越像真实课程问题越高。
- topic_hint 从给定 topics 中选择 0~2 个最相关项；若不确定可为空数组。

置信度（confidence）打分要求（必须拉开区间）：
- 0.85~1.00：单意图、明确课程问题，几乎可直接检索。
- 0.60~0.84：问题基本明确，但有歧义/上下文依赖/或多意图（即使已拆分）。
- 0.00~0.59：更像心得陈述、表达含糊、或不确定是否课程相关。

JSON Schema（必须严格匹配字段名）：
{
  "is_course_question": true/false,
  "normalized_question": "…？" or null,
  "split_questions": ["…？", "...？"],
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

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def build_file_meta_map(metadata_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mp = {}
    for it in metadata_results:
        fpath = it.get("file")
        if fpath:
            mp[fpath] = it
    return mp

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

SPEAKER_PREFIX_RE = re.compile(r"^\s*[\u4e00-\u9fa5]{2,10}\s*[：:]\s*")

def preprocess_raw_text(raw_text: str) -> str:
    t = (raw_text or "").strip()
    t = t.replace("?", "？")
    t = re.sub(r"\s+", " ", t).strip()
    t = SPEAKER_PREFIX_RE.sub("", t).strip()
    return t

def ensure_qmark(s: str) -> str:
    s = (s or "").strip().replace("?", "？")
    if s and not s.endswith("？"):
        s += "？"
    return s

def normalize_llm_output(obj: Dict[str, Any], topics: List[Any]) -> Dict[str, Any]:
    if "is_course_question" not in obj:
        obj["is_course_question"] = False
    if "confidence" not in obj:
        obj["confidence"] = 0.0
    if "topic_hint" not in obj or obj["topic_hint"] is None:
        obj["topic_hint"] = []
    if "notes" not in obj or obj["notes"] is None:
        obj["notes"] = ""

    topics_set = set(map(str, topics or []))
    obj["topic_hint"] = [t for t in (obj.get("topic_hint") or []) if str(t) in topics_set][:2]

    if obj.get("is_course_question") is False:
        obj["normalized_question"] = None
        obj["split_questions"] = []
    else:
        nq = obj.get("normalized_question")
        if not nq or not isinstance(nq, str) or not nq.strip():
            obj["is_course_question"] = False
            obj["normalized_question"] = None
            obj["split_questions"] = []
        else:
            nq = ensure_qmark(nq)
            obj["normalized_question"] = nq

            sq = obj.get("split_questions") or []
            sq2: List[str] = []
            for x in sq:
                if isinstance(x, str):
                    x = ensure_qmark(x).strip()
                    if x:
                        sq2.append(x)
            sq2 = list(dict.fromkeys(sq2))

            if nq not in sq2:
                sq2 = [nq] + sq2
            else:
                sq2 = [nq] + [x for x in sq2 if x != nq]
            obj["split_questions"] = sq2

    try:
        c = float(obj.get("confidence", 0.0))
    except Exception:
        c = 0.0
    obj["confidence"] = max(0.0, min(1.0, c))
    obj["notes"] = (obj.get("notes") or "")[:20]
    return obj

def call_deepseek(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    api_key = DEEPSEEK_API_KEY
    url = f"{DEEPSEEK_BASE_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT)
    if resp.status_code != 200:
        body = (resp.text or "")[:600]
        raise RuntimeError(f"HTTP {resp.status_code} {resp.reason} | body={body}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def normalize_one_candidate(raw_text: str, context_prev: Optional[str], context_next: Optional[str], meta: Dict[str, Any]) -> Dict[str, Any]:
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
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}]

    last_err = None
    for _ in range(1, MAX_RETRIES + 1):
        try:
            content = call_deepseek(messages, temperature=0.2)
            obj = extract_json_object(content)
            if obj is None:
                raise ValueError(f"JSON parse failed. head={content[:200]!r}")
            return normalize_llm_output(obj, topics=topics)
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
        "split_questions": [],
        "confidence": 0.0,
        "topic_hint": [],
        "notes": f"failed:{last_err}"[:20] if last_err else "failed",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="question_candidates_v2_2.json")
    ap.add_argument("--meta", required=True, help="metadata_results.json")
    ap.add_argument("--out", required=True, help="output normalized_questions_*.json")
    ap.add_argument("--only_file", default="", help="只跑某一个学生文件（绝对路径需与 candidates 里一致）")
    args = ap.parse_args()

    candidates_data = load_json(args.candidates)
    metadata_data = load_json(args.meta) if os.path.exists(args.meta) else []
    meta_map = build_file_meta_map(metadata_data) if isinstance(metadata_data, list) else {}

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
    save_json(args.out, {"summary": summary, "results": results})
    print(f"\nSaved: {args.out}")
    print(f"Summary: {summary}")

if __name__ == "__main__":
    main()
