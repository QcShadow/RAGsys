import os
import json
import time
import re
from typing import Any, Dict, List, Optional

import requests


# ============ 配置区（写死） ============
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_KEY = "sk-66df7e992c384ffd8d352fb948465d2e"  # 建议你本地替换；不要在公开场合暴露

DEFAULT_TIMEOUT = 60
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 0.2

# 写死输入输出（按你的习惯）
CANDIDATES_PATH = "question_candidates_v2_2.json"
METADATA_PATH = "metadata_results.json"
OUT_PATH = "normalized_questions_v2.3.json"


# ============ Prompt（升级：支持多问拆分 split_questions；拉开置信度区间） ============
SYSTEM_PROMPT = """你是“程序设计基础（C语言）”课程的助教。
你将收到学生报告中抽取出的“疑问候选句”（通常已是单条问题，但可能仍含多个问题或夹杂陈述）。

你的任务：
1) 判断它是否是与课程学习相关、且具有明确提问意图的问题（is_course_question）。
2) 若是：改写为更规范、可检索的一句话中文问句（normalized_question），保持原意，不引入新知识点。
3) 若候选句实际包含多个可分离的问题：请拆分为多条问句放到 split_questions 数组中（每条必须以“？”结尾）。
   - 此时 normalized_question 仍需给出“最核心/最主要”的那一条（用于兼容旧流程）。
4) 若不是：is_course_question=false，normalized_question=null，split_questions=[]。

重要约束：
- 只输出一个 JSON 对象，不能输出任何额外文字、解释、markdown、代码块。
- normalized_question 必须是中文问句，末尾必须是“？”；split_questions 中每条也必须以“？”结尾。
- 允许做轻量纠错：如“ASCLL码”纠正为“ASCII码”，但不得改变问题意图。
- topic_hint 只能从给定 topics 中选择 0~2 个最相关项；不确定就返回 []。
- notes 只允许写“multi/single/spellfix/unclear”这类极短标签，不要写长解释（<=20字）。

置信度（confidence）打分要求（必须拉开区间）：
- 0.85~1.00：单意图、明确课程问题，几乎可直接检索。
- 0.60~0.84：问题基本明确，但有歧义/上下文依赖/或多意图（即使已拆分）。
- 0.00~0.59：更像心得陈述、表达含糊、或不确定是否课程相关。

JSON Schema（字段名必须一致）：
{
  "is_course_question": true/false,
  "normalized_question": "…？" or null,
  "split_questions": ["…？", "...？"],
  "confidence": 0.0~1.0,
  "topic_hint": ["..."],
  "notes": "不超过20字"
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

请按 schema 输出严格 JSON："""


# ============ 工具函数 ============
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def norm_file_key(path: str) -> str:
    """用于把 metadata file 和 candidates file 对齐的 key。"""
    if not path:
        return ""
    try:
        p = os.path.abspath(path)
        p = os.path.normpath(p)
        return p.lower()
    except Exception:
        return str(path).strip().lower()


def build_file_meta_map(metadata_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mp = {}
    for it in metadata_results:
        fpath = it.get("file")
        k = norm_file_key(fpath)
        if k:
            mp[k] = it
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
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return None


SPEAKER_PREFIX_RE = re.compile(r"^\s*[\u4e00-\u9fa5]{2,6}\s*[：:]\s*")  # 张三：

def preprocess_raw_text(raw_text: str) -> str:
    t = (raw_text or "").strip()
    t = t.replace("?", "？")
    t = re.sub(r"\s+", " ", t).strip()
    t = SPEAKER_PREFIX_RE.sub("", t).strip()
    return t


def call_deepseek(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY in code.")

    url = f"{DEEPSEEK_BASE_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
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


def _ensure_question_mark(s: str) -> str:
    s = (s or "").strip().replace("?", "？")
    if s and not s.endswith("？"):
        s += "？"
    return s


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

    cleaned_text = preprocess_raw_text(raw_text)

    user_prompt = USER_TEMPLATE.format(
        doc_type=doc_type,
        week=week,
        topics=topics,
        people=people,
        raw_text=cleaned_text,
        context_prev=(context_prev or ""),
        context_next=(context_next or ""),
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_err = None
    for _ in range(MAX_RETRIES):
        try:
            content = call_deepseek(messages, temperature=0.2)
            obj = extract_json_object(content)
            if obj is None:
                raise ValueError(f"JSON parse failed. head={content[:200]!r}")

            # ---- 字段兜底 ----
            if "is_course_question" not in obj:
                obj["is_course_question"] = False
            if "confidence" not in obj:
                obj["confidence"] = 0.0
            if "topic_hint" not in obj or obj["topic_hint"] is None:
                obj["topic_hint"] = []
            if "notes" not in obj or obj["notes"] is None:
                obj["notes"] = ""
            if "split_questions" not in obj or obj["split_questions"] is None:
                obj["split_questions"] = []

            # topic_hint 只能从 topics 里选
            topics_set = set(map(str, topics))
            obj["topic_hint"] = [t for t in obj.get("topic_hint", []) if str(t) in topics_set][:2]

            # notes 限长
            obj["notes"] = (obj.get("notes") or "")[:20]

            # confidence 规整到 [0,1]
            try:
                c = float(obj.get("confidence", 0.0))
            except Exception:
                c = 0.0
            c = max(0.0, min(1.0, c))
            obj["confidence"] = c

            # ---- 规范化问句字段 ----
            if obj.get("is_course_question") is False:
                obj["normalized_question"] = None
                obj["split_questions"] = []
                return obj

            # normalized_question
            nq = obj.get("normalized_question")
            if not nq or not isinstance(nq, str):
                # 不合格：降级为非问题
                obj["is_course_question"] = False
                obj["normalized_question"] = None
                obj["split_questions"] = []
                obj["confidence"] = min(obj["confidence"], 0.3)
                return obj
            obj["normalized_question"] = _ensure_question_mark(nq)

            # split_questions：每条都规整为问句；去重；最多保留 5 条
            sq = obj.get("split_questions", [])
            if not isinstance(sq, list):
                sq = []
            cleaned_sq = []
            seen = set()
            for x in sq:
                if not isinstance(x, str):
                    continue
                q = _ensure_question_mark(x)
                if not q:
                    continue
                if q in seen:
                    continue
                cleaned_sq.append(q)
                seen.add(q)
                if len(cleaned_sq) >= 5:
                    break

            # 如果拆分里包含 normalized_question，就保留；否则把 normalized_question 放进第一条
            if cleaned_sq:
                if obj["normalized_question"] not in cleaned_sq:
                    cleaned_sq = [obj["normalized_question"]] + cleaned_sq
                    cleaned_sq = cleaned_sq[:5]

                # 多意图：confidence 不要给到 0.95 这种满分，做个轻量下调
                obj["confidence"] = min(obj["confidence"], 0.84)

            obj["split_questions"] = cleaned_sq
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
        "split_questions": [],
        "confidence": 0.0,
        "topic_hint": [],
        "notes": (f"failed:{last_err}" if last_err else "failed")[:20],
    }


# ============ 主流程 ============
def main():
    candidates_data = load_json(CANDIDATES_PATH)

    meta_map: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(METADATA_PATH):
        metadata_data = load_json(METADATA_PATH)
        if isinstance(metadata_data, list):
            meta_map = build_file_meta_map(metadata_data)

    results = []
    total = 0
    kept = 0

    for file_item in candidates_data:
        fpath = file_item.get("file")
        if not fpath:
            continue

        meta = meta_map.get(norm_file_key(fpath), {})  # 关键：对齐路径 key
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
    save_json(OUT_PATH, final_out)
    print(f"\nSaved: {OUT_PATH}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
