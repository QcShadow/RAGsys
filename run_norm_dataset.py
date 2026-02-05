# run_norm_dataset.py
# 作用：用“时间戳run目录”管理 norm_dataset 的所有输出，避免 json 覆盖旧结果
import os
import sys
import subprocess
from datetime import datetime

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(THIS_DIR, "norm_dataset")
if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(f"找不到数据集目录：{DATASET_DIR}")

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(THIS_DIR, "runs", "norm_dataset", RUN_ID)
os.makedirs(OUT_DIR, exist_ok=True)

def run_cmd(cmd, cwd=None):
    print("\n[CMD]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

def main():
    print(f"[INFO] dataset_dir = {DATASET_DIR}")
    print(f"[INFO] out_dir     = {OUT_DIR}")

    py = sys.executable  # 当前 python 解释器

    # 1) 元数据抽取（可传 --out）
    run_cmd([py, os.path.join(THIS_DIR, "meta_extract.py"), DATASET_DIR,
             "--out", os.path.join(OUT_DIR, "metadata_results.json")])

    # 2) 疑问候选抽取（可传 --out）
    # run_cmd([py, os.path.join(THIS_DIR, "question_extract_v2.4.py"), DATASET_DIR,
    #          "--out", os.path.join(OUT_DIR, "question_candidates_v2_4.json")])
    run_cmd([py, os.path.join(THIS_DIR, "question_extract_v2_3_fallback_llm_markercontv4.py"), DATASET_DIR,
             "--out", os.path.join(OUT_DIR, "question_candidates_v2_3_fallback_llm_markercontv4.json")])

    # 3) 规范化（用我们新增的“可传参复制版”，避免写死文件名）
    # run_cmd([py, os.path.join(THIS_DIR, "normalized_questions_run.py"),
    #          "--candidates", os.path.join(OUT_DIR, "question_candidates_v2_4.json"),
    #          "--meta", os.path.join(OUT_DIR, "metadata_results.json"),
    #          "--out", os.path.join(OUT_DIR, "normalized_questions_v2_4.json")])
    run_cmd([py, os.path.join(THIS_DIR, "normalized_questions_run.py"),
             "--candidates", os.path.join(OUT_DIR, "question_candidates_v2_3_fallback_llm_markercontv4.json"),
             "--meta", os.path.join(OUT_DIR, "metadata_results.json"),
             "--out", os.path.join(OUT_DIR, "normalized_questions_v2_4.json")])

    # 4) RAG回答（pipeline_answerv2 已支持 --in/--out）
    run_cmd([py, os.path.join(THIS_DIR, "pipeline_answerv2.py"),
             "--in", os.path.join(OUT_DIR, "normalized_questions_v2_4.json"),
             "--out", os.path.join(OUT_DIR, "qa_with_citations_v2.json")])
    # run_cmd([py, os.path.join(THIS_DIR, "pipeline_answer_llmonly.py"),
    #          "--in", "F:\\pycode\\rag\\llm_only_candidates_strict.json",
    #          "--out", os.path.join(OUT_DIR, "qa_with_citations_llmonly.json")])

    print("\n[DONE] 本次运行输出目录：", OUT_DIR)

if __name__ == "__main__":
    main()
