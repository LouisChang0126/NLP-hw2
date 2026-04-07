# ============================================================
# inference.py — 推論與 Kaggle CSV 生成 (支援 TTA 多數決)
# ============================================================
#
# 用法:
#   python inference.py --adapter_dir outputs/gemma-4-E4B-it_04071230/final_adapter
#   python inference.py --adapter_dir outputs/.../final_adapter --batch_size 4
#

import argparse
import csv
import re
import os
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

import config
from dataset import build_prompt, load_json, SWAP_VERDICT

# -------------------- 解析命令列 --------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--adapter_dir",
    type=str,
    required=True,
    help="微調後的 adapter 目錄 (e.g. outputs/.../final_adapter)",
)
parser.add_argument(
    "--test_json",
    type=str,
    default=config.TEST_JSON,
)
parser.add_argument(
    "--output_csv",
    type=str,
    default=None,
    help="輸出 CSV 路徑，預設存在 adapter_dir 的父目錄",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
    help="推論 batch size",
)
args = parser.parse_args()

# -------------------- 輸出路徑 --------------------
if args.output_csv is None:
    parent = os.path.dirname(args.adapter_dir.rstrip("/"))
    args.output_csv = os.path.join(parent, "submission.csv")

# -------------------- 載入模型 --------------------
print(f"[INFO] 載入 base model: {config.MODEL_NAME}")
if config.USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

print(f"[INFO] 載入 adapter: {args.adapter_dir}")
model = PeftModel.from_pretrained(base_model, args.adapter_dir)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# -------------------- 推論長度 --------------------
max_new_tokens = 250 if config.AUG_REVERSE_COT else 5

# -------------------- 解析 verdict --------------------
VALID_VERDICTS = {"A", "B", "tie", "neither"}

def parse_verdict(answer: str) -> str:
    """從生成的新 token 文字中提取 verdict。支援 CoT 模式（答案在最後）。"""
    answer = answer.strip()

    if not answer:
        return "tie"

    lines = [l.strip() for l in answer.split("\n") if l.strip()]
    last_line = lines[-1] if lines else answer

    # 嘗試直接匹配最後一行的第一個 token
    first_token = last_line.split()[0] if last_line.split() else ""
    first_token_clean = re.sub(r"[^a-zA-Z]", "", first_token)
    if first_token_clean in VALID_VERDICTS:
        return first_token_clean

    # 也嘗試第一行（非 CoT 模式）
    first_line = lines[0] if lines else answer
    first_word = first_line.split()[0] if first_line.split() else ""
    first_word_clean = re.sub(r"[^a-zA-Z]", "", first_word)
    if first_word_clean in VALID_VERDICTS:
        return first_word_clean

    # 從整段回答中搜尋（優先從後面找）
    last_lower = last_line.lower()
    for v in ["neither", "tie"]:
        if v in last_lower:
            return v
    if last_lower.startswith("a") or "response a" in last_lower:
        return "A"
    if last_lower.startswith("b") or "response b" in last_lower:
        return "B"

    # 全文搜尋 fallback
    answer_lower = answer.lower()
    for v in ["neither", "tie"]:
        if v in answer_lower:
            return v
    if "response a" in answer_lower:
        return "A"
    if "response b" in answer_lower:
        return "B"

    return "tie"

# ============================================================
# 批次推論核心函式
# ============================================================
def run_batch_inference(prompts: list[str]) -> list[str]:
    """對一批 prompt 執行生成，回傳解析後的 verdict 列表。"""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    verdicts = []
    for output in outputs:
        new_tokens = output[input_len:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
        verdicts.append(parse_verdict(answer))
    return verdicts

# ============================================================
# TTA: 建立所有推論視角
# ============================================================
test_data = load_json(args.test_json)
print(f"[INFO] 測試資料筆數: {len(test_data)}")

use_tta = config.TTA_ENABLED
tta_templates = config.TTA_PROMPT_TEMPLATES if use_tta else [0]
tta_swap = config.TTA_POSITION_SWAP if use_tta else False

# 建立 (sample_idx, prompt, is_swapped) 的完整推論清單
inference_jobs: list[tuple[int, str, bool]] = []

for idx, sample in enumerate(test_data):
    for tid in tta_templates:
        # 原順序
        prompt = build_prompt(sample["dialog_1"], sample["dialog_2"], tokenizer, tid)
        inference_jobs.append((idx, prompt, False))

        # 反順序
        if tta_swap:
            prompt_swap = build_prompt(
                sample["dialog_2"], sample["dialog_1"], tokenizer, tid
            )
            inference_jobs.append((idx, prompt_swap, True))

num_views = len(tta_templates) * (2 if tta_swap else 1)
print(f"[INFO] TTA: {use_tta} | 模板數: {len(tta_templates)} | "
      f"位置交換: {tta_swap} | 每筆視角數: {num_views}")
print(f"[INFO] 總推論次數: {len(inference_jobs)}, batch_size: {args.batch_size}")

# ============================================================
# 執行批次推論
# ============================================================
# 收集每筆 sample 的所有投票
votes: dict[int, list[str]] = {i: [] for i in range(len(test_data))}
batch_size = args.batch_size

for i in tqdm(range(0, len(inference_jobs), batch_size), desc="TTA Inference"):
    batch_jobs = inference_jobs[i : i + batch_size]
    batch_prompts = [job[1] for job in batch_jobs]

    batch_verdicts = run_batch_inference(batch_prompts)

    for (sample_idx, _, is_swapped), verdict in zip(batch_jobs, batch_verdicts):
        if is_swapped:
            # 反順序的結果需要翻轉 A↔B，tie/neither 不變
            verdict = SWAP_VERDICT[verdict]
        votes[sample_idx].append(verdict)

# ============================================================
# 多數決 (Majority Vote)
# ============================================================
def majority_vote(vote_list: list[str]) -> str:
    """多數決。平手時優先選 A/B (避免模型預設偏 tie)。"""
    count = Counter(vote_list)
    max_count = max(count.values())
    candidates = [v for v, c in count.items() if c == max_count]

    if len(candidates) == 1:
        return candidates[0]

    # 平手打破規則：A/B 優先於 tie/neither
    priority = ["A", "B", "tie", "neither"]
    for p in priority:
        if p in candidates:
            return p
    return candidates[0]

results = []
for idx, sample in enumerate(test_data):
    final_verdict = majority_vote(votes[idx])
    results.append({"id": sample["id"], "verdict": final_verdict})

# -------------------- 寫入 CSV --------------------
results.sort(key=lambda x: x["id"])

with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "verdict"])
    writer.writeheader()
    writer.writerows(results)

print(f"[INFO] 已生成 {len(results)} 筆預測")
print(f"[INFO] CSV 已儲存至: {args.output_csv}")

# -------------------- 統計 --------------------
dist = Counter(r["verdict"] for r in results)
print(f"[INFO] 最終 Verdict 分佈: {dict(dist)}")

# TTA 投票一致性統計
if use_tta:
    unanimous = sum(1 for v in votes.values() if len(set(v)) == 1)
    print(f"[INFO] TTA 全票一致: {unanimous}/{len(test_data)} "
          f"({unanimous/len(test_data)*100:.1f}%)")

    # 各視角的原始分佈
    all_raw = [v for vlist in votes.values() for v in vlist]
    raw_dist = Counter(all_raw)
    print(f"[INFO] TTA 原始投票分佈 (含翻轉後): {dict(raw_dist)}")
