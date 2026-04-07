# ============================================================
# inference.py — 推論與 Kaggle CSV 生成
# ============================================================
#
# 用法:
#   python inference.py --adapter_dir outputs/gemma-3-4b-it_04071230/final_adapter
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
from dataset import build_prompt, load_json

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
# 推論時必須使用 left padding，才能正確做 batch generation
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

    # CoT 模式：verdict 在最後一行
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
    answer_lower = answer.lower()
    last_lower = last_line.lower()

    for v in ["neither", "tie"]:
        if v in last_lower:
            return v
    if last_lower.startswith("a") or "response a" in last_lower:
        return "A"
    if last_lower.startswith("b") or "response b" in last_lower:
        return "B"

    # 全文搜尋 fallback
    for v in ["neither", "tie"]:
        if v in answer_lower:
            return v
    if "response a" in answer_lower:
        return "A"
    if "response b" in answer_lower:
        return "B"

    return "tie"

# -------------------- 批次推論 --------------------
test_data = load_json(args.test_json)
print(f"[INFO] 測試資料筆數: {len(test_data)}")
print(f"[INFO] Batch size: {args.batch_size}, max_new_tokens: {max_new_tokens}")

results = []
batch_size = args.batch_size

for i in tqdm(range(0, len(test_data), batch_size), desc="Batched Inference"):
    batch_samples = test_data[i : i + batch_size]
    prompts = [
        build_prompt(s["dialog_1"], s["dialog_2"], tokenizer)
        for s in batch_samples
    ]

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

    for j, output in enumerate(outputs):
        # 只解碼新生成的 token，避免 chat template special token 造成字串切片錯誤
        new_tokens = output[input_len:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
        verdict = parse_verdict(answer)
        results.append({"id": batch_samples[j]["id"], "verdict": verdict})

# -------------------- 寫入 CSV --------------------
results.sort(key=lambda x: x["id"])

with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "verdict"])
    writer.writeheader()
    writer.writerows(results)

print(f"[INFO] 已生成 {len(results)} 筆預測")
print(f"[INFO] CSV 已儲存至: {args.output_csv}")

dist = Counter(r["verdict"] for r in results)
print(f"[INFO] Verdict 分佈: {dict(dist)}")
