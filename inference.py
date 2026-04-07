# ============================================================
# inference.py — 推論與 Kaggle CSV 生成
# ============================================================
#
# 用法:
#   python inference.py --adapter_dir outputs/gemma-3-4b-it_04071230/final_adapter
#

import argparse
import csv
import re
import os

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
    default=1,
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

# -------------------- 解析 verdict --------------------
VALID_VERDICTS = {"A", "B", "tie", "neither"}

def parse_verdict(generated_text: str, prompt: str) -> str:
    """從生成文本中提取 verdict。"""
    # 移除 prompt 部分，只看生成的新 token
    answer = generated_text[len(prompt):].strip()

    # 嘗試直接匹配
    first_token = answer.split()[0] if answer.split() else ""
    # 去除標點
    first_token_clean = re.sub(r"[^a-zA-Z]", "", first_token)

    if first_token_clean in VALID_VERDICTS:
        return first_token_clean

    # 嘗試在整個回答中搜尋
    answer_lower = answer.lower()
    for v in ["neither", "tie"]:  # 先檢查多字元的
        if v in answer_lower:
            return v
    if answer_lower.startswith("a") or "response a" in answer_lower:
        return "A"
    if answer_lower.startswith("b") or "response b" in answer_lower:
        return "B"

    # fallback
    return "tie"

# -------------------- 推論 --------------------
test_data = load_json(args.test_json)
print(f"[INFO] 測試資料筆數: {len(test_data)}")

results = []
for sample in tqdm(test_data, desc="Inference"):
    prompt = build_prompt(sample["dialog_1"], sample["dialog_2"])
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    verdict = parse_verdict(generated, prompt)
    results.append({"id": sample["id"], "verdict": verdict})

# -------------------- 寫入 CSV --------------------
# 依 id 排序確保順序正確
results.sort(key=lambda x: x["id"])

with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "verdict"])
    writer.writeheader()
    writer.writerows(results)

print(f"[INFO] 已生成 {len(results)} 筆預測")
print(f"[INFO] CSV 已儲存至: {args.output_csv}")

# 統計 verdict 分佈
from collections import Counter
dist = Counter(r["verdict"] for r in results)
print(f"[INFO] Verdict 分佈: {dict(dist)}")
