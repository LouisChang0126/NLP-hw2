# ============================================================
# generate_cot.py — 策略2: 本地逆向思維鏈生成
# ============================================================
#
# 用法 (在正式微調前執行):
#   python generate_cot.py
#
# 此腳本使用 base model 為每筆 train 資料生成 rationale，
# 結果存入 data/train_cot.json，供 train.py 使用。
# ============================================================

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

import config
from dataset import flatten_dialog, load_json

# -------------------- 設定 --------------------
OUTPUT_PATH = config.AUG_REVERSE_COT_FILE
MAX_NEW_TOKENS = 150

VERDICT_EXPLAIN = {
    "A": "the human preferred Response A over Response B",
    "B": "the human preferred Response B over Response A",
    "tie": "the human considered both responses equally good",
    "neither": "the human considered both responses equally bad",
}

def build_cot_prompt(dialog_1, dialog_2, verdict, tokenizer):
    """建立要求模型生成理由的 prompt，使用 chat template。"""
    flat_1 = flatten_dialog(dialog_1)
    flat_2 = flatten_dialog(dialog_2)
    explanation = VERDICT_EXPLAIN[verdict]
    user_content = (
        f"You are an objective AI evaluator. Below are two AI assistant responses "
        f"to the same user query. In this comparison, {explanation}.\n\n"
        f"### Response A\n{flat_1}\n\n"
        f"### Response B\n{flat_2}\n\n"
        f"Briefly explain why this judgment makes sense in 2-3 sentences:"
    )
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def is_valid_rationale(text: str) -> bool:
    """簡單過濾：長度合理且非亂碼。"""
    text = text.strip()
    if len(text) < 20 or len(text) > 1000:
        return False
    # 非 ASCII 字元佔比過高表示亂碼
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    if ascii_ratio < 0.5:
        return False
    return True

# -------------------- 載入模型 --------------------
print(f"[INFO] 載入模型: {config.MODEL_NAME}")
if config.USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
model.eval()

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------- 生成 --------------------
train_data = load_json(config.TRAIN_JSON)
results = []
success = 0

for sample in tqdm(train_data, desc="Generating CoT"):
    prompt = build_cot_prompt(
        sample["dialog_1"], sample["dialog_2"], sample["verdict"], tokenizer
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=config.MAX_SEQ_LENGTH
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    rationale = generated[len(prompt):].strip()

    # 過濾
    if is_valid_rationale(rationale):
        success += 1
    else:
        rationale = ""

    results.append({
        "id": sample["id"],
        "verdict": sample["verdict"],
        "rationale": rationale,
    })

# -------------------- 儲存 --------------------
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"[INFO] 完成！成功生成 {success}/{len(train_data)} 筆理由")
print(f"[INFO] 已儲存至: {OUTPUT_PATH}")
