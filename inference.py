# ============================================================
# inference.py — 序列分類推論 + TTA (softmax 平均) + Kaggle CSV 生成
# (Gemma4 無原生 SeqCls，用 CausalLM + 自訂分類頭)
# ============================================================
#
# 用法:
#   python inference.py --adapter_dir outputs/gemma-4-E4B-it_04071230/final_adapter
#   python inference.py --adapter_dir outputs/.../final_adapter --batch_size 4
#

import argparse
import csv
import gc
import os
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from peft import PeftModel
from tqdm import tqdm
from collections import Counter

import config
from dataset import build_prompt, load_json, ID2LABEL

# ============================================================
# 分類 forward (與 train.py 相同)
# ============================================================
def _cls_forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
    _allowed = ("position_ids", "past_key_values", "inputs_embeds", "cache_position")
    body_kwargs = {k: v for k, v in kwargs.items() if k in _allowed}

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
        **body_kwargs,
    )
    hidden_states = outputs.last_hidden_state

    if attention_mask is not None:
        seq_len = attention_mask.shape[1]
        sequence_lengths = seq_len - 1 - attention_mask.flip(dims=[1]).argmax(dim=1)
    else:
        sequence_lengths = input_ids.shape[1] - 1

    batch_size = hidden_states.shape[0]
    pooled = hidden_states[
        torch.arange(batch_size, device=hidden_states.device),
        sequence_lengths,
    ]
    logits = self.score(pooled).float()

    loss = None
    if labels is not None:
        loss = F.cross_entropy(logits, labels)

    return SequenceClassifierOutputWithPast(loss=loss, logits=logits)

# -------------------- 解析命令列 --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--adapter_dir", type=str, required=True,
                    help="微調後的 adapter 目錄")
parser.add_argument("--test_json", type=str, default=config.TEST_JSON)
parser.add_argument("--output_csv", type=str, default=None,
                    help="輸出 CSV 路徑，預設存在 adapter_dir 的父目錄")
parser.add_argument("--batch_size", type=int, default=16, help="推論 batch size")
args = parser.parse_args()

# -------------------- 輸出路徑 --------------------
if args.output_csv is None:
    parent = os.path.dirname(args.adapter_dir.rstrip("/"))
    args.output_csv = os.path.join(parent, "submission.csv")

# -------------------- 載入模型 --------------------
print(f"[INFO] 載入 base model: {config.MODEL_NAME}")
attn_impl = "sdpa"

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
        attn_implementation=attn_impl,
    )
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )

# 刪除不需要的多模態 Encoder
for attr in ("vision_tower", "embed_vision", "audio_tower", "embed_audio"):
    if hasattr(base_model.model, attr):
        delattr(base_model.model, attr)
        print(f"[INFO] 已刪除 {attr}")
torch.cuda.empty_cache()
gc.collect()

# 加掛分類頭 & monkey-patch forward (PEFT 載入時會覆蓋 score 權重)
hidden_size = (base_model.config.text_config.hidden_size
               if hasattr(base_model.config, "text_config")
               else base_model.config.hidden_size)
base_model.score = nn.Linear(hidden_size, config.NUM_LABELS, bias=False)
base_model.score = base_model.score.to(device=base_model.device, dtype=torch.bfloat16)
base_model.forward = types.MethodType(_cls_forward, base_model)

print(f"[INFO] 載入 adapter: {args.adapter_dir}")
model = PeftModel.from_pretrained(base_model, args.adapter_dir)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # 批次推論使用 left padding

# ============================================================
# 批次推論核心函式 — 回傳 softmax 機率
# ============================================================
def run_batch_inference(prompts: list[str]) -> np.ndarray:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits.float(), dim=-1)

    return probs.cpu().numpy()

# ============================================================
# 載入測試資料 & 建立 prompts
# ============================================================
test_data = load_json(args.test_json)
print(f"[INFO] 測試資料筆數: {len(test_data)}")

use_tta = config.TTA_ENABLED
print(f"[INFO] TTA: {use_tta}")

normal_prompts = [
    build_prompt(sample["dialog_1"], sample["dialog_2"], tokenizer)
    for sample in test_data
]
if use_tta:
    swapped_prompts = [
        build_prompt(sample["dialog_2"], sample["dialog_1"], tokenizer)
        for sample in test_data
    ]

# ============================================================
# 執行批次推論 — 正常順序
# ============================================================
batch_size = args.batch_size
all_probs = []

print("[INFO] 推論中 (正常順序)...")
for i in tqdm(range(0, len(normal_prompts), batch_size), desc="Normal"):
    batch = normal_prompts[i : i + batch_size]
    probs = run_batch_inference(batch)
    all_probs.append(probs)

probs_normal = np.concatenate(all_probs, axis=0)  # [N, 4]

# ============================================================
# 執行批次推論 — TTA 反順序
# ============================================================
if use_tta:
    all_probs_swap = []
    print("[INFO] 推論中 (TTA 反順序)...")
    for i in tqdm(range(0, len(swapped_prompts), batch_size), desc="TTA Swap"):
        batch = swapped_prompts[i : i + batch_size]
        probs = run_batch_inference(batch)
        all_probs_swap.append(probs)

    probs_swap = np.concatenate(all_probs_swap, axis=0)  # [N, 4]

    # 關鍵對齊：反順序中 A(idx=0) 和 B(idx=1) 意義互換
    probs_swap_aligned = probs_swap.copy()
    probs_swap_aligned[:, 0] = probs_swap[:, 1]  # 原 B -> 對齊後 A
    probs_swap_aligned[:, 1] = probs_swap[:, 0]  # 原 A -> 對齊後 B

    final_probs = (probs_normal + probs_swap_aligned) / 2.0
else:
    final_probs = probs_normal

# ============================================================
# Argmax 取最終 verdict
# ============================================================
pred_ids = np.argmax(final_probs, axis=-1)

results = []
for idx, sample in enumerate(test_data):
    verdict = ID2LABEL[pred_ids[idx]]
    results.append({"id": sample["id"], "verdict": verdict})

results.sort(key=lambda x: x["id"])

with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "verdict"])
    writer.writeheader()
    writer.writerows(results)

print(f"[INFO] 已生成 {len(results)} 筆預測")
print(f"[INFO] CSV 已儲存至: {args.output_csv}")

dist = Counter(r["verdict"] for r in results)
print(f"[INFO] 最終 Verdict 分佈: {dict(dist)}")

if use_tta:
    normal_preds = np.argmax(probs_normal, axis=-1)
    consistent = np.sum(normal_preds == pred_ids)
    print(f"[INFO] TTA 前後一致: {consistent}/{len(test_data)} "
          f"({consistent/len(test_data)*100:.1f}%)")
