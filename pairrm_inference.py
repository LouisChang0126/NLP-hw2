# ============================================================
# pairrm_inference.py — 直接使用 llm-blender/PairRM 推論，無需訓練
# ============================================================
#
# 4 個 class：
#   A       — Response 1 (dialog_1) is better
#   B       — Response 2 (dialog_2) is better
#   tie     — Both responses are equally good
#   neither — Both responses are equally bad
#
# 策略：
#   1. PairRM compare() 給出 logit = log P(A > B)
#      → sigmoid 離 0.5 夠遠 → 判 A 或 B
#      → sigmoid 接近 0.5   → 「模糊區」，再用品質代理判 tie/neither
#   2. 品質代理 = (回答 A 字數 + 回答 B 字數) / 2
#      → 平均字數 >= neither_word_threshold → tie  (兩者都夠完整)
#      → 平均字數 <  neither_word_threshold → neither (兩者都太短/差)
#
# 安裝依賴：
#   pip install llm-blender
#
# 用法：
#   python pairrm_inference.py
#   python pairrm_inference.py --output_csv outputs/pairrm/submission.csv
#   python pairrm_inference.py --eval_train          # 在 train.json 驗算準確率
#   python pairrm_inference.py --eval_train --margin_threshold 0.2 --neither_word_threshold 40
#

import argparse
import csv
import json
import os
from collections import Counter

import torch

# -------------------- 解析命令列 --------------------
parser = argparse.ArgumentParser(description="PairRM zero-shot 4-class inference")
parser.add_argument("--test_json",  type=str, default="data/test.json")
parser.add_argument("--train_json", type=str, default="data/train.json")
parser.add_argument("--output_csv", type=str, default="outputs/pairrm/submission.csv")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument(
    "--margin_threshold", type=float, default=0.49,
    help="sigmoid(logit) 距 0.5 超過此值才判為 A 或 B（預設 0.49）",
)
parser.add_argument(
    "--neither_word_threshold", type=int, default=30,
    help="模糊區中，回答平均字數低於此值判為 neither，否則判為 tie（預設 30）",
)
parser.add_argument(
    "--eval_train", action="store_true",
    help="在 train.json 上評估準確率（不輸出 CSV，用於調整閾值）",
)
args = parser.parse_args()

# -------------------- 工具函式 --------------------
def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_instruction(dialog: list[dict]) -> str:
    """取出 user 輪的全部內容，作為 PairRM 的 instruction。"""
    return "\n".join(m["content"] for m in dialog if m["role"] == "user")

def extract_response(dialog: list[dict]) -> str:
    """取出 assistant 輪的全部內容，作為 PairRM 的候選回答。"""
    return "\n".join(m["content"] for m in dialog if m["role"] == "assistant")

def word_count(text: str) -> int:
    return len(text.split())

def score_to_verdict(
    logit: float,
    resp_a: str,
    resp_b: str,
    margin_threshold: float,
    neither_word_threshold: int,
) -> str:
    """
    將 PairRM logit 與兩個回答的品質代理轉為 4-class verdict。

    Args:
        logit: PairRM 回傳的原始 logit（正 = A 較好，負 = B 較好）
        resp_a / resp_b: 兩個 assistant 的回答文字
        margin_threshold: sigmoid 距 0.5 的判斷邊界
        neither_word_threshold: 平均字數低於此值 → neither
    """
    prob_a = torch.sigmoid(torch.tensor(logit, dtype=torch.float32)).item()

    # --- 步驟 1：A vs B ---
    if prob_a >= 0.5 + margin_threshold:
        return "A"
    if prob_a <= 0.5 - margin_threshold:
        return "B"

    # --- 步驟 2：tie vs neither（品質代理） ---
    avg_words = (word_count(resp_a) + word_count(resp_b)) / 2
    if avg_words >= neither_word_threshold:
        return "tie"
    return "neither"

# -------------------- 載入 PairRM --------------------
# llm-blender 0.0.2 與 transformers v5 有兩個不相容問題，在 import 前統一修補：
#
# 1. TRANSFORMERS_CACHE 被移除，改用 HF_HUB_CACHE
# 2. batch_encode_plus 被移除，改用 tokenizer.__call__
#
import transformers.utils.hub as _transformers_hub
if not hasattr(_transformers_hub, "TRANSFORMERS_CACHE"):
    from huggingface_hub.constants import HF_HUB_CACHE as _cache
    _transformers_hub.TRANSFORMERS_CACHE = _cache

try:
    import llm_blender
except ImportError:
    raise ImportError(
        "請先安裝 llm-blender：\n"
        "  pip install git+https://github.com/yuchenlin/LLM-Blender.git"
    )

# patch 2: batch_encode_plus → tokenizer.__call__
import llm_blender.pair_ranker.collator as _collator

def _encode_texts_patched(texts, tokenizer, max_length=None):
    p = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
    )
    return p["input_ids"], p["attention_mask"]

_collator.encode_texts = _encode_texts_patched

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] 使用裝置: {device}")
print("[INFO] 載入 llm-blender/PairRM ...")
blender = llm_blender.Blender()
blender.loadranker("llm-blender/PairRM", device=device)
print("[INFO] PairRM 載入完成")
print(f"[INFO] margin_threshold={args.margin_threshold}, "
      f"neither_word_threshold={args.neither_word_threshold}\n")

# ============================================================
# 核心推論函式
# ============================================================
def run_inference(data: list[dict]) -> list[dict]:
    """
    對 data 中每筆樣本推論，回傳 [{"id": ..., "verdict": ...}, ...]。
    Verdict 為 A / B / tie / neither 之一。
    compare() 內建 batch_size 與進度條，不需手動分批。
    mode='[A,B]+[B,A]' 預設對兩個方向都算一次再平均，自動去除位置偏差。
    """
    instructions = [extract_instruction(s["dialog_1"]) for s in data]
    cands_a      = [extract_response(s["dialog_1"]) for s in data]
    cands_b      = [extract_response(s["dialog_2"]) for s in data]

    # return_logits=True：回傳原始 logit（正 = A 較好，負 = B 較好）
    logits = blender.compare(
        instructions, cands_a, cands_b,
        return_logits=True,
        batch_size=args.batch_size,
    )

    results = []
    for sample, logit, ra, rb in zip(data, logits, cands_a, cands_b):
        verdict = score_to_verdict(
            float(logit), ra, rb,
            args.margin_threshold,
            args.neither_word_threshold,
        )
        results.append({"id": sample["id"], "verdict": verdict})

    return results

# ============================================================
# 主邏輯
# ============================================================
if args.eval_train:
    # ---- 在訓練集上評估，用來調整閾值 ----
    print(f"[INFO] 讀取訓練資料: {args.train_json}")
    train_data = load_json(args.train_json)
    print(f"[INFO] 訓練資料筆數: {len(train_data)}")

    preds = run_inference(train_data)

    correct = sum(
        p["verdict"] == s["verdict"]
        for p, s in zip(preds, train_data)
    )
    total = len(train_data)
    print(f"\n[RESULT] 整體準確率: {correct}/{total} = {correct/total*100:.2f}%")

    # 各 class 準確率
    class_correct: dict[str, int] = Counter()
    class_total:   dict[str, int] = Counter()
    for p, s in zip(preds, train_data):
        label = s["verdict"]
        class_total[label] += 1
        if p["verdict"] == label:
            class_correct[label] += 1
    for label in ["A", "B", "tie", "neither"]:
        ct = class_total[label]
        cc = class_correct[label]
        print(f"  {label:7s}: {cc}/{ct} = {cc/ct*100:.1f}%" if ct else f"  {label}: 無樣本")

    pred_dist = Counter(p["verdict"] for p in preds)
    gold_dist = Counter(s["verdict"] for s in train_data)
    print(f"\n[RESULT] 預測分佈: {dict(pred_dist)}")
    print(f"[RESULT] 真實分佈: {dict(gold_dist)}")

else:
    # ---- 在測試集上推論並輸出 CSV ----
    print(f"[INFO] 讀取測試資料: {args.test_json}")
    test_data = load_json(args.test_json)
    print(f"[INFO] 測試資料筆數: {len(test_data)}")

    results = run_inference(test_data)
    results.sort(key=lambda x: x["id"])

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "verdict"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[INFO] 已生成 {len(results)} 筆預測")
    print(f"[INFO] CSV 儲存至: {args.output_csv}")

    dist = Counter(r["verdict"] for r in results)
    print(f"[INFO] Verdict 分佈: {dict(dist)}")
