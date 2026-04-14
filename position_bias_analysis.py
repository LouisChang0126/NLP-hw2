# ============================================================
# position_bias_analysis.py — Q3: Position Bias Analysis
# ============================================================
#
# 實驗設計：
#   1. 從 train.json 切出 validation set（與 train.py 相同的分層抽樣）
#   2. 用訓練好的模型分別對 val set 做：
#      (a) 原始順序推論（dialog_1 在前，dialog_2 在後）
#      (b) 交換順序推論（dialog_2 在前，dialog_1 在後）
#   3. 分析 position bias：
#      - 預測分佈（A vs B vs tie vs neither）
#      - 一致性：交換位置後，A/B 預測是否正確反轉
#      - 各 class 的準確率對比
#   4. 評估 TTA debiasing 的效果
#
# 用法:
#   python position_bias_analysis.py \
#       --adapter_dir outputs/Qwen3-8B_04112128/final_adapter \
#       --batch_size 16
#

import argparse
import gc
import types
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

import config
from dataset import (
    build_prompt, load_json, load_train_val,
    LABEL2ID, ID2LABEL, SWAP_VERDICT,
)
from model_head import build_head


# ============================================================
# 兼容舊版 JudgeHead（只有 norm + out）
# ============================================================
class SimpleJudgeHead(torch.nn.Module):
    """與訓練時的 JudgeHead 結構一致：LayerNorm + Linear。"""
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.out = torch.nn.Linear(hidden_size, num_labels, bias=False)

    def forward(self, x):
        return self.out(self.norm(x))


def build_head_from_adapter(adapter_dir, hidden_size, num_labels, dropout=0.1):
    """根據 adapter 中實際儲存的 key 決定用哪種分類頭。"""
    from safetensors import safe_open
    import os
    sf_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    f = safe_open(sf_path, framework="pt")
    keys = [k for k in f.keys() if "score" in k]
    # 如果有 score.out.weight 而沒有 score.dense.weight → 舊版 SimpleJudgeHead
    has_dense = any("score.dense" in k for k in keys)
    has_out = any("score.out.weight" in k for k in keys)
    if has_out and not has_dense:
        print(f"[INFO] Detected legacy SimpleJudgeHead in adapter (keys: {keys})")
        return SimpleJudgeHead(hidden_size, num_labels)
    else:
        return build_head(config.HEAD_TYPE, hidden_size, num_labels, dropout=dropout)


# ============================================================
# 分類 forward（與 train.py / inference.py 完全相同）
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


# ============================================================
# 主程式
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Q3: Position Bias Analysis")
    parser.add_argument("--adapter_dir", type=str, required=True,
                        help="微調後的 adapter 目錄")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--template_id", type=int, default=0,
                        help="Prompt template ID (0-3)")
    args = parser.parse_args()

    # -------------------- 載入 Validation Set --------------------
    print("=" * 60)
    print("Q3: Position Bias Analysis")
    print("=" * 60)

    _, val_data = load_train_val()
    print(f"\n[DATA] Validation set: {len(val_data)} samples")

    gt_labels = [sample["verdict"] for sample in val_data]
    gt_dist = Counter(gt_labels)
    print(f"[DATA] Ground-truth distribution: {dict(gt_dist)}")

    # -------------------- 載入模型 --------------------
    print(f"\n[MODEL] Loading base model: {config.MODEL_NAME}")
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

    for attr in ("vision_tower", "embed_vision", "audio_tower", "embed_audio"):
        if hasattr(base_model.model, attr):
            delattr(base_model.model, attr)
    torch.cuda.empty_cache()
    gc.collect()

    hidden_size = (base_model.config.text_config.hidden_size
                   if hasattr(base_model.config, "text_config")
                   else base_model.config.hidden_size)
    base_model.score = build_head_from_adapter(
        args.adapter_dir, hidden_size, config.NUM_LABELS, dropout=config.HEAD_DROPOUT,
    )
    base_model.score = base_model.score.to(device=base_model.device, dtype=torch.bfloat16)
    base_model.forward = types.MethodType(_cls_forward, base_model)

    print(f"[MODEL] Loading adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # 與訓練時一致

    # ============================================================
    # 批次推論函式
    # ============================================================
    def run_batch_inference(prompts: list[str]) -> np.ndarray:
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=config.MAX_SEQ_LENGTH,
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits.float(), dim=-1)
        return probs.cpu().numpy()

    def run_all(prompts: list[str], desc: str) -> np.ndarray:
        all_probs = []
        for i in tqdm(range(0, len(prompts), args.batch_size), desc=desc):
            batch = prompts[i : i + args.batch_size]
            probs = run_batch_inference(batch)
            all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)

    # ============================================================
    # 建立 Prompts — 原始順序 & 交換順序
    # ============================================================
    tid = args.template_id
    normal_prompts = [
        build_prompt(s["dialog_1"], s["dialog_2"], tokenizer, tid) for s in val_data
    ]
    swapped_prompts = [
        build_prompt(s["dialog_2"], s["dialog_1"], tokenizer, tid) for s in val_data
    ]

    # ============================================================
    # 推論
    # ============================================================
    print("\n[INFERENCE] Running normal order...")
    probs_normal = run_all(normal_prompts, "Normal")

    print("[INFERENCE] Running swapped order...")
    probs_swap = run_all(swapped_prompts, "Swapped")

    # ============================================================
    # 對齊 swapped 的 A/B
    # ============================================================
    probs_swap_aligned = probs_swap.copy()
    probs_swap_aligned[:, 0] = probs_swap[:, 1]  # swap 的 B -> 對齊後的 A
    probs_swap_aligned[:, 1] = probs_swap[:, 0]  # swap 的 A -> 對齊後的 B

    # TTA: 平均兩個方向的機率
    probs_tta = (probs_normal + probs_swap_aligned) / 2.0

    # 取 argmax
    preds_normal = np.argmax(probs_normal, axis=-1)
    preds_swap_raw = np.argmax(probs_swap, axis=-1)  # 未對齊的 (swap 空間)
    preds_swap_aligned = np.argmax(probs_swap_aligned, axis=-1)  # 對齊後
    preds_tta = np.argmax(probs_tta, axis=-1)

    gt_ids = np.array([LABEL2ID[v] for v in gt_labels])

    # ============================================================
    # 分析 1: 預測分佈
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Prediction Distribution")
    print("=" * 60)

    def show_dist(name, preds):
        labels = [ID2LABEL[p] for p in preds]
        dist = Counter(labels)
        total = len(labels)
        print(f"\n  [{name}]")
        for v in ["A", "B", "tie", "neither"]:
            cnt = dist.get(v, 0)
            pct = cnt / total * 100
            print(f"    {v:>7s}: {cnt:4d} ({pct:5.1f}%)")
        return dist

    print("\n  [Ground Truth]")
    for v in ["A", "B", "tie", "neither"]:
        cnt = gt_dist.get(v, 0)
        pct = cnt / len(gt_labels) * 100
        print(f"    {v:>7s}: {cnt:4d} ({pct:5.1f}%)")

    dist_normal = show_dist("Normal Order", preds_normal)
    dist_swap_raw = show_dist("Swapped Order (raw, in swap space)", preds_swap_raw)
    dist_swap_aligned = show_dist("Swapped Order (aligned)", preds_swap_aligned)
    dist_tta = show_dist("TTA (debiased)", preds_tta)

    # ============================================================
    # 分析 2: Position Bias 量化指標
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Position Bias Metrics")
    print("=" * 60)

    # 2a. 在 normal order 中，模型傾向 A(第一位) 還是 B(第二位)?
    n_pred_A_normal = np.sum(preds_normal == 0)
    n_pred_B_normal = np.sum(preds_normal == 1)
    n_gt_A = np.sum(gt_ids == 0)
    n_gt_B = np.sum(gt_ids == 1)
    total = len(gt_ids)

    print(f"\n  Normal order prediction: A={n_pred_A_normal} ({n_pred_A_normal/total*100:.1f}%), "
          f"B={n_pred_B_normal} ({n_pred_B_normal/total*100:.1f}%)")
    print(f"  Ground truth:            A={n_gt_A} ({n_gt_A/total*100:.1f}%), "
          f"B={n_gt_B} ({n_gt_B/total*100:.1f}%)")

    # 2b. 在 swapped order 中（raw swap space），模型是否也偏好 A (即第一位)?
    n_pred_A_swap = np.sum(preds_swap_raw == 0)
    n_pred_B_swap = np.sum(preds_swap_raw == 1)
    print(f"\n  Swapped order prediction (raw): A={n_pred_A_swap} ({n_pred_A_swap/total*100:.1f}%), "
          f"B={n_pred_B_swap} ({n_pred_B_swap/total*100:.1f}%)")
    print(f"  (In swapped order, 'A' means the model chose the FIRST position,")
    print(f"   which is actually the original dialog_2)")

    # 2c. Position Bias Score
    # 如果沒有 position bias，normal A% 應該 ≈ swap B% (raw)
    # Position bias = 模型選擇第一位置的比率 - 真實第一位置(A)的比率
    first_position_rate_normal = n_pred_A_normal / total
    first_position_rate_swap = n_pred_A_swap / total  # raw swap space 的 A = 第一位置
    avg_first_position_rate = (first_position_rate_normal + first_position_rate_swap) / 2
    true_first_rate = n_gt_A / total

    print(f"\n  --- Position Bias Score ---")
    print(f"  Model picks 1st position (normal order): {first_position_rate_normal*100:.1f}%")
    print(f"  Model picks 1st position (swapped order): {first_position_rate_swap*100:.1f}%")
    print(f"  Average 1st-position preference:          {avg_first_position_rate*100:.1f}%")
    print(f"  True 1st-position (A) rate:               {true_first_rate*100:.1f}%")
    print(f"  Position bias (avg_1st - true_1st):       {(avg_first_position_rate - true_first_rate)*100:+.1f}%")

    # ============================================================
    # 分析 3: 一致性分析（交換後 A/B 是否正確翻轉）
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Consistency Analysis")
    print("=" * 60)

    # 完全一致：normal 和 swap_aligned 預測相同
    consistent = np.sum(preds_normal == preds_swap_aligned)
    print(f"\n  Consistent predictions (normal == swap_aligned): "
          f"{consistent}/{total} ({consistent/total*100:.1f}%)")

    # 按 class 分析一致性
    print(f"\n  Per-class consistency:")
    for cls_name, cls_id in LABEL2ID.items():
        mask = (preds_normal == cls_id) | (preds_swap_aligned == cls_id)
        if mask.sum() == 0:
            continue
        both_agree = ((preds_normal == cls_id) & (preds_swap_aligned == cls_id)).sum()
        either = mask.sum()
        print(f"    {cls_name:>7s}: {both_agree}/{either} agree "
              f"({both_agree/either*100:.1f}%)")

    # 不一致的樣本分析：哪些 class 最容易被翻轉
    inconsistent_mask = preds_normal != preds_swap_aligned
    n_inconsistent = inconsistent_mask.sum()
    print(f"\n  Inconsistent samples: {n_inconsistent}/{total} ({n_inconsistent/total*100:.1f}%)")

    if n_inconsistent > 0:
        print(f"\n  Flip patterns (normal -> swap_aligned):")
        flip_counter = Counter()
        for i in range(total):
            if inconsistent_mask[i]:
                from_label = ID2LABEL[preds_normal[i]]
                to_label = ID2LABEL[preds_swap_aligned[i]]
                flip_counter[(from_label, to_label)] += 1

        for (f, t), cnt in flip_counter.most_common():
            print(f"    {f:>7s} -> {t:<7s}: {cnt:3d} ({cnt/n_inconsistent*100:.1f}% of flips)")

    # ============================================================
    # 分析 4: 準確率比較
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 4: Accuracy Comparison")
    print("=" * 60)

    acc_normal = accuracy_score(gt_ids, preds_normal)
    acc_swap = accuracy_score(gt_ids, preds_swap_aligned)
    acc_tta = accuracy_score(gt_ids, preds_tta)

    print(f"\n  Normal order accuracy:     {acc_normal*100:.2f}%")
    print(f"  Swapped order accuracy:    {acc_swap*100:.2f}%")
    print(f"  TTA (debiased) accuracy:   {acc_tta*100:.2f}%")
    print(f"  Accuracy gap (normal-swap): {(acc_normal - acc_swap)*100:+.2f}%")
    print(f"  TTA improvement over normal: {(acc_tta - acc_normal)*100:+.2f}%")

    # Per-class accuracy
    print(f"\n  Per-class accuracy:")
    print(f"  {'Class':>10s} | {'Normal':>8s} | {'Swapped':>8s} | {'TTA':>8s} | {'# samples':>9s}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}")
    for cls_name, cls_id in LABEL2ID.items():
        mask = gt_ids == cls_id
        n_cls = mask.sum()
        if n_cls == 0:
            continue
        a_n = (preds_normal[mask] == cls_id).sum() / n_cls * 100
        a_s = (preds_swap_aligned[mask] == cls_id).sum() / n_cls * 100
        a_t = (preds_tta[mask] == cls_id).sum() / n_cls * 100
        print(f"  {cls_name:>10s} | {a_n:7.1f}% | {a_s:7.1f}% | {a_t:7.1f}% | {n_cls:9d}")

    # ============================================================
    # 分析 5: 分類報告 (Classification Report)
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 5: Detailed Classification Reports")
    print("=" * 60)

    label_names = ["A", "B", "tie", "neither"]

    print("\n  --- Normal Order ---")
    print(classification_report(gt_ids, preds_normal, target_names=label_names, digits=4))

    print("  --- Swapped Order (aligned) ---")
    print(classification_report(gt_ids, preds_swap_aligned, target_names=label_names, digits=4))

    print("  --- TTA (debiased) ---")
    print(classification_report(gt_ids, preds_tta, target_names=label_names, digits=4))

    # ============================================================
    # 分析 6: Confusion Matrices
    # ============================================================
    print("=" * 60)
    print("ANALYSIS 6: Confusion Matrices")
    print("=" * 60)

    for name, preds in [("Normal", preds_normal),
                         ("Swapped (aligned)", preds_swap_aligned),
                         ("TTA (debiased)", preds_tta)]:
        cm = confusion_matrix(gt_ids, preds, labels=[0, 1, 2, 3])
        print(f"\n  --- {name} ---")
        print(f"  {'':>10s}  pred_A  pred_B  pred_tie  pred_neither")
        for i, row_name in enumerate(label_names):
            row_str = "  ".join(f"{v:6d}" for v in cm[i])
            print(f"  {row_name:>10s}  {row_str}")

    # ============================================================
    # 分析 7: Softmax 機率分析 (更精細的 bias 量化)
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS 7: Softmax Probability Analysis")
    print("=" * 60)

    # 比較 normal 和 swap 的平均 softmax 機率
    avg_prob_normal = probs_normal.mean(axis=0)
    avg_prob_swap_raw = probs_swap.mean(axis=0)
    avg_prob_swap_aligned = probs_swap_aligned.mean(axis=0)
    avg_prob_tta = probs_tta.mean(axis=0)

    print(f"\n  Average softmax probabilities:")
    print(f"  {'':>20s}  {'P(A)':>8s}  {'P(B)':>8s}  {'P(tie)':>8s}  {'P(neither)':>10s}")
    print(f"  {'Normal':>20s}  {avg_prob_normal[0]:8.4f}  {avg_prob_normal[1]:8.4f}  "
          f"{avg_prob_normal[2]:8.4f}  {avg_prob_normal[3]:10.4f}")
    print(f"  {'Swap (raw)':>20s}  {avg_prob_swap_raw[0]:8.4f}  {avg_prob_swap_raw[1]:8.4f}  "
          f"{avg_prob_swap_raw[2]:8.4f}  {avg_prob_swap_raw[3]:10.4f}")
    print(f"  {'Swap (aligned)':>20s}  {avg_prob_swap_aligned[0]:8.4f}  {avg_prob_swap_aligned[1]:8.4f}  "
          f"{avg_prob_swap_aligned[2]:8.4f}  {avg_prob_swap_aligned[3]:10.4f}")
    print(f"  {'TTA (debiased)':>20s}  {avg_prob_tta[0]:8.4f}  {avg_prob_tta[1]:8.4f}  "
          f"{avg_prob_tta[2]:8.4f}  {avg_prob_tta[3]:10.4f}")

    # Position Bias in probability space
    # If no bias: P_normal(A) ≈ P_swap_aligned(A)
    prob_diff = avg_prob_normal - avg_prob_swap_aligned
    print(f"\n  Probability difference (Normal - Swap_aligned):")
    print(f"    dP(A)={prob_diff[0]:+.4f}, dP(B)={prob_diff[1]:+.4f}, "
          f"dP(tie)={prob_diff[2]:+.4f}, dP(neither)={prob_diff[3]:+.4f}")

    # Per-sample bias: P_normal(A) - P_swap_aligned(A) (should be ~0 if no bias)
    per_sample_bias = probs_normal[:, 0] - probs_swap_aligned[:, 0]
    print(f"\n  Per-sample position bias P_normal(A) - P_swap_aligned(A):")
    print(f"    Mean:   {per_sample_bias.mean():+.4f}")
    print(f"    Std:    {per_sample_bias.std():.4f}")
    print(f"    Median: {np.median(per_sample_bias):+.4f}")
    print(f"    |bias| > 0.1: {(np.abs(per_sample_bias) > 0.1).sum()}/{total} "
          f"({(np.abs(per_sample_bias) > 0.1).sum()/total*100:.1f}%)")
    print(f"    |bias| > 0.2: {(np.abs(per_sample_bias) > 0.2).sum()}/{total} "
          f"({(np.abs(per_sample_bias) > 0.2).sum()/total*100:.1f}%)")

    # ============================================================
    # 總結
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    bias_direction = "1st position (A)" if avg_first_position_rate > true_first_rate else "2nd position (B)"
    bias_magnitude = abs(avg_first_position_rate - true_first_rate) * 100

    print(f"""
  Model: {config.MODEL_NAME}
  Adapter: {args.adapter_dir}
  Validation samples: {total}

  Position Bias:
    - The model shows a {bias_magnitude:.1f}% bias toward the {bias_direction}
    - In normal order: predicts A {n_pred_A_normal} times vs B {n_pred_B_normal} times
    - In swapped order: predicts 1st-position {n_pred_A_swap} times vs 2nd-position {n_pred_B_swap} times
    - Prediction consistency across swap: {consistent}/{total} ({consistent/total*100:.1f}%)

  Accuracy:
    - Normal order:  {acc_normal*100:.2f}%
    - Swapped order:  {acc_swap*100:.2f}%
    - TTA debiased:  {acc_tta*100:.2f}%

  Debiasing (TTA with position swap):
    - TTA vs Normal: {(acc_tta - acc_normal)*100:+.2f}%
    - TTA vs Swapped: {(acc_tta - acc_swap)*100:+.2f}%
    - TTA {'improves' if acc_tta >= max(acc_normal, acc_swap) else 'does not improve'} over single-direction inference
""")


if __name__ == "__main__":
    main()
