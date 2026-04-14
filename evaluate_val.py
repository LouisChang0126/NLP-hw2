#!/usr/bin/env python3
"""
evaluate_val.py — 在 held-out val split 上評估模型，產出精確度與錯誤分析
用法:
  python evaluate_val.py --model_name google/gemma-4-E4B-it \
      --adapter_dir outputs/gemma-4-E4B-it_04141214/final_adapter
  python evaluate_val.py --model_name Qwen/Qwen3-8B \
      --adapter_dir outputs/Qwen3-8B_04092343/final_adapter
"""

import argparse
import gc
import json
import os
import types
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from peft import PeftModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

# ── 標籤映射 ──
LABEL2ID = {"A": 0, "B": 1, "tie": 2, "neither": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ── 對話展平 ──
def flatten_dialog(dialog):
    lines = []
    for msg in dialog:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)

# ── Prompt 模板 ──
def _user_content_0(flat_1, flat_2):
    return (
        "You are a fair and impartial judge. Compare the two AI assistant responses "
        "and decide which one is better.\n\n"
        "Rules:\n"
        "- A: Response A is better\n"
        "- B: Response B is better\n"
        "- tie: Both responses are equally good\n"
        "- neither: Both responses are equally bad\n\n"
        f"### Response A\n{flat_1}\n\n"
        f"### Response B\n{flat_2}"
    )

def build_prompt(dialog_1, dialog_2, tokenizer, template_id=0):
    flat_1 = flatten_dialog(dialog_1)
    flat_2 = flatten_dialog(dialog_2)
    user_content = _user_content_0(flat_1, flat_2)
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return prompt

# ── 分類 forward ──
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


def run_batch_inference(model, tokenizer, prompts, batch_size=8, max_length=2048):
    all_probs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Inference"):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits.float(), dim=-1)
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--train_json", type=str, default="data/train.json")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tta", action="store_true", default=True,
                        help="Enable TTA position swap")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save per-sample results to JSON")
    args = parser.parse_args()

    # ── 1. 載入資料並切分 val set (與訓練一致) ──
    with open(args.train_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    verdicts = [s["verdict"] for s in data]
    _, val_data = train_test_split(
        data, test_size=args.val_ratio, random_state=args.seed, stratify=verdicts
    )
    print(f"[INFO] Val set: {len(val_data)} samples")
    val_verdicts = Counter(s["verdict"] for s in val_data)
    print(f"[INFO] Val distribution: {dict(val_verdicts)}")

    # ── 2. 載入模型 ──
    print(f"[INFO] Loading base model: {args.model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    for attr in ("vision_tower", "embed_vision", "audio_tower", "embed_audio"):
        if hasattr(base_model.model, attr):
            delattr(base_model.model, attr)
            print(f"[INFO] Deleted {attr}")
    torch.cuda.empty_cache()
    gc.collect()

    hidden_size = (base_model.config.text_config.hidden_size
                   if hasattr(base_model.config, "text_config")
                   else base_model.config.hidden_size)
    base_model.score = nn.Linear(hidden_size, 4, bias=False)
    base_model.score = base_model.score.to(device=base_model.device, dtype=torch.bfloat16)
    base_model.forward = types.MethodType(_cls_forward, base_model)

    print(f"[INFO] Loading adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── 3. 建立 prompts ──
    normal_prompts = [
        build_prompt(s["dialog_1"], s["dialog_2"], tokenizer)
        for s in val_data
    ]

    # ── 4. 推論 ──
    print("[INFO] Running normal inference...")
    probs_normal = run_batch_inference(
        model, tokenizer, normal_prompts,
        batch_size=args.batch_size, max_length=args.max_length,
    )

    if args.tta:
        swapped_prompts = [
            build_prompt(s["dialog_2"], s["dialog_1"], tokenizer)
            for s in val_data
        ]
        print("[INFO] Running TTA swapped inference...")
        probs_swap = run_batch_inference(
            model, tokenizer, swapped_prompts,
            batch_size=args.batch_size, max_length=args.max_length,
        )
        probs_swap_aligned = probs_swap.copy()
        probs_swap_aligned[:, 0] = probs_swap[:, 1]
        probs_swap_aligned[:, 1] = probs_swap[:, 0]
        final_probs = (probs_normal + probs_swap_aligned) / 2.0
    else:
        final_probs = probs_normal

    pred_ids = np.argmax(final_probs, axis=-1)
    pred_labels = [ID2LABEL[p] for p in pred_ids]
    true_labels = [s["verdict"] for s in val_data]

    # ── 5. 計算指標 ──
    acc = accuracy_score(true_labels, pred_labels)
    print(f"\n{'='*60}")
    print(f"  Model: {args.model_name}")
    print(f"  Adapter: {args.adapter_dir}")
    print(f"  Overall Accuracy: {acc:.4f} ({int(acc*len(val_data))}/{len(val_data)})")
    print(f"{'='*60}")

    # Confusion Matrix
    labels_order = ["A", "B", "tie", "neither"]
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_order)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(f"{'':>10} {'A':>6} {'B':>6} {'tie':>6} {'neit':>6}")
    for i, row_label in enumerate(labels_order):
        row = cm[i]
        print(f"{row_label:>10} {row[0]:>6} {row[1]:>6} {row[2]:>6} {row[3]:>6}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, labels=labels_order, digits=4))

    # Per-class accuracy
    print("Per-class Accuracy:")
    for label in labels_order:
        mask = [t == label for t in true_labels]
        correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p == label)
        total = sum(mask)
        class_acc = correct / total if total > 0 else 0
        print(f"  {label:>8}: {correct}/{total} = {class_acc:.4f}")

    # ── 6. 錯誤分析 ──
    errors = []
    for idx, (sample, pred, true) in enumerate(zip(val_data, pred_labels, true_labels)):
        if pred != true:
            flat_1 = flatten_dialog(sample["dialog_1"])
            flat_2 = flatten_dialog(sample["dialog_2"])
            errors.append({
                "id": sample["id"],
                "true": true,
                "pred": pred,
                "num_turns": sample.get("num_turns", 1),
                "len_dialog_1": len(flat_1),
                "len_dialog_2": len(flat_2),
                "total_len": len(flat_1) + len(flat_2),
                "len_diff": abs(len(flat_1) - len(flat_2)),
                "dialog_1_preview": flat_1[:200],
                "dialog_2_preview": flat_2[:200],
                "confidence": float(np.max(final_probs[idx])),
                "probs": {ID2LABEL[i]: float(final_probs[idx][i]) for i in range(4)},
            })

    print(f"\n{'='*60}")
    print(f"  Error Analysis: {len(errors)} errors / {len(val_data)} total")
    print(f"{'='*60}")

    # Error type distribution
    error_types = Counter((e["true"], e["pred"]) for e in errors)
    print("\nMost common error types (true -> pred):")
    for (t, p), count in error_types.most_common(15):
        print(f"  {t:>8} -> {p:<8}: {count} ({count/len(errors)*100:.1f}%)")

    # Errors by true label
    print("\nErrors by true label:")
    for label in labels_order:
        label_errors = [e for e in errors if e["true"] == label]
        total = sum(1 for t in true_labels if t == label)
        pred_dist = Counter(e["pred"] for e in label_errors)
        print(f"  {label:>8}: {len(label_errors)}/{total} errors "
              f"-> predicted as: {dict(pred_dist)}")

    # Errors by num_turns
    print("\nErrors by num_turns:")
    turns_all = Counter(s.get("num_turns", 1) for s in val_data)
    turns_err = Counter(e["num_turns"] for e in errors)
    for t in sorted(turns_all.keys()):
        total = turns_all[t]
        err = turns_err.get(t, 0)
        err_rate = err / total if total > 0 else 0
        print(f"  {t} turns: {err}/{total} errors ({err_rate:.2%})")

    # Errors by total dialog length (quartiles)
    all_lens = [len(flatten_dialog(s["dialog_1"])) + len(flatten_dialog(s["dialog_2"]))
                for s in val_data]
    q25, q50, q75 = np.percentile(all_lens, [25, 50, 75])
    print(f"\nDialog length quartiles: Q25={q25:.0f}, Q50={q50:.0f}, Q75={q75:.0f}")

    len_bins = {"short (<Q25)": [], "medium (Q25-Q75)": [], "long (>Q75)": []}
    for idx, sample in enumerate(val_data):
        total_len = all_lens[idx]
        is_error = pred_labels[idx] != true_labels[idx]
        if total_len < q25:
            len_bins["short (<Q25)"].append(is_error)
        elif total_len > q75:
            len_bins["long (>Q75)"].append(is_error)
        else:
            len_bins["medium (Q25-Q75)"].append(is_error)

    print("Error rate by dialog length:")
    for bin_name, errs in len_bins.items():
        if errs:
            err_rate = sum(errs) / len(errs)
            print(f"  {bin_name}: {sum(errs)}/{len(errs)} ({err_rate:.2%})")

    # Length difference analysis for A/B errors
    print("\nLength difference analysis (|len_A - len_B|):")
    for label in ["A", "B"]:
        label_samples = [(idx, s) for idx, s in enumerate(val_data) if s["verdict"] == label]
        correct_diffs = []
        wrong_diffs = []
        for idx, s in label_samples:
            diff = abs(len(flatten_dialog(s["dialog_1"])) - len(flatten_dialog(s["dialog_2"])))
            if pred_labels[idx] == true_labels[idx]:
                correct_diffs.append(diff)
            else:
                wrong_diffs.append(diff)
        if correct_diffs and wrong_diffs:
            print(f"  True={label}: correct avg_diff={np.mean(correct_diffs):.0f}, "
                  f"wrong avg_diff={np.mean(wrong_diffs):.0f}")

    # Position bias analysis
    print("\nPosition bias analysis:")
    a_pred_count = sum(1 for p in pred_labels if p == "A")
    b_pred_count = sum(1 for p in pred_labels if p == "B")
    a_true_count = sum(1 for t in true_labels if t == "A")
    b_true_count = sum(1 for t in true_labels if t == "B")
    print(f"  True A/B: {a_true_count}/{b_true_count} = {a_true_count/b_true_count:.3f}")
    print(f"  Pred A/B: {a_pred_count}/{b_pred_count} = {a_pred_count/b_pred_count:.3f}")

    # TTA consistency
    if args.tta:
        normal_preds = np.argmax(probs_normal, axis=-1)
        consistent = np.sum(normal_preds == pred_ids)
        print(f"\nTTA consistency: {consistent}/{len(val_data)} "
              f"({consistent/len(val_data)*100:.1f}%)")

        # Flip analysis: cases where TTA changed the answer
        flips = []
        for idx in range(len(val_data)):
            if normal_preds[idx] != pred_ids[idx]:
                flips.append({
                    "id": val_data[idx]["id"],
                    "true": true_labels[idx],
                    "pred_before_tta": ID2LABEL[normal_preds[idx]],
                    "pred_after_tta": pred_labels[idx],
                })
        helped = sum(1 for f in flips if f["pred_after_tta"] == f["true"])
        hurt = sum(1 for f in flips if f["pred_before_tta"] == f["true"])
        print(f"  TTA flipped: {len(flips)} samples")
        print(f"  TTA helped (wrong->correct): {helped}")
        print(f"  TTA hurt (correct->wrong): {hurt}")

    # ── 7. 具體錯誤範例 ──
    print(f"\n{'='*60}")
    print("  Concrete Error Examples (top 5 most confident wrong)")
    print(f"{'='*60}")
    errors_sorted = sorted(errors, key=lambda e: e["confidence"], reverse=True)
    for i, e in enumerate(errors_sorted[:5]):
        print(f"\n--- Example {i+1} (id={e['id']}) ---")
        print(f"  True: {e['true']}, Pred: {e['pred']}, Confidence: {e['confidence']:.4f}")
        print(f"  Probs: {e['probs']}")
        print(f"  Turns: {e['num_turns']}, Len_A: {e['len_dialog_1']}, Len_B: {e['len_dialog_2']}")
        print(f"  Dialog A preview: {e['dialog_1_preview'][:150]}...")
        print(f"  Dialog B preview: {e['dialog_2_preview'][:150]}...")

    # Show examples from each major error type
    print(f"\n{'='*60}")
    print("  Examples by Error Category")
    print(f"{'='*60}")

    # tie -> A or B (tie misclassified)
    tie_errors = [e for e in errors if e["true"] == "tie"]
    if tie_errors:
        e = tie_errors[0]
        print(f"\n[tie misclassified] id={e['id']}: true=tie, pred={e['pred']}")
        print(f"  Probs: {e['probs']}")
        print(f"  Dialog A: {e['dialog_1_preview'][:200]}...")
        print(f"  Dialog B: {e['dialog_2_preview'][:200]}...")

    # neither -> A or B (neither misclassified)
    neither_errors = [e for e in errors if e["true"] == "neither"]
    if neither_errors:
        e = neither_errors[0]
        print(f"\n[neither misclassified] id={e['id']}: true=neither, pred={e['pred']}")
        print(f"  Probs: {e['probs']}")
        print(f"  Dialog A: {e['dialog_1_preview'][:200]}...")
        print(f"  Dialog B: {e['dialog_2_preview'][:200]}...")

    # A <-> B confusion
    ab_errors = [e for e in errors if e["true"] in ("A","B") and e["pred"] in ("A","B")]
    if ab_errors:
        e = ab_errors[0]
        print(f"\n[A<->B confusion] id={e['id']}: true={e['true']}, pred={e['pred']}")
        print(f"  Probs: {e['probs']}")
        print(f"  Dialog A: {e['dialog_1_preview'][:200]}...")
        print(f"  Dialog B: {e['dialog_2_preview'][:200]}...")

    # Save results
    if args.output_json:
        results = {
            "model": args.model_name,
            "adapter": args.adapter_dir,
            "accuracy": acc,
            "num_errors": len(errors),
            "errors": errors,
            "per_class_accuracy": {},
            "error_type_distribution": {f"{t}->{p}": c for (t,p), c in error_types.items()},
        }
        for label in labels_order:
            mask = [t == label for t in true_labels]
            correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p == label)
            total = sum(mask)
            results["per_class_accuracy"][label] = correct / total if total > 0 else 0

        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] Results saved to: {args.output_json}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
