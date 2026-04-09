#!/usr/bin/env python3
"""Top-3 Weight Averaging for LoRA adapters.

Usage:
    python top3_averaging.py --adapter_dir outputs/{model}_{MMDDHHMM}/

Finds the top-3 checkpoints by accuracy (parsed from directory names like
ckpt-epoch1.49-acc0.6125), averages their weights, and saves the result
to {adapter_dir}/ckpt-top3/.
"""

import argparse
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def find_top_k_checkpoints(adapter_dir: Path, k: int = 3) -> list[tuple[str, float]]:
    """Find top-k checkpoints by accuracy from directory names."""
    pattern = re.compile(r"ckpt-epoch[\d.]+-acc([\d.]+)")
    ckpts = []
    for d in adapter_dir.iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            acc = float(m.group(1))
            ckpts.append((d.name, acc))

    ckpts.sort(key=lambda x: x[1], reverse=True)
    return ckpts[:k]


def average_weights(adapter_dir: Path, ckpt_names: list[str]) -> dict[str, torch.Tensor]:
    """Load and average safetensor weights from multiple checkpoints."""
    k = len(ckpt_names)
    avg_state: dict[str, torch.Tensor] = {}

    for i, name in enumerate(ckpt_names):
        path = adapter_dir / name / "adapter_model.safetensors"
        print(f"  Loading [{i+1}/{k}] {name}")
        state = load_file(str(path))
        for key, tensor in state.items():
            if key in avg_state:
                avg_state[key] = avg_state[key] + tensor
            else:
                avg_state[key] = tensor.clone()

    for key in avg_state:
        avg_state[key] = avg_state[key] / k

    return avg_state


def main():
    parser = argparse.ArgumentParser(description="Top-K Weight Averaging for LoRA adapters")
    parser.add_argument("--adapter_dir", type=str, required=True,
                        help="Path to the training output directory")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of top checkpoints to average (default: 3)")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Output directory name (default: ckpt-top{k})")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Directory not found: {adapter_dir}")

    k = args.k
    output_name = args.output_name or f"ckpt-top{k}"
    output_dir = adapter_dir / output_name

    # Find top-k checkpoints
    top_ckpts = find_top_k_checkpoints(adapter_dir, k)
    if len(top_ckpts) < k:
        print(f"Warning: only found {len(top_ckpts)} checkpoints, expected {k}")
    if not top_ckpts:
        raise RuntimeError("No checkpoints found matching ckpt-epoch*-acc* pattern")

    print(f"Top-{len(top_ckpts)} checkpoints:")
    for name, acc in top_ckpts:
        print(f"  {name}  (acc={acc:.4f})")

    # Average weights
    print("\nAveraging weights...")
    avg_state = average_weights(adapter_dir, [name for name, _ in top_ckpts])

    # Save to output directory (copy config files from the best checkpoint)
    best_ckpt_dir = adapter_dir / top_ckpts[0][0]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save averaged weights
    save_file(avg_state, str(output_dir / "adapter_model.safetensors"))
    print(f"\nSaved averaged weights to {output_dir / 'adapter_model.safetensors'}")

    # Copy non-weight files from the best checkpoint
    copy_files = [
        "adapter_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]
    for fname in copy_files:
        src = best_ckpt_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    print(f"\nDone! Averaged adapter saved to: {output_dir}")


if __name__ == "__main__":
    main()
