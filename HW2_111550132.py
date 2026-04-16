# ============================================================
# HW2_111550132.py — 一鍵執行 Train + Inference
# ============================================================
#
# 用法:
#   python HW2_111550132.py
#
# 可選參數:
#   --skip_train                     跳過訓練，直接用既有 adapter 推論
#   --adapter_dir PATH               指定 adapter 目錄（跳過訓練時必須提供，
#                                    或覆寫訓練後的自動選取結果）
#   --batch_size N                   推論 batch size（預設 16）
#   --output_csv PATH                推論輸出 CSV 路徑
#   --test_json PATH                 測試資料路徑
#
# 流程：
#   1. 執行 train.py（會在 outputs/{model}_{timestamp}/ 下產生 final_adapter）
#   2. 掃描 outputs/ 找出最新一次訓練的 final_adapter
#   3. 執行 inference.py 產生 submission.csv
# ============================================================

import argparse
import os
import subprocess
import sys
from glob import glob


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run(cmd: list[str]) -> None:
    print(f"\n[HW2] $ {' '.join(cmd)}\n", flush=True)
    ret = subprocess.call(cmd, cwd=SCRIPT_DIR)
    if ret != 0:
        print(f"[HW2] Command failed with exit code {ret}", file=sys.stderr)
        sys.exit(ret)


def find_latest_final_adapter() -> str:
    candidates = glob(os.path.join(SCRIPT_DIR, "outputs", "*", "final_adapter"))
    if not candidates:
        raise FileNotFoundError(
            "找不到任何 final_adapter，請確認 train.py 已成功完成"
        )
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_train", action="store_true",
                        help="跳過訓練，直接執行推論")
    parser.add_argument("--adapter_dir", type=str, default=None,
                        help="指定 adapter 目錄（預設自動選最新的 final_adapter）")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--test_json", type=str, default=None)
    args = parser.parse_args()

    # ---------- 1. Train ----------
    if not args.skip_train:
        run([sys.executable, "train.py"])
    else:
        print("[HW2] --skip_train 啟用，跳過訓練階段")

    # ---------- 2. 選擇 adapter ----------
    adapter_dir = args.adapter_dir or find_latest_final_adapter()
    print(f"[HW2] 使用 adapter: {adapter_dir}")

    # ---------- 3. Inference ----------
    infer_cmd = [
        sys.executable, "inference.py",
        "--adapter_dir", adapter_dir,
        "--batch_size", str(args.batch_size),
    ]
    if args.output_csv:
        infer_cmd += ["--output_csv", args.output_csv]
    if args.test_json:
        infer_cmd += ["--test_json", args.test_json]
    run(infer_cmd)

    print("\n[HW2] 全流程執行完成")


if __name__ == "__main__":
    main()
