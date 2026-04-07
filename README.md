# LLM-as-a-Judge: Predicting Human Preferences

INLP HW2 — 使用 LoRA / QLoRA 微調開源 LLM，建立預測人類偏好的 LLM 裁判。

## 任務說明

給定兩段 AI 助手的對話回覆，預測人類偏好的 verdict：

| Verdict | 意義 |
|---------|------|
| A | Response A (dialog_1) 較好 |
| B | Response B (dialog_2) 較好 |
| tie | 兩者一樣好 |
| neither | 兩者一樣差 |

## 專案結構

```
.
├── config.py          # 全局設定中心（模型、LoRA、訓練超參數、擴增開關）
├── dataset.py         # 資料讀取、對話展平、Prompt 組合、資料擴增
├── train.py           # LoRA / QLoRA 微調主程式
├── inference.py       # 推論與 Kaggle CSV 生成
├── generate_cot.py    # (可選) 本地逆向思維鏈生成
├── data/
│   ├── train.json             # 4000 筆標註資料
│   ├── test.json              # 1000 筆測試資料
│   └── sample_submission.csv  # Kaggle 提交範例
└── outputs/                   # 訓練產出（自動建立）
    └── {model}_{MMDDHHMM}/
        ├── config.py          # 本次實驗的 config 備份
        ├── checkpoint-*/      # 訓練 checkpoint
        └── final_adapter/     # 最終 adapter 權重
```

## 環境需求

- Python 3.10+
- CUDA GPU（建議 RTX 4090 24GB）

```bash
pip install torch transformers peft trl bitsandbytes accelerate scikit-learn
```

## 快速開始

### 1. 訓練

調整 `config.py` 中的參數後直接執行：

```bash
python train.py
```

訓練會自動：
- 建立帶時間戳的實驗目錄 `outputs/{model}_{MMDDHHMM}/`
- 備份當次使用的 `config.py`
- 使用 `DataCollatorForCompletionOnlyLM` 只對 verdict 計算 loss
- 訓練結束載入 best checkpoint 並儲存至 `final_adapter/`

### 2. 推論

```bash
python inference.py --adapter_dir outputs/{model}_{MMDDHHMM}/final_adapter
```

可選參數：
- `--batch_size 4`：批次推論大小（預設 4）
- `--output_csv path/to/submission.csv`：指定輸出路徑
- `--test_json path/to/test.json`：指定測試資料

產出的 CSV 格式為 `id,verdict`，共 1001 列（1 header + 1000 predictions）。

### 3. (可選) 生成 CoT 理由

在微調前執行，讓 base model 為訓練資料生成 rationale：

```bash
python generate_cot.py
```

生成完成後，在 `config.py` 中設定 `AUG_REVERSE_COT = True` 再執行 `train.py`。

## 設定說明

所有參數集中在 `config.py`：

### 模型與 LoRA

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `MODEL_NAME` | `google/gemma-4-E4B-it` | HuggingFace 模型路徑 |
| `USE_QLORA` | `True` | 啟用 4-bit QLoRA 量化 |
| `LORA_R` | `16` | LoRA rank |
| `LORA_ALPHA` | `32` | LoRA alpha |
| `LORA_DROPOUT` | `0.05` | LoRA dropout |

### 訓練超參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `LEARNING_RATE` | `2e-4` | 學習率 |
| `BATCH_SIZE` | `2` | per-device batch size |
| `GRAD_ACCUMULATION_STEPS` | `8` | 梯度累積（effective batch = 16）|
| `NUM_EPOCHS` | `3` | 訓練 epoch 數 |
| `MAX_SEQ_LENGTH` | `4096` | 最大序列長度 |

### 資料擴增

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `AUG_POSITION_SWAP` | `True` | 位置交換與標籤反轉（A↔B），訓練資料翻倍 |
| `AUG_REVERSE_COT` | `False` | 逆向思維鏈，需先執行 `generate_cot.py` |
| `AUG_PROMPT_DIVERSE` | `True` | 訓練時隨機使用 4 種 prompt 模板 |

## 設計細節

### Prompt 策略
- 使用 `tokenizer.apply_chat_template` 包裝 prompt，符合 IT 模型的原生格式
- 推論固定使用模板 0；訓練時可隨機切換 4 種模板以提升泛化能力

### 訓練優化
- `DataCollatorForCompletionOnlyLM`：只對 verdict（及 CoT rationale）計算 loss，不浪費在背 prompt
- `load_best_model_at_end=True`：自動載入 val loss 最低的 checkpoint
- 分層抽樣（Stratified Split）：確保 train/val 的 4 類 verdict 比例一致
- 全域 seed 固定（`random` / `numpy` / `torch` / `cudnn`）：相同 config 可復現結果

### 位置偏差對策
- Position Swap 擴增：每筆資料額外產生 dialog 交換版本，迫使模型不依賴位置判斷
