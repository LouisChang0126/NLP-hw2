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

## 方法總覽

不採用「讓模型生成 verdict 字串」的做法，而是把 CausalLM 改造成 **4-class 序列分類器**：

- 取 CausalLM 的 hidden states，在最後一個非 padding token 上接一個線性分類頭 (`score: hidden → 4`)
- 用 `cross_entropy` 直接在 4 類上訓練，避免 token-level loss 與 logits materialization 的浪費
- 推論時直接讀分類頭的 softmax，不需要 generation
- 同一套 forward (`_cls_forward`) 同時被 [train.py](train.py) 與 [inference.py](inference.py) 使用，避免訓練／推論偏差

## 專案結構

```
.
├── config.py              # 全局設定中心（模型、LoRA、訓練超參數、擴增、TTA 開關）
├── dataset.py             # 資料讀取、對話展平、Prompt 組合、資料擴增
├── collator.py            # 早期保留（目前主流程已不使用）
├── train.py               # LoRA / QLoRA 序列分類微調主程式
├── inference.py           # 微調後 adapter 推論 + TTA + Kaggle CSV 生成
├── top3_averaging.py      # 取 top-3 checkpoint 做權重平均，輸出 ckpt-top3 adapter
├── pairrm_inference.py    # 不需訓練的 llm-blender/PairRM zero-shot baseline
├── data/
│   ├── train.json             # 4000 筆標註資料
│   ├── test.json              # 1000 筆測試資料
│   └── sample_submission.csv  # Kaggle 提交範例
└── outputs/                   # 訓練產出（自動建立）
    └── {model}_{MMDDHHMM}/
        ├── config.py              # 本次實驗的 config 備份
        ├── ckpt-epoch{e}-acc{a}/  # checkpoint（依驗證 accuracy 命名）
        ├── ckpt-top3/             # (可選) top3_averaging.py 的權重平均輸出
        └── final_adapter/         # 最終 adapter 權重（best ckpt）
```

## 環境需求

- Python 3.10+
- CUDA GPU（Qwen3-8B QLoRA：RTX 4090 24GB 可運行；A6000 48GB 更寬裕）

Gemma 4 需要從 source 安裝 transformers 與 peft：

```bash
pip install torch>=2.1.0
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/huggingface/peft.git
pip install trl==1.0.0 bitsandbytes liger-kernel accelerate datasets scikit-learn
```

或直接使用 requirements.txt：

```bash
pip install -r requirements.txt
```

## 快速開始

### 1. 訓練

調整 [config.py](config.py) 中的參數後直接執行：

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True python train.py
```

> **記憶體注意**：Gemma 4 為多模態模型（含 vision / audio encoder），即使 text-only 訓練，模型本體也需約 16 GB VRAM；Qwen3-8B QLoRA 在 24 GB VRAM 下需要搭配 `BATCH_SIZE=1~2` 與梯度累積。

訓練會自動：
- 建立帶時間戳的實驗目錄 `outputs/{model}_{MMDDHHMM}/`
- 備份當次使用的 `config.py`
- 在 CausalLM body 上掛 4-class 分類頭，用 cross-entropy 訓練
- 每 `EVAL_EVERY_N_EPOCH` 做一次驗證，checkpoint 重新命名為 `ckpt-epoch{e}-acc{a}` 並保留 metric
- 訓練結束載入 best checkpoint（依 val accuracy）並儲存至 `final_adapter/`

### 2. 推論

```bash
python inference.py --adapter_dir outputs/{model}_{MMDDHHMM}/final_adapter
```

可選參數：
- `--batch_size 16`：批次推論大小（預設 16）
- `--output_csv path/to/submission.csv`：指定輸出路徑（預設存到 `adapter_dir` 父目錄）
- `--test_json path/to/test.json`：指定測試資料

推論流程內建 **Test-Time Augmentation**（由 `config.py` 中的 `TTA_*` 控制）：
- `TTA_PROMPT_TEMPLATES`：對指定的 prompt 模板分別跑一次
- `TTA_POSITION_SWAP`：再對 dialog 反順序跑一次（並把分類頭的 A/B 機率對調回來）
- 全部結果用 **softmax 平均**，最後 argmax 取得 verdict

產出的 CSV 格式為 `id,verdict`，共 1001 列（1 header + 1000 predictions）。

### 3. (可選) Top-3 Checkpoint Weight Averaging

若想對同一次訓練的多個 checkpoint 做權重平均（通常比單一 best checkpoint 更穩健）：

```bash
python top3_averaging.py --adapter_dir outputs/{model}_{MMDDHHMM}/
```

會掃描 `ckpt-epoch*-acc*` 目錄，按 accuracy 排序取前 k 個（預設 3），把 `adapter_model.safetensors` 平均後輸出到 `ckpt-top3/`。之後可直接把這個目錄當成 adapter 餵給 `inference.py`：

```bash
python inference.py --adapter_dir outputs/{model}_{MMDDHHMM}/ckpt-top3
```

可選參數：
- `--k 5`：改變要平均的 checkpoint 數
- `--output_name my-avg`：自訂輸出目錄名

### 4. (可選) PairRM Zero-shot Baseline

不需要任何訓練，直接用 [llm-blender/PairRM](https://github.com/yuchenlin/LLM-Blender) 推論 4 類：

```bash
python pairrm_inference.py
python pairrm_inference.py --eval_train       # 在 train.json 上評估準確率，便於調閾值
```

策略：PairRM 給出 `logit = log P(A > B)`，sigmoid 距 0.5 夠遠就判 A/B；落在模糊區則用「兩個回答平均字數」當品質代理區分 tie / neither。可調 `--margin_threshold` 與 `--neither_word_threshold`。

## 設定說明

所有參數集中在 [config.py](config.py)：

### 模型與 LoRA

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `MODEL_NAME` | `Qwen/Qwen3-8B` | HuggingFace 模型路徑（也支援 `google/gemma-4-E4B-it`）|
| `USE_QLORA` | `True` | 啟用 4-bit QLoRA 量化 |
| `LORA_R` | `64` | LoRA rank |
| `LORA_ALPHA` | `LORA_R * 0.75` | LoRA alpha |
| `LORA_DROPOUT` | `0.05` | LoRA dropout |
| `LORA_TARGET_MODULES` | 自動判斷 | Gemma-4 用 regex 排除 vision/audio tower；純文字模型直接 list 七個 proj |
| `NUM_LABELS` | `4` | 分類頭輸出維度 |

### 訓練超參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `LEARNING_RATE` | `5e-5` | 學習率 |
| `BATCH_SIZE` | `2` | per-device batch size |
| `GRAD_ACCUMULATION_STEPS` | `32 // BATCH_SIZE` | 梯度累積（effective batch = 32）|
| `NUM_EPOCHS` | `4` | 訓練 epoch 數 |
| `MAX_SEQ_LENGTH` | `2048` | 最大序列長度 |
| `WARMUP_RATIO` | `0.1` | warm-up 比例 |
| `WEIGHT_DECAY` | `0.01` | weight decay |
| `LR_SCHEDULER_TYPE` | `cosine` | LR scheduler |
| `EVAL_EVERY_N_EPOCH` | `0.25` | 每 n 個 epoch 做一次 validation |
| `BF16` / `FP16` | `True` / `False` | QLoRA 下建議用 bf16 |

### 資料擴增

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `AUG_POSITION_SWAP` | `True` | 位置交換與標籤反轉（A↔B），訓練資料翻倍 |
| `AUG_PROMPT_DIVERSE` | `True` | 訓練時隨機使用 4 種 prompt 模板 |

### Test-Time Augmentation

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `TTA_ENABLED` | `True` | 啟用 TTA（softmax 平均後 argmax）|
| `TTA_POSITION_SWAP` | `True` | TTA：原順序 + 反順序 |
| `TTA_PROMPT_TEMPLATES` | `[0, 1, 2, 3]` | TTA：使用哪些 prompt 模板（索引）|

## 設計細節

### 為什麼用分類頭而非生成？
- 4 個 verdict 可以直接視為 4-class 分類，分類頭只需 `hidden_dim × 4` 個參數
- 訓練時不需要 token-level loss masking，也不需要把 vocab × seq_len 的 logits 物化
- 推論時不需要 generation，TTA 可以直接在 softmax 機率上做加權平均，做到「軟投票」而非「硬投票」
- `_cls_forward`（[train.py](train.py)、[inference.py](inference.py) 共用）取最後一個非 padding token 的 hidden state 做 pool

### Prompt 策略
- 使用 `tokenizer.apply_chat_template` 包裝 prompt，符合 IT 模型的原生格式
- 訓練時可隨機切換 4 種 user-message 模板（[dataset.py](dataset.py)）以提升泛化能力
- 推論時透過 `TTA_PROMPT_TEMPLATES` 對多個模板分別跑，再做 softmax 平均

### 訓練優化
- `RenameCheckpointCallback`（[train.py](train.py)）：在每次 evaluate 後把 `checkpoint-N` 重命名為 `ckpt-epoch{e}-acc{a}`，並把 metric 存進 `eval_metrics.json`，方便後續 top-k 平均
- `load_best_model_at_end=True`：自動載入 val accuracy 最高的 checkpoint
- 分層抽樣（Stratified Split）：確保 train/val 的 4 類 verdict 比例一致
- 全域 seed 固定（`random` / `numpy` / `torch` / `cudnn`）：相同 config 可復現結果

### 位置偏差對策
- **訓練時** Position Swap 擴增：每筆資料額外產生 dialog 交換版本，迫使模型不依賴位置判斷
- **推論時** TTA Position Swap：跑一次反順序，把分類頭的 A/B 機率交換回來再平均

### Top-3 Weight Averaging
- 同一次訓練裡，最後幾個 epoch 的 checkpoint 通常落在相近的 loss basin，平均它們的權重往往比挑單一 best 更穩健
- [top3_averaging.py](top3_averaging.py) 直接對 LoRA `adapter_model.safetensors` 做算術平均，不需要重新訓練

### Gemma 4 相容性
Gemma 4 是多模態模型，包含 vision tower 和 audio tower：
- `LORA_TARGET_MODULES` 改用 regex（`.*language_model\..*\.(q_proj|...)`）限定只套用 LoRA 到 language model，避免 peft 嘗試對 `Gemma4ClippableLinear`（vision encoder 專用層）套 LoRA
- [pairrm_inference.py](pairrm_inference.py) 為 transformers v5 補了 `TRANSFORMERS_CACHE` 與 `batch_encode_plus` 兩個 monkey-patch，讓 llm-blender 0.0.2 可以正常載入
