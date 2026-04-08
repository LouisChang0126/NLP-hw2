# Vibe Coding Task: LLM-as-a-Judge (Kaggle 衝刺高分版)

## 1. 專案背景與重構目標
目前專案使用的是 `AutoModelForCausalLM` (生成式) 搭配 `SFTTrainer`，這容易導致格式錯誤且對多輪長對話的分類邊界不夠清晰。
**本次重構的核心目標：** 將整個 Pipeline 全面切換為 **`AutoModelForSequenceClassification` (序列分類器)** 架構，並引入 Kaggle 頂級玩家的策略 (TTA、邊界規則、分層抽樣、大 Rank LoRA) 以極限提升 Accuracy 分數。

專案包含四個核心檔案：`config.py`, `dataset.py`, `train.py`, `inference.py`。
請依照下列 Phase 1 到 Phase 4 的指示進行程式碼重構。

---

## Phase 1: 更新配置中心 (`config.py`)
請修改或新增以下參數，這是分類任務與提升效能的關鍵：
1. **學習率與預熱:** 將 `LEARNING_RATE` 調降至 `5e-5` (因為分類頭是隨機初始化的，太高會崩潰)；新增 `WARMUP_RATIO = 0.1`。

---

## Phase 2: 資料處理與智慧截斷 (`dataset.py`)
1. **分層抽樣 (Stratified Split):** 修改 `load_train_val()`，使用 `sklearn.model_selection.train_test_split` 並設定 `stratify=verdicts`，確保 train/val 的 A, B, tie, neither 比例完全一致。
2. **Prompt 調整:** 因為是分類模型，不需要模型「生成文字」。請移除 Prompt 模板最後面的 `Verdict:` 提示語。只需將 Dialog_1 和 Dialog_2 清晰排版並回傳即可。
3. **Dataset 輸出格式:** `__getitem__` 現在不需要回傳 label 字串，請直接回傳 `text` 與 `labels` (int 型態：0, 1, 2, 3，分別對應 A, B, tie, neither)。

---

## Phase 3: 訓練主幹切換 (`train.py`)
這是最重要的架構切換。
1. **模型載入:** 拔除 `AutoModelForCausalLM`，改用 `AutoModelForSequenceClassification.from_pretrained(..., num_labels=4)`。
2. **Flash Attention:** 載入模型時，若 `config.USE_FLASH_ATTENTION` 為 True，加入 `attn_implementation="flash_attention_2"` 參數。
3. **Trainer 替換:** - 捨棄 `trl.SFTTrainer`，改用 `transformers.Trainer`。
   - 使用 `DataCollatorWithPadding(tokenizer=tokenizer)` 處理 Padding。
   - 實作 `compute_metrics` 函數：傳入 `eval_preds`，計算並回傳 Accuracy 和 Log Loss。
4. **抓取最佳模型:** 在 `TrainingArguments` 中設定 `load_best_model_at_end=True` 與 `metric_for_best_model="eval_loss"`，搭配 `save_strategy="epoch"` 與 `eval_strategy="epoch"`。

---

## Phase 4: 推論與測試時增強 (`inference.py`)
推論腳本需要大幅重構，加入以下三層防護網：
1. **批次推論 (Batched Inference):** 將 `tokenizer.padding_side` 設為 `"left"`，並實作 Batch 推論迴圈以榨乾算力。
2. **實作 TTA (Test-Time Augmentation) 核心邏輯:**
   針對每一筆資料 (排除上述邊界測資的)，模型必須推論兩次：
   - 第一次 (`Prompt_1`): 正常順序 `[Dialog 1, Dialog 2]` -> 取得 softmax 預測機率 `probs1`。
   - 第二次 (`Prompt_2`): 顛倒順序 `[Dialog 2, Dialog 1]` -> 取得 softmax 預測機率 `probs2`。
   - **關鍵對齊:** 因為輸入顛倒了，模型預測的 A 和 B 意義互換。必須將 `probs2` 的 index 0 (原 A) 和 index 1 (原 B) 對調，得到 `probs2_aligned`。
   - **最終決策:** 計算 `final_probs = (probs1 + probs2_aligned) / 2`，取 `argmax` 作為最終 verdict。

---