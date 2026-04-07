# Vibe Coding Task: LLM-as-a-Judge (Kaggle HW2)

## 1. 專案核心目標
使用 LoRA / QLoRA 進行參數高效微調，建立一個預測人類偏好的 LLM 裁判。
- 目標模型：`Gemma-4-E4B-IT` 或 `Qwen3.5-9B`
- 訓練硬體：單張 RTX 4090 (24GB VRAM)
- 任務類型：分類任務 (輸出 A, B, tie, neither)

## 2. 目錄與檔案架構設計
請為我實作以下四個核心 Python 檔案，並建立乾淨的專案結構：
1. `config.py` (全局設定中心)
2. `train.py` (微調主程式)
3. `dataset.py` (資料讀取與 Prompt 組合)
4. `inference.py` (推論與 Kaggle CSV 檔生成)

## 3. 核心功能實作限制與細節

### A. `config.py` (強制要求)
這必須是一個獨立的參數設定檔，讓我能集中調整所有的訓練與模型參數。必須包含以下變數：
- `MODEL_NAME`: 模型 HuggingFace 路徑或本地路徑
- `USE_QLORA`: 布林值 (True 啟用 4-bit 量化，False 使用常規 LoRA)
- `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`: LoRA 超參數
- `LEARNING_RATE`, `BATCH_SIZE`, `GRAD_ACCUMULATION_STEPS`, `NUM_EPOCHS`, `MAX_SEQ_LENGTH`
- `BASE_OUTPUT_DIR`: 預設為 `"outputs/"`

### B. 動態輸出目錄與備份機制 (重點邏輯)
在 `train.py` 啟動的最初期，請務必實作以下 Python 邏輯：
1. 使用 `datetime` 模組取得當前時間，格式化為 `MMDDHHSS` (月月日日時時秒秒)。
2. 從模型路徑提取乾淨的模型名稱 (例如把 `Qwen/Qwen3.5-9B` 轉為 `Qwen3.5-9B`)。
3. 建立本次實驗的專屬目錄：`outputs/{model_name}_{MMDDHHSS}/`。
4. **關鍵動作：** 使用 `shutil.copy` 將當下使用的 `config.py` 複製一份到這個專屬目錄下，作為實驗紀錄。
5. 將 Hugging Face `Trainer` 或 `SFTTrainer` 的 `output_dir` 參數指向這個目錄，確保所有的 checkpoints (ckpt) 和最終的 Adapter 權重都存放在此。

### C. `dataset.py` 邏輯 (資料集解析與處理)
- 讀取 `data/train.json`。資料結構包含 `id`, `dialog_1`, `dialog_2`, `num_turns`, `verdict`。
- **對話展平邏輯 (關鍵)：** `dialog_1` 與 `dialog_2` 是包含字典的陣列 (例如 `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`)。請寫一個 function 將這些多輪交替對話展平成易讀的純文本格式 (例如加上 "User: " 和 "Assistant: " 的前綴)。
- 設計一個清晰的 Prompt Template，將展平後的 `dialog_1` 與 `dialog_2` 組合在一起，明確指示模型扮演裁判，並輸出四個類別之一 ("A", "B", "tie", "neither")。
- 處理 Tokenization 與 Label 的轉換 (字串轉 ID，或處理 Text Generation 的格式)。

### D. `train.py` 邏輯
- 讀取 `config.py`，並執行上述的目錄建立與備份邏輯。
- 依照設定載入 Base Model (量化或不量化) 與 PEFT LoRA 配置。
- 啟動微調，並在控制台輸出清晰的進度與 Loss。

### E. `inference.py` 邏輯 (Kaggle 提交格式對齊)
- 允許讀取指定的 `outputs/{model_name}_{MMDDHHSS}/` 目錄來載入微調後的權重。
- 讀取 `data/test.json` 進行推論。請務必使用與 `dataset.py` 完全相同的對話展平與 Prompt Template 邏輯。
- 解析模型的生成結果，對應回 "A", "B", "tie", "neither" 之一。
- 生成結果並儲存為 CSV。**嚴格要求：** 參考 `data/sample_submission.csv` 的格式，輸出的檔案必須精準包含 1001 列 (1 列標題 + 1000 筆預測)。標題列的欄位名稱必須是 `id,verdict`，`id` 需對應 `data/test.json` 中的 `id`。
