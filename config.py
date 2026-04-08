# ============================================================
# config.py — 全局設定中心
# ============================================================

# ---------- 模型 ----------
MODEL_NAME = "google/gemma-4-E4B-it"   # HuggingFace 路徑或本地路徑
# MODEL_NAME = "Qwen/Qwen3.5-9B" # "Qwen/Qwen3-8B"

# ---------- LoRA / QLoRA ----------
USE_QLORA = True                      # True = 4-bit QLoRA, False = 常規 LoRA
LORA_R = 64 # 32
LORA_ALPHA = LORA_R * 0.5
LORA_DROPOUT = 0.05
# Regex: only match language_model layers, exclude vision_tower (Gemma4ClippableLinear)
LORA_TARGET_MODULES = r".*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"

# ---------- 訓練超參數 ----------
LEARNING_RATE = 5e-5
BATCH_SIZE = 2
GRAD_ACCUMULATION_STEPS = 32 // BATCH_SIZE          # effective batch = 2 * 16 = 32
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 2048 # 1536
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "cosine"

# ---------- 資料 ----------
TRAIN_JSON = "data/train.json"
TEST_JSON = "data/test.json"
SAMPLE_SUBMISSION = "data/sample_submission.csv"
VAL_RATIO = 0.1                       # 從 train 切出的驗證比例

# ---------- 輸出 ----------
BASE_OUTPUT_DIR = "outputs/"

# ---------- 其他 ----------
SEED = 42
FP16 = False                          # QLoRA 下建議使用 bf16
BF16 = True
LOGGING_STEPS = 10
EVAL_EVERY_N_EPOCH = 0.25              # 每 n 個 epoch 做一次 validation
SAVE_STRATEGY = "epoch"
USE_FLASH_ATTENTION = False            # flash_attn_2 與 torch 2.6 不相容，sdpa 已內建 flash kernel
NUM_LABELS = 4                         # 分類類別數

# ---------- 資料擴增 ----------
AUG_POSITION_SWAP = True              # 策略1: 位置交換與標籤反轉
AUG_REVERSE_COT = False               # 策略2: 本地逆向思維鏈 (需先跑 generate_cot.py)
AUG_REVERSE_COT_FILE = "data/train_cot.json"  # CoT 生成結果路徑
AUG_PROMPT_DIVERSE = True             # 策略3: Prompt 模板多樣化

# ---------- Test-Time Augmentation ----------
TTA_ENABLED = False                    # 啟用 TTA (多數決)
TTA_POSITION_SWAP = True              # TTA: 原順序 + 反順序
TTA_PROMPT_TEMPLATES = [0, 1, 2, 3]   # TTA: 使用哪些 prompt 模板 (索引)
