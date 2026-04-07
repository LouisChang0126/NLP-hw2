# ============================================================
# config.py — 全局設定中心
# ============================================================

# ---------- 模型 ----------
MODEL_NAME = "google/gemma-3-4b-it"   # HuggingFace 路徑或本地路徑

# ---------- LoRA / QLoRA ----------
USE_QLORA = True                      # True = 4-bit QLoRA, False = 常規 LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# ---------- 訓練超參數 ----------
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRAD_ACCUMULATION_STEPS = 8           # effective batch = 2 * 8 = 16
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 4096
WARMUP_RATIO = 0.05
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
SAVE_STRATEGY = "epoch"

# ---------- 資料擴增 ----------
AUG_POSITION_SWAP = True              # 策略1: 位置交換與標籤反轉
AUG_REVERSE_COT = False               # 策略2: 本地逆向思維鏈 (需先跑 generate_cot.py)
AUG_REVERSE_COT_FILE = "data/train_cot.json"  # CoT 生成結果路徑
AUG_PROMPT_DIVERSE = True             # 策略3: Prompt 模板多樣化
