# ============================================================
# dataset.py — 資料讀取、對話展平、Prompt 組合、資料擴增
# ============================================================

import json
import copy
import random
from torch.utils.data import Dataset

import config

# -------------------- 標籤映射 --------------------
LABEL2ID = {"A": 0, "B": 1, "tie": 2, "neither": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# -------------------- 對話展平 --------------------
def flatten_dialog(dialog: list[dict]) -> str:
    """將 [{"role": ..., "content": ...}, ...] 展平為易讀純文本。"""
    lines = []
    for msg in dialog:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)

# ============================================================
# Prompt Template 多樣化 (策略 3)
# ============================================================

# --- 模板 0 (預設 / 推論用) ---
SYSTEM_PROMPT_0 = (
    "You are a fair and impartial judge. You will be given a user instruction "
    "and two AI assistant responses (Response A and Response B). "
    "Compare the two responses carefully and decide which one is better.\n\n"
    "Rules:\n"
    "- Output ONLY one of: A, B, tie, neither\n"
    "- A: Response A is better\n"
    "- B: Response B is better\n"
    "- tie: Both responses are equally good\n"
    "- neither: Both responses are equally bad\n"
)

def _template_0(flat_1: str, flat_2: str) -> str:
    return (
        f"{SYSTEM_PROMPT_0}"
        f"### Response A\n{flat_1}\n\n"
        f"### Response B\n{flat_2}\n\n"
        f"### Verdict\n"
    )

# --- 模板 1 ---
_SYS_1 = (
    "You are an objective evaluator. Below are two conversations between a user "
    "and two different AI assistants. Your task is to determine which assistant "
    "provided the better response.\n\n"
    "Answer with exactly one of: A, B, tie, neither\n"
)

def _template_1(flat_1: str, flat_2: str) -> str:
    return (
        f"{_SYS_1}"
        f"[Conversation 1]\n{flat_1}\n\n"
        f"[Conversation 2]\n{flat_2}\n\n"
        f"Which is better? Answer:\n"
    )

# --- 模板 2 ---
_SYS_2 = (
    "As a strict linguist and quality assessor, compare the following two AI "
    "assistant responses to the same user query. Judge based on helpfulness, "
    "accuracy, and clarity.\n\n"
    "Respond with only: A, B, tie, or neither\n"
)

def _template_2(flat_1: str, flat_2: str) -> str:
    return (
        f"{_SYS_2}"
        f"--- Assistant A ---\n{flat_1}\n\n"
        f"--- Assistant B ---\n{flat_2}\n\n"
        f"Verdict:\n"
    )

# --- 模板 3 ---
_SYS_3 = (
    "You will act as a judge in a blind comparison. Two AI assistants answered "
    "the same question. Decide which answer is better, or if they are equal.\n\n"
    "Output one of: A, B, tie, neither\n"
)

def _template_3(flat_1: str, flat_2: str) -> str:
    return (
        f"{_SYS_3}"
        f"<<Response A>>\n{flat_1}\n\n"
        f"<<Response B>>\n{flat_2}\n\n"
        f"Your judgment:\n"
    )

_TEMPLATES = [_template_0, _template_1, _template_2, _template_3]

# ============================================================
# Prompt 建構
# ============================================================

def build_prompt(dialog_1: list[dict], dialog_2: list[dict],
                 template_id: int = 0) -> str:
    """組合展平後的兩段對話為完整 prompt。
    template_id=0 為預設（推論時固定使用）。"""
    flat_1 = flatten_dialog(dialog_1)
    flat_2 = flatten_dialog(dialog_2)
    return _TEMPLATES[template_id](flat_1, flat_2)

def build_prompt_with_cot(dialog_1: list[dict], dialog_2: list[dict],
                          rationale: str, template_id: int = 0) -> str:
    """帶有 CoT 理由的 prompt (策略 2)。"""
    prompt = build_prompt(dialog_1, dialog_2, template_id)
    return prompt + rationale + "\n"

def build_train_text(sample: dict, use_diverse_prompt: bool = False,
                     use_cot: bool = False) -> str:
    """建立完整的訓練文本 (prompt + [rationale +] label)。"""
    if use_diverse_prompt:
        tid = random.randint(0, len(_TEMPLATES) - 1)
    else:
        tid = 0

    verdict = sample["verdict"]

    if use_cot and "rationale" in sample and sample["rationale"]:
        prompt = build_prompt(sample["dialog_1"], sample["dialog_2"], tid)
        return prompt + sample["rationale"].strip() + "\n" + verdict
    else:
        prompt = build_prompt(sample["dialog_1"], sample["dialog_2"], tid)
        return prompt + verdict

# ============================================================
# 策略 1: 位置交換與標籤反轉
# ============================================================
SWAP_VERDICT = {"A": "B", "B": "A", "tie": "tie", "neither": "neither"}

def position_swap(data: list[dict]) -> list[dict]:
    """對每筆資料產生位置交換版本，回傳擴增後的新資料列表。"""
    swapped = []
    for sample in data:
        new = copy.deepcopy(sample)
        new["dialog_1"], new["dialog_2"] = sample["dialog_2"], sample["dialog_1"]
        new["verdict"] = SWAP_VERDICT[sample["verdict"]]
        if "rationale" in new:
            del new["rationale"]  # CoT 理由不適用於交換後
        new["id"] = f"{sample['id']}_swap"
        swapped.append(new)
    return swapped

# ============================================================
# Dataset 類別
# ============================================================
class JudgeDataset(Dataset):
    """用於 SFTTrainer 的 Dataset，回傳 dict 含 'text' 欄位。"""

    def __init__(self, data: list[dict],
                 use_diverse_prompt: bool = False,
                 use_cot: bool = False):
        self.data = data
        self.use_diverse_prompt = use_diverse_prompt
        self.use_cot = use_cot

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = build_train_text(
            sample,
            use_diverse_prompt=self.use_diverse_prompt,
            use_cot=self.use_cot,
        )
        return {"text": text}

# ============================================================
# 載入輔助
# ============================================================
def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_train_val(val_ratio: float = None):
    """讀取 train.json，套用擴增策略，切分為 train / val。"""
    if val_ratio is None:
        val_ratio = config.VAL_RATIO
    data = load_json(config.TRAIN_JSON)

    random.seed(config.SEED)
    random.shuffle(data)

    # 先切分，擴增只用在 train set
    split = int(len(data) * (1 - val_ratio))
    train_data = data[:split]
    val_data = data[split:]

    # --- 策略 2: 合併 CoT 理由 ---
    if config.AUG_REVERSE_COT:
        try:
            cot_data = load_json(config.AUG_REVERSE_COT_FILE)
            cot_map = {item["id"]: item.get("rationale", "") for item in cot_data}
            for sample in train_data:
                if sample["id"] in cot_map:
                    sample["rationale"] = cot_map[sample["id"]]
            print(f"[AUG] Reverse-CoT: 成功載入 {len(cot_map)} 筆理由")
        except FileNotFoundError:
            print(f"[AUG] 警告: CoT 檔案 {config.AUG_REVERSE_COT_FILE} 不存在，跳過")

    # --- 策略 1: 位置交換 ---
    if config.AUG_POSITION_SWAP:
        swapped = position_swap(train_data)
        train_data = train_data + swapped
        random.shuffle(train_data)
        print(f"[AUG] Position Swap: 訓練資料擴增至 {len(train_data)} 筆")

    return train_data, val_data
