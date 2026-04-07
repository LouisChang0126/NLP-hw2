# ============================================================
# dataset.py — 資料讀取、對話展平、Prompt 組合
# ============================================================

import json
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

# -------------------- Prompt Template --------------------
SYSTEM_PROMPT = (
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

def build_prompt(dialog_1: list[dict], dialog_2: list[dict]) -> str:
    """組合展平後的兩段對話為完整 prompt。"""
    flat_1 = flatten_dialog(dialog_1)
    flat_2 = flatten_dialog(dialog_2)
    prompt = (
        f"{SYSTEM_PROMPT}"
        f"### Response A\n{flat_1}\n\n"
        f"### Response B\n{flat_2}\n\n"
        f"### Verdict\n"
    )
    return prompt

def build_train_text(sample: dict) -> str:
    """建立完整的訓練文本 (prompt + label)。"""
    prompt = build_prompt(sample["dialog_1"], sample["dialog_2"])
    verdict = sample["verdict"]
    return prompt + verdict

# -------------------- Dataset 類別 --------------------
class JudgeDataset(Dataset):
    """用於 SFTTrainer 的 Dataset，回傳 dict 含 'text' 欄位。"""

    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = build_train_text(sample)
        return {"text": text}

# -------------------- 載入輔助 --------------------
def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_train_val(val_ratio: float = None):
    """讀取 train.json 並切分為 train / val。"""
    if val_ratio is None:
        val_ratio = config.VAL_RATIO
    data = load_json(config.TRAIN_JSON)

    import random
    random.seed(config.SEED)
    random.shuffle(data)

    split = int(len(data) * (1 - val_ratio))
    return data[:split], data[split:]
