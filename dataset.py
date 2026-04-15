# ============================================================
# dataset.py — 資料讀取、對話展平、Prompt 組合、資料擴增
# ============================================================

import json
import copy
import random
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

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
# Prompt 內容多樣化 (策略 3) — 只變 user message 內容
# ============================================================

def _user_content_0(flat_1: str, flat_2: str) -> str:
    return (
        "You are a fair and impartial judge. Compare the two AI assistant responses "
        "and decide which one is better.\n\n"
        "Rules:\n"
        "- A: Response A is better\n"
        "- B: Response B is better\n"
        "- tie: Both responses are equally good\n"
        "- neither: Both responses are equally bad\n\n"
        f"### Response A\n{flat_1}\n\n"
        f"### Response B\n{flat_2}"
    )

def _user_content_1(flat_1: str, flat_2: str) -> str:
    return (
        "You are an objective evaluator. Below are two conversations between a user "
        "and two different AI assistants. Determine which assistant provided "
        "the better response.\n\n"
        f"[Conversation 1]\n{flat_1}\n\n"
        f"[Conversation 2]\n{flat_2}"
    )

def _user_content_2(flat_1: str, flat_2: str) -> str:
    return (
        "As a strict linguist and quality assessor, compare the following two AI "
        "assistant responses. Judge based on helpfulness, accuracy, and clarity.\n\n"
        f"--- Assistant A ---\n{flat_1}\n\n"
        f"--- Assistant B ---\n{flat_2}"
    )

def _user_content_3(flat_1: str, flat_2: str) -> str:
    return (
        "You are a judge in a blind comparison. Two AI assistants answered "
        "the same question. Decide which answer is better, or if they are equal.\n\n"
        f"<<Response A>>\n{flat_1}\n\n"
        f"<<Response B>>\n{flat_2}"
    )

_USER_CONTENTS = [_user_content_0, _user_content_1, _user_content_2, _user_content_3]

# ============================================================
# Prompt 建構 (使用 Chat Template)
# ============================================================

def build_prompt(dialog_1: list[dict], dialog_2: list[dict],
                 tokenizer, template_id: int = 0) -> str:
    """使用 tokenizer 的 chat template 組合 prompt（分類模式，不加 generation prompt）。"""
    flat_1 = flatten_dialog(dialog_1)
    flat_2 = flatten_dialog(dialog_2)
    user_content = _USER_CONTENTS[template_id](flat_1, flat_2)

    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return prompt

# ============================================================
# 策略 1: 位置交換與標籤反轉
# ============================================================
SWAP_VERDICT = {"A": "B", "B": "A", "tie": "tie", "neither": "neither"}

def position_swap(data: list[dict]) -> list[dict]:
    """對每筆資料產生位置交換版本。"""
    swapped = []
    for sample in data:
        new = copy.deepcopy(sample)
        new["dialog_1"], new["dialog_2"] = sample["dialog_2"], sample["dialog_1"]
        new["verdict"] = SWAP_VERDICT[sample["verdict"]]
        if "rationale" in new:
            del new["rationale"]
        new["id"] = f"{sample['id']}_swap"
        swapped.append(new)
    return swapped

# ============================================================
# Dataset 類別
# ============================================================
class JudgeDataset(Dataset):
    """序列分類 Dataset，回傳 dict 含 'text' (str) 與 'labels' (int) 欄位。"""

    def __init__(self, data: list[dict], tokenizer,
                 use_diverse_prompt: bool = False):
        self.data = data
        self.tokenizer = tokenizer
        self.use_diverse_prompt = use_diverse_prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        tid = random.randint(0, len(_USER_CONTENTS) - 1) if self.use_diverse_prompt else 0
        text = build_prompt(sample["dialog_1"], sample["dialog_2"], self.tokenizer, tid)
        label = LABEL2ID[sample["verdict"]]
        return {"text": text, "labels": label}

# ============================================================
# 載入輔助
# ============================================================
def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_train_val(val_ratio: float = None):
    """讀取 train.json，套用擴增策略，以分層抽樣切分為 train / val。"""
    if val_ratio is None:
        val_ratio = config.VAL_RATIO
    data = load_json(config.TRAIN_JSON)

    # --- 修改點：使用 Stratified Split 取代純隨機切分 ---
    # 提取所有資料的 verdict 作為分層的依據
    verdicts = [sample["verdict"] for sample in data]
    
    train_data, val_data = train_test_split(
        data, 
        test_size=val_ratio, 
        random_state=config.SEED, 
        stratify=verdicts  # <--- 關鍵：保證 train 和 val 的 4 個 class 比例一致
    )
    # --------------------------------------------------

    # --- 策略 1: 位置交換 ---
    if config.AUG_POSITION_SWAP:
        swapped = position_swap(train_data)
        train_data = train_data + swapped
        random.seed(config.SEED)
        random.shuffle(train_data)
        print(f"[AUG] Position Swap: 訓練資料擴增至 {len(train_data)} 筆")

    return train_data, val_data
