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
        f"### Response B\n{flat_2}\n\n"
        "Output ONLY your verdict (A, B, tie, neither)."
    )

def _user_content_1(flat_1: str, flat_2: str) -> str:
    return (
        "You are an objective evaluator. Below are two conversations between a user "
        "and two different AI assistants. Determine which assistant provided "
        "the better response.\n\n"
        f"[Conversation 1]\n{flat_1}\n\n"
        f"[Conversation 2]\n{flat_2}\n\n"
        "Answer with exactly one of: A, B, tie, neither"
    )

def _user_content_2(flat_1: str, flat_2: str) -> str:
    return (
        "As a strict linguist and quality assessor, compare the following two AI "
        "assistant responses. Judge based on helpfulness, accuracy, and clarity.\n\n"
        f"--- Assistant A ---\n{flat_1}\n\n"
        f"--- Assistant B ---\n{flat_2}\n\n"
        "Respond with only: A, B, tie, or neither"
    )

def _user_content_3(flat_1: str, flat_2: str) -> str:
    return (
        "You are a judge in a blind comparison. Two AI assistants answered "
        "the same question. Decide which answer is better, or if they are equal.\n\n"
        f"<<Response A>>\n{flat_1}\n\n"
        f"<<Response B>>\n{flat_2}\n\n"
        "Output one of: A, B, tie, neither"
    )

_USER_CONTENTS = [_user_content_0, _user_content_1, _user_content_2, _user_content_3]

# ============================================================
# Prompt 建構 (使用 Chat Template)
# ============================================================

def build_prompt(dialog_1: list[dict], dialog_2: list[dict],
                 tokenizer, template_id: int = 0) -> str:
    """使用 tokenizer 的 chat template 組合 prompt。"""
    flat_1 = flatten_dialog(dialog_1)
    flat_2 = flatten_dialog(dialog_2)
    user_content = _USER_CONTENTS[template_id](flat_1, flat_2)

    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt

def get_response_template(tokenizer) -> str:
    """自動偵測 chat template 中 model/assistant 回覆的起始標記。
    用於 DataCollatorForCompletionOnlyLM。"""
    dummy = [{"role": "user", "content": "X"}]
    without = tokenizer.apply_chat_template(
        dummy, tokenize=False, add_generation_prompt=False
    )
    with_gen = tokenizer.apply_chat_template(
        dummy, tokenize=False, add_generation_prompt=True
    )
    # generation prompt = with_gen 比 without 多出來的部分
    response_template = with_gen[len(without):]
    if not response_template:
        # fallback: 嘗試常見格式
        response_template = "<start_of_turn>model\n"
    return response_template

def build_train_text(sample: dict, tokenizer,
                     use_diverse_prompt: bool = False,
                     use_cot: bool = False) -> str:
    """建立完整的訓練文本 (prompt + [rationale +] label)。"""
    tid = random.randint(0, len(_USER_CONTENTS) - 1) if use_diverse_prompt else 0
    verdict = sample["verdict"]
    prompt = build_prompt(sample["dialog_1"], sample["dialog_2"], tokenizer, tid)

    if use_cot and sample.get("rationale"):
        return prompt + sample["rationale"].strip() + "\n" + verdict
    else:
        return prompt + verdict

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
    """用於 SFTTrainer 的 Dataset，回傳 dict 含 'text' 欄位。"""

    def __init__(self, data: list[dict], tokenizer,
                 use_diverse_prompt: bool = False,
                 use_cot: bool = False):
        self.data = data
        self.tokenizer = tokenizer
        self.use_diverse_prompt = use_diverse_prompt
        self.use_cot = use_cot

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = build_train_text(
            sample, self.tokenizer,
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

    # 後面的策略 2 (CoT) 和 策略 1 (Position Swap) 邏輯完全不用動，保持原樣
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
