# ============================================================
# train.py — LoRA / QLoRA 序列分類微調主程式
# (Gemma4 無原生 SeqCls，改用 CausalLM + 自訂分類頭)
# ============================================================

import gc
import os
import shutil
import types
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    DataCollatorWithPadding,
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score
from datasets import Dataset as HFDataset

import config
from dataset import JudgeDataset, load_train_val, LABEL2ID, ID2LABEL

# ============================================================
# 分類 forward (monkey-patch 到 CausalLM 模型上)
# ============================================================
def _cls_forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
    """Replace causal-LM forward with sequence classification forward."""
    # Strip extra kwargs that may come from Trainer / PEFT
    _allowed = ("position_ids", "past_key_values", "inputs_embeds",
                "cache_position")
    body_kwargs = {k: v for k, v in kwargs.items() if k in _allowed}

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
        **body_kwargs,
    )
    hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]

    # Pool: last non-padding token (works for both left & right padding)
    if attention_mask is not None:
        seq_len = attention_mask.shape[1]
        sequence_lengths = seq_len - 1 - attention_mask.flip(dims=[1]).argmax(dim=1)
    else:
        sequence_lengths = input_ids.shape[1] - 1

    batch_size = hidden_states.shape[0]
    pooled = hidden_states[
        torch.arange(batch_size, device=hidden_states.device),
        sequence_lengths,
    ]
    logits = self.score(pooled).float()

    loss = None
    if labels is not None:
        loss = F.cross_entropy(logits, labels)

    return SequenceClassifierOutputWithPast(loss=loss, logits=logits)


# ============================================================
# 0. 全域 Seed 固定
# ============================================================
import random

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# 1. 動態輸出目錄與 config 備份
# ============================================================
timestamp = datetime.now().strftime("%m%d%H%M")
model_short_name = config.MODEL_NAME.rstrip("/").split("/")[-1]
run_dir = os.path.join(config.BASE_OUTPUT_DIR, f"{model_short_name}_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

shutil.copy("config.py", os.path.join(run_dir, "config.py"))

# --- 同時輸出到 console 和 log.txt ---
import logging
import sys

log_path = os.path.join(run_dir, "log.txt")
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
fh.setFormatter(fmt)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(sh)

def log(msg):
    logger.info(msg)

log(f"實驗目錄: {run_dir}")

# ============================================================
# 2. 載入資料
# ============================================================
train_data, val_data = load_train_val()

# ============================================================
# 3. 載入 Tokenizer
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(
    config.MODEL_NAME,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================================================
# 3.5 建立 Dataset 並 tokenize
# ============================================================
train_dataset = JudgeDataset(
    train_data, tokenizer,
    use_diverse_prompt=config.AUG_PROMPT_DIVERSE,
)
val_dataset = JudgeDataset(val_data, tokenizer)
log(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
log(f"擴增策略: Position Swap: {config.AUG_POSITION_SWAP}, "
    f"Prompt Diverse: {config.AUG_PROMPT_DIVERSE}")


def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
    )
    tokenized["labels"] = examples["labels"]
    return tokenized


def to_hf_dataset(torch_ds):
    texts, labels = [], []
    for i in range(len(torch_ds)):
        item = torch_ds[i]
        texts.append(item["text"])
        labels.append(item["labels"])
    return HFDataset.from_dict({"text": texts, "labels": labels})


hf_train = to_hf_dataset(train_dataset).map(tokenize_function, batched=True, remove_columns=["text"])
hf_val = to_hf_dataset(val_dataset).map(tokenize_function, batched=True, remove_columns=["text"])

# ============================================================
# 4. 載入模型 (CausalLM + 自訂分類頭)
# ============================================================
attn_impl = "sdpa"

if config.USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
else:
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        device_map="cuda:0",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )

# 刪除不需要的多模態 Encoder 以釋放 VRAM
for attr in ("vision_tower", "embed_vision", "audio_tower", "embed_audio"):
    if hasattr(model.model, attr):
        delattr(model.model, attr)
        log(f"已刪除 model.model.{attr}")
torch.cuda.empty_cache()
gc.collect()

# --- 加掛分類頭 & monkey-patch forward ---
hidden_size = (model.config.text_config.hidden_size
               if hasattr(model.config, "text_config")
               else model.config.hidden_size)
model.score = nn.Linear(hidden_size, config.NUM_LABELS, bias=False)
model.score = model.score.to(device=model.device, dtype=torch.bfloat16)
model.forward = types.MethodType(_cls_forward, model)
log(f"分類頭: Linear({hidden_size}, {config.NUM_LABELS})")

# ============================================================
# 5. 套用 LoRA (modules_to_save 確保分類頭一起訓練 & 儲存)
# ============================================================
lora_config = LoraConfig(
    r=config.LORA_R,
    lora_alpha=config.LORA_ALPHA,
    lora_dropout=config.LORA_DROPOUT,
    target_modules=config.LORA_TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["score"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# 6. compute_metrics
# ============================================================
from sklearn.metrics import log_loss as sklearn_log_loss

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    logloss = sklearn_log_loss(labels, probs, labels=[0, 1, 2, 3])
    return {"accuracy": acc, "log_loss": logloss}

# ============================================================
# 7. 訓練參數
# ============================================================
import math
steps_per_epoch = math.ceil(len(hf_train) / (config.BATCH_SIZE * config.GRAD_ACCUMULATION_STEPS))
eval_steps = max(1, int(steps_per_epoch * config.EVAL_EVERY_N_EPOCH))
log(f"Steps per epoch: {steps_per_epoch}, eval/save every {eval_steps} steps "
    f"(every {config.EVAL_EVERY_N_EPOCH} epoch)")

training_args = TrainingArguments(
    output_dir=run_dir,
    num_train_epochs=config.NUM_EPOCHS,
    per_device_train_batch_size=config.BATCH_SIZE,
    per_device_eval_batch_size=config.BATCH_SIZE,
    gradient_accumulation_steps=config.GRAD_ACCUMULATION_STEPS,
    learning_rate=config.LEARNING_RATE,
    lr_scheduler_type=config.LR_SCHEDULER_TYPE,
    warmup_ratio=config.WARMUP_RATIO,
    weight_decay=config.WEIGHT_DECAY,
    fp16=config.FP16,
    bf16=config.BF16,
    logging_steps=config.LOGGING_STEPS,
    save_strategy="steps",
    save_steps=eval_steps,
    eval_strategy="steps",
    eval_steps=eval_steps,
    load_best_model_at_end=False,   # 關掉，自己管理 best model（避免改名衝突）
    save_total_limit=None,
    seed=config.SEED,
    report_to="none",
    optim="paged_adamw_8bit" if config.USE_QLORA else "adamw_torch",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_num_workers=8,
    dataloader_prefetch_factor=2,
)

# ============================================================
# 8. Checkpoint 重命名 Callback
#    on_save  → 記錄剛存好的 checkpoint 路徑
#    on_evaluate → 拿到 accuracy 後立刻重命名
# ============================================================
class RenameCheckpointCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.best_acc = 0.0
        self.best_ckpt = None
        self._pending_ckpt = None       # on_save 暫存

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            self._pending_ckpt = ckpt_dir

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self._pending_ckpt is None:
            return
        acc = metrics.get("eval_accuracy", 0.0)
        epoch = state.epoch or 0.0

        new_name = os.path.join(
            self.output_dir,
            f"ckpt-epoch{epoch:.2f}-acc{acc:.4f}",
        )
        try:
            if os.path.exists(new_name):
                shutil.rmtree(new_name)
            os.rename(self._pending_ckpt, new_name)
            log(f"Checkpoint renamed → {os.path.basename(new_name)}")
        except OSError as e:
            log(f"Checkpoint rename failed: {e}")
            new_name = self._pending_ckpt   # fallback: 用原名

        if acc > self.best_acc:
            self.best_acc = acc
            self.best_ckpt = new_name
            log(f"New best accuracy: {acc:.4f}")

        self._pending_ckpt = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        """把 Trainer 的 training log 也寫進 log.txt。"""
        if logs:
            log(f"step={state.global_step} {logs}")

rename_cb = RenameCheckpointCallback(run_dir)

# ============================================================
# 9. 啟動 Trainer 微調
# ============================================================
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_train,
    eval_dataset=hf_val,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[rename_cb],
)

log("開始訓練...")
trainer.train()

# ============================================================
# 10. 儲存最佳 Adapter 權重 (含分類頭)
# ============================================================
final_dir = os.path.join(run_dir, "final_adapter")
if rename_cb.best_ckpt and os.path.exists(rename_cb.best_ckpt):
    shutil.copytree(rename_cb.best_ckpt, final_dir)
    log(f"Best checkpoint (acc={rename_cb.best_acc:.4f}): "
        f"{os.path.basename(rename_cb.best_ckpt)}")
else:
    model.save_pretrained(final_dir)
    log("No best checkpoint found, saving current model")
tokenizer.save_pretrained(final_dir)
log(f"Adapter 已儲存至: {final_dir}")
log("訓練完成!")
