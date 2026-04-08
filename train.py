# ============================================================
# train.py — LoRA / QLoRA 微調主程式 (Unsloth)
# ============================================================

import os
import shutil
from datetime import datetime

import torch
from transformers import TrainerCallback
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from collator import DataCollatorForCompletionOnlyLM
from datasets import Dataset as HFDataset

import config
from dataset import JudgeDataset, load_train_val, get_response_template_ids, build_prompt


class EvalAccCallback(TrainerCallback):
    """每次 eval 後用 generation 算 val accuracy，並將 checkpoint 重新命名為
    ckpt_{epoch}_{val_acc} 格式。"""

    def __init__(self, val_data, tokenizer, run_dir):
        self.val_data = val_data
        # Unsloth may return a Processor; extract the underlying tokenizer for encoding
        self.tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
        self.run_dir = run_dir
        self._last_val_acc = 0.0
        self.best_val_acc = 0.0
        self.best_ckpt = None

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        # Switch to inference mode for generation
        FastLanguageModel.for_inference(model)
        correct = 0
        total = len(self.val_data)

        for sample in self.val_data:
            prompt = build_prompt(
                sample["dialog_1"], sample["dialog_2"], self.tokenizer
            )
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=config.MAX_SEQ_LENGTH,
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=3, do_sample=False)
            pred = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip().lower()
            pred = pred.split()[0] if pred else ""
            if pred == sample["verdict"].lower():
                correct += 1

        self._last_val_acc = correct / total if total > 0 else 0
        print(f"[EVAL] Val Accuracy: {self._last_val_acc:.4f} ({correct}/{total})")

        # Switch back to training mode
        FastLanguageModel.for_training(model)

        # Rename the checkpoint saved just before this evaluation
        ckpt_dir = os.path.join(self.run_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(ckpt_dir):
            epoch_str = f"{state.epoch:.2f}"
            acc_str = f"{self._last_val_acc:.4f}"
            new_name = os.path.join(self.run_dir, f"ckpt_{epoch_str}_{acc_str}")
            os.rename(ckpt_dir, new_name)
            print(f"[INFO] Checkpoint renamed → {os.path.basename(new_name)}")
            if self._last_val_acc > self.best_val_acc:
                self.best_val_acc = self._last_val_acc
                self.best_ckpt = new_name


# ============================================================
# 0. 全域 Seed 固定 (確保可復現)
# ============================================================
import random
import numpy as np

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
print(f"[INFO] 實驗目錄: {run_dir}")

# ============================================================
# 2. 載入資料
# ============================================================
train_data, val_data = load_train_val()

# ============================================================
# 3. 載入模型 & Tokenizer (Unsloth)
# ============================================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config.MODEL_NAME,
    max_seq_length=config.MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=config.USE_QLORA,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================================================
# 4. 套用 LoRA (Unsloth optimized)
# ============================================================
model = FastLanguageModel.get_peft_model(
    model,
    r=config.LORA_R,
    lora_alpha=config.LORA_ALPHA,
    lora_dropout=config.LORA_DROPOUT,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=config.SEED,
    use_rslora=False,
    max_seq_length=config.MAX_SEQ_LENGTH,
)
model.print_trainable_parameters()

# ============================================================
# 5. 建立 Dataset
# ============================================================
train_dataset = JudgeDataset(
    train_data, tokenizer,
    use_diverse_prompt=config.AUG_PROMPT_DIVERSE,
    use_cot=config.AUG_REVERSE_COT,
)
val_dataset = JudgeDataset(val_data, tokenizer)
print(f"[INFO] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
aug_flags = (f"  Position Swap: {config.AUG_POSITION_SWAP}, "
             f"Reverse-CoT: {config.AUG_REVERSE_COT}, "
             f"Prompt Diverse: {config.AUG_PROMPT_DIVERSE}")
print(f"[INFO] 擴增策略:{aug_flags}")

# ============================================================
# 6. DataCollator — 只對 Verdict 部分計算 Loss
# ============================================================
response_template_ids = get_response_template_ids(tokenizer)
print(f"[INFO] Response template IDs: {response_template_ids}")
print(f"[INFO] Response template decoded: {repr(tokenizer.decode(response_template_ids))}")
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids,
    tokenizer=tokenizer,
)

# ============================================================
# 7. 訓練參數
# ============================================================
import math
steps_per_epoch = math.ceil(len(train_dataset) / (config.BATCH_SIZE * config.GRAD_ACCUMULATION_STEPS))
eval_steps = max(1, int(steps_per_epoch * config.EVAL_EVERY_N_EPOCH))
print(f"[INFO] Steps per epoch: {steps_per_epoch}, eval/save every {eval_steps} steps "
      f"(every {config.EVAL_EVERY_N_EPOCH} epoch)")

training_args = SFTConfig(
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
    save_total_limit=None,
    seed=config.SEED,
    report_to="none",
    optim="adamw_8bit",
    max_length=config.MAX_SEQ_LENGTH,
    dataset_text_field="text",
    dataloader_num_workers=8,
    dataloader_prefetch_factor=2,
)

# ============================================================
# 8. 啟動 SFTTrainer 微調
# ============================================================
hf_train = HFDataset.from_dict({"text": [train_dataset[i]["text"] for i in range(len(train_dataset))]})
hf_val = HFDataset.from_dict({"text": [val_dataset[i]["text"] for i in range(len(val_dataset))]})

eval_callback = EvalAccCallback(val_data, tokenizer, run_dir)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=hf_train,
    eval_dataset=hf_val,
    args=training_args,
    data_collator=collator,
    callbacks=[eval_callback],
)

print("[INFO] 開始訓練...")
trainer.train()

# ============================================================
# 9. 儲存最佳 Adapter 權重
# ============================================================
final_dir = os.path.join(run_dir, "final_adapter")
if eval_callback.best_ckpt and os.path.exists(eval_callback.best_ckpt):
    shutil.copytree(eval_callback.best_ckpt, final_dir)
    print(f"[INFO] Best checkpoint (acc={eval_callback.best_val_acc:.4f}): "
          f"{os.path.basename(eval_callback.best_ckpt)}")
else:
    model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"[INFO] Adapter 已儲存至: {final_dir}")
print("[INFO] 訓練完成!")
