# ============================================================
# train.py — LoRA / QLoRA 微調主程式
# ============================================================

import os
import shutil
from datetime import datetime

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import config
from dataset import JudgeDataset, load_train_val, get_response_template

# ============================================================
# 1. 動態輸出目錄與 config 備份
# ============================================================
timestamp = datetime.now().strftime("%m%d%H%S")
model_short_name = config.MODEL_NAME.rstrip("/").split("/")[-1]
run_dir = os.path.join(config.BASE_OUTPUT_DIR, f"{model_short_name}_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

# 備份 config.py 到實驗目錄
shutil.copy("config.py", os.path.join(run_dir, "config.py"))
print(f"[INFO] 實驗目錄: {run_dir}")

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
# 3.5 建立 Dataset (需要 tokenizer)
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
# 4. 載入模型 (QLoRA 4-bit 或常規)
# ============================================================
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
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

# ============================================================
# 5. 套用 LoRA
# ============================================================
lora_config = LoraConfig(
    r=config.LORA_R,
    lora_alpha=config.LORA_ALPHA,
    lora_dropout=config.LORA_DROPOUT,
    target_modules=config.LORA_TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# 6. DataCollator — 只對 Verdict 部分計算 Loss
# ============================================================
response_template = get_response_template(tokenizer)
print(f"[INFO] Response template: {repr(response_template)}")
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

# ============================================================
# 7. 訓練參數
# ============================================================
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
    save_strategy=config.SAVE_STRATEGY,
    eval_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    seed=config.SEED,
    report_to="none",
    optim="paged_adamw_8bit" if config.USE_QLORA else "adamw_torch",
)

# ============================================================
# 8. 啟動 SFTTrainer 微調
# ============================================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
    max_seq_length=config.MAX_SEQ_LENGTH,
    data_collator=collator,
)

print("[INFO] 開始訓練...")
trainer.train()

# ============================================================
# 9. 儲存最終 Adapter 權重 (已自動載入 best checkpoint)
# ============================================================
final_dir = os.path.join(run_dir, "final_adapter")
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"[INFO] Adapter 已儲存至: {final_dir}")
print("[INFO] 訓練完成!")
