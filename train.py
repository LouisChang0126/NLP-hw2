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
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from collator import DataCollatorForCompletionOnlyLM
from datasets import Dataset as HFDataset
import torch.nn as nn

import config
from dataset import JudgeDataset, load_train_val, get_response_template_ids


def _chunked_ce_forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
    """Monkey-patch for Gemma4ForConditionalGeneration.forward:
    Process hidden_states through lm_head in chunks to avoid allocating
    the full [n_tokens, vocab_size=262144] logits tensor all at once.
    """
    import torch.nn.functional as F
    from transformers.models.gemma4.modeling_gemma4 import Gemma4CausalLMOutputWithPast

    # Strip flags that are not part of the original API
    for key in ("skip_logits", "return_token_accuracy", "use_token_scaling",
                "num_items_in_batch"):
        kwargs.pop(key, None)

    # Only pass kwargs that Gemma4Model accepts (text-only training: no pixel_values etc.)
    body_kwargs = {k: v for k, v in kwargs.items()
                   if k in ("position_ids", "past_key_values", "inputs_embeds",
                             "mm_token_type_ids", "image_position_ids",
                             "video_position_ids", "input_features_mask",
                             "pixel_values", "pixel_values_videos", "input_features")}

    # Run the model body (vision/audio/language encoder) — skip lm_head
    body_out = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
        **body_kwargs,
    )
    hidden_states = body_out.last_hidden_state  # [batch, seq_len, hidden_dim]

    loss = None
    if labels is not None:
        # Align hidden states and labels (teacher forcing: predict next token)
        # hidden_states[:, :-1, :] predicts labels[:, 1:]
        hs = hidden_states[:, :-1, :].contiguous()   # [batch, seq_len-1, H]
        shift_labels = labels[:, 1:].contiguous()     # [batch, seq_len-1]

        # Apply attention mask to select valid (non-padding) positions
        if attention_mask is not None:
            shift_attn = attention_mask[:, -hs.shape[1]:]
            valid_pos = shift_attn.bool()
        else:
            valid_pos = shift_labels != -100

        flat_hs = hs[valid_pos]            # [n_valid, H]
        flat_labels = shift_labels[valid_pos]  # [n_valid]

        # Further filter -100 (masked) positions
        keep = flat_labels != -100
        flat_hs = flat_hs[keep]
        flat_labels = flat_labels[keep]

        n_tokens = flat_hs.shape[0]
        softcap = getattr(self.config.get_text_config(), "final_logit_softcapping", None)

        # Chunked lm_head + CE: each chunk needs [CHUNK, vocab] in float32 (~67 MB at 64 tokens)
        CHUNK = 64
        loss_parts = []
        for start in range(0, n_tokens, CHUNK):
            chunk_hs = flat_hs[start: start + CHUNK]
            chunk_labels = flat_labels[start: start + CHUNK]
            chunk_logits = self.lm_head(chunk_hs).float()  # [chunk, vocab_size]
            if softcap is not None:
                chunk_logits = (chunk_logits / softcap).tanh() * softcap
            ce = F.cross_entropy(chunk_logits, chunk_labels, reduction="sum")
            loss_parts.append(ce)
            del chunk_logits  # free immediately

        if loss_parts and n_tokens > 0:
            loss = torch.stack(loss_parts).sum() / n_tokens
        else:
            loss = hidden_states.sum() * 0  # differentiable zero

    return Gemma4CausalLMOutputWithPast(
        loss=loss,
        logits=None,          # not needed for training; avoids 1+ GB allocation
        past_key_values=body_out.past_key_values,
        hidden_states=body_out.hidden_states,
        attentions=body_out.attentions,
        image_hidden_states=body_out.image_hidden_states,
        audio_hidden_states=body_out.audio_hidden_states,
    )


class MemoryEfficientSFTTrainer(SFTTrainer):
    """Use chunked cross-entropy loss; model returns None for logits to save memory."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs["use_cache"] = False
        inputs.pop("_prediction_loss_only", None)
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is None:
            raise ValueError("Model did not return a loss. Ensure `labels` are in inputs.")
        return (loss, outputs) if return_outputs else loss


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
        device_map="cuda:0",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
else:
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        device_map="cuda:0",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

# Apply chunked cross-entropy monkey-patch to avoid OOM from full logits materialization
import types
model._orig_forward = model.forward
model.forward = types.MethodType(_chunked_ce_forward, model)

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
    save_strategy=config.SAVE_STRATEGY,
    eval_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    seed=config.SEED,
    report_to="none",
    optim="paged_adamw_8bit" if config.USE_QLORA else "adamw_torch",
    max_length=config.MAX_SEQ_LENGTH,
    dataset_text_field="text",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    use_liger_kernel=True,
)

# ============================================================
# 8. 啟動 SFTTrainer 微調
# ============================================================
# Convert torch Datasets to HuggingFace Datasets (required by trl 1.0+)
hf_train = HFDataset.from_dict({"text": [train_dataset[i]["text"] for i in range(len(train_dataset))]})
hf_val = HFDataset.from_dict({"text": [val_dataset[i]["text"] for i in range(len(val_dataset))]})

trainer = MemoryEfficientSFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=hf_train,
    eval_dataset=hf_val,
    args=training_args,
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
