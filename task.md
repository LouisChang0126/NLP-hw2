# Task: Migrate `train.py` to Unsloth Framework

## 📍 Context & Objective
We are currently fine-tuning a Large Language Model (`google/gemma-4-E4B-it`) for an "LLM-as-a-Judge" task using QLoRA. 
Currently, the codebase relies on native Hugging Face `transformers` and `peft`, along with a manual Monkey Patch for chunked cross-entropy (`_chunked_ce_forward`) and manual deletion of multimodal layers to prevent VRAM OOM.

**Objective:** Refactor `train.py` (and relevant parts of `config.py`) to use the **Unsloth** framework (`unsloth.FastLanguageModel`). Unsloth provides heavily optimized Triton kernels for RoPE, Attention, Cross-Entropy, and LoRA, which will naturally solve our VRAM issues and speed up training by 2x. I want to fine-tuning `unsloth/gemma-4-E4B-it-bnb-4bit`

## 🛠️ Specific Instructions

### 1. Clean Up Legacy Hacks in `train.py`
Since Unsloth handles memory-efficient training natively, please **DELETE** the following legacy workarounds in `train.py`:
- Delete the `_chunked_ce_forward` function and the monkey-patching logic (`model._orig_forward = model.forward`, etc.).
- Delete the custom `MemoryEfficientSFTTrainer` class.
- Delete the manual layer deletion logic (`del model.model.vision_tower`, etc.).
- Remove `BitsAndBytesConfig` and `prepare_model_for_kbit_training` imports, as Unsloth handles 4-bit loading internally.

### 2. Refactor Model & Tokenizer Loading
Replace the `AutoModelForCausalLM` loading logic with `FastLanguageModel.from_pretrained`.
- Import: `from unsloth import FastLanguageModel`
- Setup parameters:
  - `model_name = config.MODEL_NAME`
  - `max_seq_length = config.MAX_SEQ_LENGTH`
  - `dtype = None` (Auto-detect)
  - `load_in_4bit = config.USE_QLORA`
- The `FastLanguageModel.from_pretrained` call returns BOTH `model` and `tokenizer`. Please use the returned tokenizer instead of loading it separately via `AutoTokenizer`.
- Ensure `tokenizer.padding_side = "right"` is set after loading.

### 3. Refactor PEFT / LoRA Application
Replace the standard `get_peft_model` and `LoraConfig` with Unsloth's optimized method:
- Use `model = FastLanguageModel.get_peft_model(...)`
- Pass in parameters from `config.py`: `r`, `lora_alpha`, `lora_dropout`, `target_modules`.
- Set `bias="none"`, `use_gradient_checkpointing="unsloth"`, `random_state=config.SEED`, and `use_rslora=False`.

### 4. SFTTrainer Adjustments
- Revert back to using the standard `SFTTrainer` from the `trl` library (instead of `MemoryEfficientSFTTrainer`).
- Keep the existing `DataCollatorForCompletionOnlyLM` logic. This is crucial for our task so we only compute loss on the verdict.
- Ensure `dataset_text_field="text"` and `max_seq_length=config.MAX_SEQ_LENGTH` are passed properly to the SFTTrainer.
- You may remove `use_liger_kernel=True` from `SFTConfig` as Unsloth replaces the need for Liger.

### 5. Config Updates (`config.py`)
- Ensure `MAX_SEQ_LENGTH` is appropriately set (e.g., `2048`).
- Check if `MODEL_NAME` needs to be changed. (If Unsloth has a pre-quantized 4-bit GGUF/HF version like `"unsloth/gemma-4-E4B-it-bnb-4bit"`, recommend adding it as a comment for fast downloading, but keep the default `google/gemma-4-E4B-it` working).

### 6. Saving the Model
Ensure the final model and tokenizer are saved correctly at the end of the script using standard `model.save_pretrained(final_dir)` and `tokenizer.save_pretrained(final_dir)`.

## ✅ Acceptance Criteria
- `train.py` runs successfully using `FastLanguageModel`.
- The manual chunked CE patch is fully removed.
- VRAM usage is stable, and training speed is improved.
- The verdict masking (via `DataCollatorForCompletionOnlyLM`) remains functional.

Please review the current `train.py` and `config.py` and apply these changes step by step.