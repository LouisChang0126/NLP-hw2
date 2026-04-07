以下是詳細的 Code Review 與修改建議：

### 🚨 1. 致命問題：微調時計算了 Prompt 的 Loss (在 `train.py`)
**問題描述：** 在使用 `SFTTrainer` 且只提供 `text` 欄位時，預設會對**整段文字（包含你的 Prompt 和前面的對話）**計算 Cross-Entropy Loss。這會導致模型花費大量精力去學習「如何背下對話歷史」和「背下系統提示詞」，而不是專注學習「如何預測 Verdict」，嚴重拖垮分類能力。

**解決方案：** 必須使用 `trl` 提供的 `DataCollatorForCompletionOnlyLM`，讓模型**只對生成的目標（Verdict 或 Rationale）計算 Loss**。

**修改 `train.py`：**
```python
# 在檔案上方新增 import
from trl import DataCollatorForCompletionOnlyLM

# ... [前面的程式碼保持不變] ...

# 在設定 SFTTrainer 之前，加入 DataCollator
# 告訴 Trainer，只有在 "### Verdict\n", "Verdict:\n" 等字眼後面的才計算 Loss
# 為了相容你的多樣化 Prompt，你可以找一個共同的結束語或分別設定，或者最簡單的是針對預設模板：
response_template = "### Verdict\n" 
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
    max_seq_length=config.MAX_SEQ_LENGTH,
    data_collator=collator, # <--- 加上這行！
)
```
*(注意：如果開啟了 `use_diverse_prompt`，你需要確保所有模板的 response_template 都能被 tokenizer 正確切分，為求穩定，建議微調階段先統一使用單一模板，或自訂一個特殊的標籤如 `<|judge_start|>`)*

### 🚨 2. 致命問題：CoT 推論長度被截斷 (在 `inference.py`)
**問題描述：** 如果在 `config.py` 開啟了 `AUG_REVERSE_COT = True`，你的模型在訓練時會學到先輸出「一長串理由」，然後才輸出「A/B/tie/neither」。但是，你在 `inference.py` 中設定了 **`max_new_tokens=5`**！這會導致推論時，模型只吐出理由的前 5 個字就被強制停止了，永遠拿不到最終的 Verdict。

**解決方案：**
推論長度與解析邏輯必須隨著 CoT 是否開啟而變動。

**修改 `inference.py`：**
```python
# 將 max_new_tokens 加長
max_tokens = 250 if config.AUG_REVERSE_COT else 5

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens, # <--- 修正這裡
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

# 並且 parse_verdict 必須從「最後面」開始找 A/B/tie/neither，而不是看第一個 token。
def parse_verdict(generated_text: str, prompt: str) -> str:
    answer = generated_text[len(prompt):].strip()
    
    # 如果有 CoT，答案通常在最後一行，可以這樣取：
    lines = answer.split('\n')
    last_line = lines[-1].strip() if lines else answer
    
    # ... 然後再對 last_line 做原本的字串比對 ...
```

### 💡 3. 強烈建議：善用 Instruction Model 的 Chat Template (在 `dataset.py`)
**問題描述：** 你目前使用的是純文字模板 (`User: ... \n Assistant: ...`)。但 `gemma-3-4b-it` (或 `Qwen-Instruct`) 這類模型在預訓練時，高度依賴特定的控制標籤 (例如 `<start_of_turn>user\n...<end_of_turn>\n`)。如果你不用它原生的 Chat Template，模型的智商會打折。

**解決方案：** 利用 tokenizer 的 `apply_chat_template` 來包裝你的 Prompt。

**建議改法：** 與其自己手刻 `SYSTEM_PROMPT_0`，不如將整包東西視為一個 user 指令發給模型：
```python
def build_prompt(dialog_1: list[dict], dialog_2: list[dict], tokenizer) -> str:
    flat_1 = flatten_dialog(dialog_1)
    flat_2 = flatten_dialog(dialog_2)
    
    user_content = (
        "You are an impartial judge. Compare the two responses and choose the best one.\n\n"
        f"### Response A\n{flat_1}\n\n"
        f"### Response B\n{flat_2}\n\n"
        "Output ONLY your verdict (A, B, tie, neither)."
    )
    
    messages = [
        {"role": "user", "content": user_content}
    ]
    # 這會自動加上 Gemma / Qwen 特定的 special tokens
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt
```
*(備註：如果採用此寫法，上面的 `DataCollatorForCompletionOnlyLM` 也要改成抓取特定 Chat Template 的 `assistant` 開頭標籤。)*

### ⚡ 4. 效能優化：Inference 沒用到 Batch (在 `inference.py`)
**問題描述：** 你有解析 `--batch_size` 參數，但在 `for sample in tqdm(test_data):` 裡面你是一筆一筆 (Batch size = 1) 丟給 GPU 的。RTX 4090 跑 batch size=1 是巨大的浪費，推論會非常慢。

**解決方案：**
將推論改成批次處理 (Batched Inference)，並**記得把 padding_side 改成 "left"**。

```python
# 1. 確保 tokenizer 用於推論時為 left padding
tokenizer.padding_side = "left"

# 2. 實作批次迴圈
batch_size = args.batch_size
results = []
for i in tqdm(range(0, len(test_data), batch_size), desc="Batched Inference"):
    batch_samples = test_data[i : i + batch_size]
    prompts = [build_prompt(s["dialog_1"], s["dialog_2"]) for s in batch_samples]
    
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, # 需要 padding
        truncation=True, 
        max_length=config.MAX_SEQ_LENGTH
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        
    for j, output in enumerate(outputs):
        generated = tokenizer.decode(output, skip_special_tokens=True)
        verdict = parse_verdict(generated, prompts[j])
        results.append({"id": batch_samples[j]["id"], "verdict": verdict})
```

### 🏆 5. 訓練細節：抓取最佳 Checkpoint (在 `train.py`)
在 `config.py` 中你的 `SAVE_STRATEGY = "epoch"`，但是目前的 TrainingArguments 沒有開啟 `load_best_model_at_end`。這代表如果 Epoch 2 已經是最佳狀態，Epoch 3 過擬合 (Overfitting) 了，你最終儲存的 `final_adapter` 會是比較差的那個。

**修改 `train.py` 中的 `TrainingArguments`：**
```python
training_args = TrainingArguments(
    # ... 前面不變 ...
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,               # 避免 checkpoint 塞爆硬碟
    load_best_model_at_end=True,      # 訓練結束時自動載入 val_loss 最低的權重
    metric_for_best_model="eval_loss" # 依據驗證集 loss 判斷
)
```
