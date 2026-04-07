為了進一步**榨乾顯存**並**大幅提升訓練速度**，以下還有幾個常常被忽略，但效果極佳的最佳化策略：

### 1. 暴力釋放顯存：刪除未使用的多模態 Encoder (針對 Gemma 4)
**效果：大幅減少 VRAM 佔用（可省下數 GB）**
Gemma 4 是一個多模態模型（Multimodal），它的底層包含了 Vision Tower 和 Audio Tower。即使你只餵給它純文字（Text-only），這些龐大的 CNN/ViT 權重依然會被載入並死死佔據你的 GPU VRAM。
既然你的任務是純文本對話評估，你可以直接在記憶體中「物理刪除」這些模塊。

* **實作方法：** 在 `train.py` 載入模型（`model = AutoModelForCausalLM.from_pretrained(...)`）之後，立刻加上這幾行程式碼：
    ```python
    # 刪除不需要的多模態 Encoder 以釋放 VRAM
    if hasattr(model.model, "vision_tower"):
        del model.model.vision_tower
    if hasattr(model.model, "audio_tower"):
        del model.model.audio_tower
    if hasattr(model.model, "image_encoder"): # 根據具體的層名稱可能有所不同
        del model.model.image_encoder

    import gc
    torch.cuda.empty_cache()
    gc.collect()
    ```
### 2. 終極開源神器：使用 Unsloth 框架 (強烈推薦)
**效果：訓練速度提升 2x，顯存額外減少 40%**
你現在自己手刻了 `_chunked_ce_forward`，非常厲害。但開源界目前針對 LoRA 訓練最佳的解決方案是 [Unsloth](https://github.com/unslothai/unsloth)。
Unsloth 用 Triton 重寫了 RoPE、Cross-Entropy 和 LoRA 的底層 Kernel，其原生的 Chunked Cross Entropy 與記憶體管理比原生的 HuggingFace 優秀非常多。
* 如果你的環境可以安裝 `unsloth`，它能無縫接軌你的 `SFTTrainer`。使用 Unsloth 的 `FastLanguageModel` 載入模型，你幾乎可以把 `MAX_SEQ_LENGTH` 輕鬆開到 4096 而不擔心 OOM，並且訓練速度會翻倍。

### 3. 推論期 (Inference) 的極致加速
在你的 `inference.py` 中，因為使用了 TTA (Test-Time Augmentation)，推論次數高達數千次。
* **動態 Batch Size：** 推論時不計算梯度，VRAM 消耗極小。你可以把 `inference.py` 的 `--batch_size` 大膽拉高到 `8` 甚至 `16`。