### 如果換成 `Qwen/Qwen3-8B` 需要改哪裡？

如果你要將模型切換為純文字架構的 `Qwen/Qwen3-8B`，必須修改以下幾個地方，否則程式會報錯或無法訓練：

#### 必須修改 1：隱藏層維度獲取方式 (`train.py`)
目前的 Gemma 似乎帶有多模態架構，所以你取得 hidden size 的方式是 `model.config.text_config.hidden_size`。純文字的 Qwen 沒有 `text_config` 這一層。
* **修改 `train.py` 第 144 行附近：**
```python
# 原本
# hidden_size = model.config.text_config.hidden_size

# 改成 (適用於標準 Qwen)
hidden_size = model.config.hidden_size
```

#### 必須修改 2：LoRA 目標模組 (`config.py`)
你原本的正規表示式特別指定了 `.*language_model\..*` 以避開 Gemma 內部的視覺 Encoder。標準 Qwen 模型架構沒有 `language_model` 這層前綴，如果沿用原本的 regex，Trainer 會找不到可以訓練的參數（Trainable parameters = 0）。
* **修改 `config.py` 第 13 行：**
```python
# 原本
# LORA_TARGET_MODULES = r".*language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"

# 改成 (針對標準 LLM)
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

#### 必須修改 3：模型名稱 (`config.py`)
* **修改 `config.py` 第 6 行：** 將註解拿掉，改成指定的模型。
```python
MODEL_NAME = "Qwen/Qwen3-8B"
```