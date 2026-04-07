以下是針對 LLM-as-a-Judge 任務，純本地端可行的 3 種強大擴增策略，全部新增進入pipeline，要可以在config整這3套分別要不要使用：

### 1. 位置交換與標籤反轉 (Position Swapping / Label Flipping)
這是成效最明顯、成本最低（零算力消耗），且能直接打擊**位置偏差 (Position Bias)** 的核心方法 [cite: 128][cite_start]。模型很容易因為哪個回答先出現就偏好誰。

* **實作邏輯：**
    * 如果原資料 `verdict` 是 `"A"`（偏好 `dialog_1`），你將 `dialog_1` 與 `dialog_2` 的內容對調，並將新資料的 `verdict` 標記為 `"B"`。
    * 如果原資料 `verdict` 是 `"B"`，對調後將 `verdict` 改為 `"A"`。
    * 如果原資料是 `"tie"` 或 `"neither"`，對調後 `verdict` 保持不變。

### 2. 本地端逆向思維鏈生成 (Local Reverse-CoT Generation)
你可以把要微調的基準模型（例如 Qwen3.5-9B）當作「資料標註員」，讓它自己生成訓練用的理由 (Rationales)。

* **實作邏輯：**
    * **步驟一（生成）：** 在正式微調前，寫一個腳本讓本地模型讀取 `train.json`。Prompt 這樣設計：「這兩個對話中，人類最終選擇了 A。請以客觀的角度，簡短分析為什麼 A 比 B 更好。」
    * **步驟二（過濾）：** 用簡單的腳本過濾掉生成失敗或格式錯亂的理由。
    * **步驟三（訓練）：** 將這些成功的理由加入微調資料中。讓你的模型在訓練時，學習先輸出「分析理由」，最後才輸出 `"A"`、`"B"`、`"tie"` 或 `"neither"`。

### 3. 提示詞模板多樣化 (Prompt Template Augmentation)
不要讓模型對單一的排版格式產生過擬合 (Overfitting)。

* **實作邏輯：** 在 `dataset.py` 中準備 3 到 4 種不同的 Prompt 組裝格式，訓練時隨機套用。
    * *格式一：* `[對話一] \n ... \n [對話二] \n ... \n 請評判：`
    * *格式二：* `Assistant A: ... \n Assistant B: ... \n 哪個比較好？`
    * *格式三：* 將系統提示詞 (System Prompt) 稍微改寫，例如有時候叫它「客觀的裁判」，有時候叫它「嚴格的語言學家」。
* **效益：** 提升模型的泛化能力 (Generalization)，當測試集 `test.json` 中的對話長度或結構有微妙變化時，模型依然能精準抓到評判重點。
