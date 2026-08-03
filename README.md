```markdown
# 📚 台灣中小學試題語句難度檢測系統 (Taiwan K-12 Exam Text Difficulty Analyzer)

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.7.0-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)](https://spacy.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

> **基於自然語言處理 (NLP) 與平均依賴距離 (MDD) 之 K-12 試題語句難度與句法複雜度自動化診斷系統**

本系統專為台灣中小學（國小、國中、高中）各學科試題與英文教材設計，運用 **spaCy 依賴句法解析 (Dependency Parsing)**、**平均依賴距離 (Mean Dependency Distance, MDD)** 以及 **機器學習/規則融合演算**，自動診斷試題文字的難易度落點、詞性分布與複句結構。

---

## ✨ 系統核心亮點 (Key Features)

- 🧠 **MDD 依存距離動態修正**：計算句法結構中的平均修飾距離，並針對長句、名詞密度與標點進行加權補償，精準反映學習者的認知負擔。
- 🧹 **試卷智慧降噪 (Smart Noise Reduction)**：全卷檢測時自動過濾「請回答下列問題」、「(A)(B)(C)(D)」等配分與指示雜訊，避免無意義短句拉低評估結果。
- 🎯 **Top 50% 鑑別度加權評估**：整份考題採前 50% 最具鑑別度的核心長句/高難度語句進行整體年級評估，防止全卷難度被簡單題目稀釋。
- 📊 **多維度視覺化儀表板**：
  - **單句分析**：儀表板羅盤（年級落點）與特徵強度雷達圖。
  - **全卷/批次分析**：採樣句年級分布圓餅圖、複句句式占比圖、MDD 區間柱狀圖。
- 📖 **學科專業術語診斷**：內建 **國語文、數學、社會、自然** 四大學科核心術語庫，精準捕捉學科專有名詞帶來的閱讀門檻。

---

## 🛠️ 技術架構 (Tech Stack)

* **UI 框架**：Streamlit >= 1.37.0
* **NLP 核心**：spaCy `3.7.0` (`zh_core_web_sm` 中文依賴句法分析模型)
* **數據處理與統計**：Pandas, NumPy
* **機器學習模型載入**：Joblib, Scikit-learn
* **繪圖與視覺化**：Plotly (Express & Graph Objects)

---

## 🚀 本地端快速安裝與執行 (Quick Start)

### 1. 複製專案庫 (Clone Repository)

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME

```

### 2. 建立並啟用虛擬環境 (Virtual Environment)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. 安裝依賴套件 (Install Dependencies)

專案已配置好包含 `zh_core_web_sm` 模型輪子檔的 `requirements.txt`：

```bash
pip install -r requirements.txt

```

> **備註**：若安裝過程未能自動下載 spaCy 模型，請手動執行：
> `python -m spacy download zh_core_web_sm`

### 4. 啟動 Streamlit 應用程式

```bash
streamlit run app.py

```

啟動後，瀏覽器將自動開啟 `http://localhost:8501`。

---

## ☁️ 部署至 Streamlit Community Cloud

本專案支援一鍵部署至 **Streamlit Community Cloud**：

1. 將專案推送 (Push) 至 GitHub。
2. 登入 [Streamlit Community Cloud](https://share.streamlit.io/)。
3. 點擊 **"New app"**，選擇你的 Repository、Branch 與 `app.py` 檔案。
4. 由於 `requirements.txt` 中已直接引入 `zh_core_web_sm` 的 GitHub Release `.whl` 連結，Streamlit 雲端平台將會自動安裝 NLP 模型並完成部署！

---

## 📂 專案目錄結構 (Project Structure)

```text
├── app.py                     # Streamlit 主程式碼
├── mdd_baseline_model.pkl     # 機器學習預訓練模型 (選用，若無則啟動動態積分引擎)
├── requirements.txt           # Python 套件依賴清單
└── README.md                  # 專案說明文件

```

---

## 🔬 特徵指標說明 (Feature Metrics)

| 指標名稱 | 說明 | 影響 |
| --- | --- | --- |
| **MDD (平均依賴距離)** | 句中各單詞與其 Head (主控詞) 之間的平均跨度距離。 | 數值越高，代表句法結構越複雜，短期記憶負擔越大。 |
| **名詞密度 (Noun Ratio)** | 名詞與專有名詞佔總詞數的比例。 | 高名詞密度常見於學科概念導向的試題（概念密度高）。 |
| **複句結構 (Clause Types)** | 自動辨識「因果、轉折、遞進、條件」等複句連接詞與關聯詞。 | 出現進階複句會顯著增加邏輯推理難度。 |
| **學科術語數 (Vocab Depth)** | 命中的學科關鍵詞數量。 | 評估該題目是否包含深度學科知識。 |

---

## 📄 授權條款 (License)

Distributed under the MIT License. See `LICENSE` for more information.

```

```
