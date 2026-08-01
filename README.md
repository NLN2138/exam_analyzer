# 📚 台灣中小學試題句子難度檢測系統

本專案是一個基於 **Streamlit** 與 **spaCy (自然語言處理)** 開發的台灣中小學試題難度自動檢測工具。透過分析文字的句法結構、詞彙深度以及平均依存距離 (MDD)，輔助教育工作者量化並評估試題的閱讀難度。

## ✨ 系統功能亮點

1. **🚀 效能優化的單題/批次檢測**
   - 運用 `spacy.pipe` 進行非同步批次解析，可快速處理數千題的 CSV/Excel 題庫。
   - 自動運算試題總字數、詞數、名詞/動詞出現比例、MDD，並產出分析報告。

2. **🧠 台灣教育語境深度分析**
   - 具備特製的台灣學科進階詞彙庫（如：同溫層、光合作用、社會公平）。
   - 自動識別複雜句式結構（因果、轉折、假設、條件與並列複句）。

3. **📊 混合式難度分級 (Hybrid Grading)**
   - 支援載入機器學習 `.pkl` 模型作為基準預測。
   - 若無模型，系統自動無縫切換為「動態權重規則引擎」，準估適用年級（低/中/高）。

---

## 💻 如何在本地端執行？

1. Clone 此專案至本地端：
   ```bash
   git clone [https://github.com/您的帳號/Taiwan-Exam-Difficulty-Analyzer.git](https://github.com/您的帳號/Taiwan-Exam-Difficulty-Analyzer.git)
   cd Taiwan-Exam-Difficulty-Analyzer
