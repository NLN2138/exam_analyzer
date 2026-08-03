```markdown
# 📚 Taiwan K-12 Exam Text Difficulty Analyzer

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.7.0-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)](https://spacy.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

> **An Automated Text Difficulty & Syntactic Complexity Diagnostic System for K-12 Exam Questions based on Natural Language Processing (NLP) and Mean Dependency Distance (MDD).**

Designed specifically for Taiwanese elementary, junior high, and senior high school exam materials, this system leverages **spaCy Dependency Parsing**, **Mean Dependency Distance (MDD)**, and a hybrid rule-based/machine learning scoring engine to automatically evaluate readability levels, part-of-speech distributions, and complex sentence structures.

---

## ✨ Key Features

- 🧠 **Dynamic MDD Compensation**: Calculates syntactic span distances while applying weighted adjustments for sentence length, noun density, and punctuation to accurately reflect cognitive processing load.
- 🧹 **Smart Exam Noise Filtering**: Automatically strips out instructional text (e.g., *"Choose the best answer"*) and structural markers (e.g., *(A)(B)(C)(D)*) to prevent artificial skewing of difficulty scores.
- 🎯 **Top 50% Discrimination Weighting**: Evaluates entire exam papers based on the top 50% most syntactically demanding sentences, preventing high-grade questions from being diluted by simple ones.
- 📊 **Multi-Dimensional Interactive Dashboards**:
  - **Single Sentence Mode**: Features a grade-level gauge chart and radar graph for multi-feature intensity.
  - **Batch & Paper Mode**: Renders grade-distribution pie charts, clause-type breakdowns, and MDD histogram bars.
- 📖 **Subject-Specific Terminology Detection**: Built-in specialized glossaries for **Chinese, Mathematics, Social Studies, and Natural Sciences** to gauge domain-specific vocabulary depth.

---

## 🛠️ Tech Stack

- **UI Framework**: Streamlit >= 1.37.0
- **NLP Core**: spaCy `3.7.0` (`zh_core_web_sm` Chinese dependency parsing model)
- **Data Processing**: Pandas, NumPy
- **ML Integration**: Joblib, Scikit-learn
- **Data Visualization**: Plotly (Express & Graph Objects)

---

## 🚀 Local Quick Start

### 1. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME

```

### 2. Set Up Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

The `requirements.txt` is pre-configured with the direct link to the `zh_core_web_sm` model wheel:

```bash
pip install -r requirements.txt

```

> **Note**: If the spaCy model fails to download automatically during pip install, run:
> `python -m spacy download zh_core_web_sm`

### 4. Run the Streamlit Application

```bash
streamlit run app.py

```

The application will open in your browser at `http://localhost:8501`.

---

## ☁️ Streamlit Community Cloud Deployment

This project is fully optimized for one-click deployment to **Streamlit Community Cloud**:

1. Push your code to your GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"**, select your Repository, Branch, and specify `app.py`.
4. Since `requirements.txt` links directly to the `zh_core_web_sm` wheel release on GitHub, Streamlit Cloud will automatically build the environment without extra setup!

---

## 📂 Project Structure

```text
├── app.py                     # Main Streamlit application
├── mdd_baseline_model.pkl     # Pre-trained ML baseline model (optional fallback to rule-engine)
├── requirements.txt           # Python dependency specifications
└── README.md                  # Project documentation

```

---

## 🔬 Feature Metrics Overview

| Metric Name | Description | Impact on Difficulty |
| --- | --- | --- |
| **MDD (Mean Dependency Distance)** | Average structural span distance between words and their syntactic heads. | Higher values indicate greater syntactic complexity and higher working memory load. |
| **Noun Ratio** | Proportion of nouns and proper nouns relative to total words. | High noun density correlates with abstract, concept-heavy academic content. |
| **Clause Types** | Automatic detection of complex conjunction patterns (causal, adversative, conditional, etc.). | Presence of advanced clause structures significantly increases logical reasoning difficulty. |
| **Vocab Depth** | Total count of recognized domain-specific academic terms. | Evaluates the depth of subject matter knowledge required. |

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

```

```
