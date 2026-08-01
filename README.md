# 📚 Taiwan K-6 Exam Question Sentence Difficulty Assessment System (Prototype)

An NLP and Machine Learning powered web application built with **Streamlit** to analyze and evaluate the difficulty level of Taiwan K-6 (elementary school) exam question sentences. 

The system leverages **spaCy** for Natural Language Processing, calculates **Mean Dependency Distance (MDD)** for syntactic complexity, detects 8 major Chinese compound sentence structures, and maps domain-specific vocabulary tailored to Taiwan's 108 Curriculum Guidelines (國語文, 數學, 社會, 自然).

---

## ✨ Features

- **✍️ Single Sentence Analysis**:
  - Predicts target grade range (Grades 1-2, 3-4, 5-6+).
  - Calculates length (character/word counts) and Part-of-Speech (POS) distribution ratios (nouns vs. verbs).
  - Visualizes Mean Dependency Distance (MDD) via a gauge meter.
  - Detects complex clause types and counts subject-specific academic vocabulary.

- **📋 Batch Processing & Analytics Dashboard**:
  - Supports **Multi-line text input** or **File Upload** (`.csv`, `.xlsx`).
  - Auto-detects text columns (`題目`, `question`, `text`, `試題`, `內容`).
  - Generates summary KPI metrics (Overall grade estimation, total word count, average MDD).
  - Provides interactive **Plotly charts**:
    - Estimated Grade Level Distribution (Pie Chart)
    - Clause Structure Types Breakdown (Pie Chart)
    - MDD Syntactic Complexity Distribution (Bar Chart)
  - One-click CSV report export encoded with `utf-8-sig` for seamless Excel compatibility.

- **🎯 Subject Domain Adaptability**:
  - Custom domain vocabulary mappings covering over 200+ core terms per subject aligned with Taiwan's 108 Curriculum Guidelines:
    - **Chinese Language (國語文)**
    - **Mathematics (數學)**
    - **Social Studies (社會)**
    - **Natural Sciences (自然)**

- **🤖 Hybrid Scoring Engine**:
  - Automatically utilizes `mdd_baseline_model.pkl` (scikit-learn model) if present.
  - Fallback dynamic heuristic scoring rule engine if no trained pickle model file is provided.

---

## 🛠️ Project Structure

```text
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python dependencies and spaCy model URL
├── mdd_baseline_model.pkl      # Optional pretrained ML model file
└── README.md                   # Project documentation

---

## 🚀 Getting Started

### 1. Prerequisites

Make sure you have **Python 3.10+** installed on your system.

### 2. Installation

1. **Clone the repository**:
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

```


2. **Create and activate a virtual environment (Recommended)**:
* **macOS / Linux**:
```bash
python3 -m venv venv
source venv/bin/activate

```


* **Windows**:
```bash
python -m venv venv
venv\Scripts\activate

```




3. **Install Dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt

```


> **Note:** `requirements.txt` automatically downloads and installs the official spaCy Chinese model wheel (`zh_core_web_sm`).



---

## 💻 Running the Application

Launch the Streamlit app locally with:

```bash
streamlit run app.py

```

Open your browser at `http://localhost:8501` to view the application.

---

## ⚙️ Configuration & Model Setup

* **Pretrained ML Model**: If you have a trained model, place `mdd_baseline_model.pkl` in the root directory. The application will automatically detect and enable it.
* **Dynamic Rule Engine**: If `mdd_baseline_model.pkl` is absent, the system gracefully falls back to the dynamic weighted scoring engine based on MDD, clause structures, and vocabulary depth.

---

## 🧰 Tech Stack

* **UI Framework**: Streamlit
* **NLP Pipeline**: spaCy Chinese (`zh_core_web_sm`)
* **Data Manipulation**: pandas, openpyxl
* **Machine Learning**: scikit-learn, joblib
* **Data Visualization**: Plotly Express & Graph Objects

---

## ⚖️ License & Acknowledgments

This project is licensed under the [MIT License](LICENSE).

### Third-Party Libraries & Assets
* **Streamlit**: Copyright © Snowflake Inc. (Licensed under Apache License 2.0).
* **spaCy & `zh_core_web_sm`**: Copyright © Explosion AI GmbH (Licensed under MIT / CC BY-SA 4.0).
* **Plotly**: Copyright © Plotly Technologies Inc. (Licensed under MIT License).
* **scikit-learn & joblib**: Copyright © 2007-2026 scikit-learn developers (Licensed under BSD 3-Clause).

### Theoretical Frameworks
* **Mean Dependency Distance (MDD)**: Liu, H. (2008). Dependency distance as a metric of language comprehension difficulty. *Journal of Cognitive Science*, *9*(2), 159–191.

```

```
