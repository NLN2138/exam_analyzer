# 📚 Taiwan K-12 Exam Text Difficulty Analyzer

> **An automated NLP system for assessing readability and syntactic complexity (MDD) in K-12 exam questions.**

[![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.7.0-09A3D5?logo=spacy&logoColor=white)](https://spacy.io/)

---

## ✨ Features

- 🧠 **MDD Analysis**: Measures Mean Dependency Distance with dynamic adjustments for long clauses, noun density, and punctuation.
- 🧹 **Smart Noise Filtering**: Strips out instructions, question numbers, and answer choices to prevent artificial score dilution.
- 🎯 **Top 50% Discrimination Weighting**: Evaluates full exams based on the most demanding sentences.
- 📊 **Interactive Charts**: Single-sentence gauge/radar charts and full-paper statistical breakdowns (Plotly).
- 📖 **Subject Glossaries**: Built-in terminology tracking for Chinese, Math, Social Studies, and Natural Sciences.

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
cd YOUR_REPOSITORY

python -m venv venv
# Windows: venv\Scripts\activate | macOS/Linux: source venv/bin/activate
