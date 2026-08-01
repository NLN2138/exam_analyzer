import os
import joblib
import pandas as pd
import spacy
import streamlit as st
import plotly.express as px
from typing import List, Dict, Any, Optional

# ==========================================
# 0. 靜態常數定義 (容易維護與擴充)
# ==========================================
ADVANCED_KEYWORDS = {
    "由於", "導致", "以致於", "即使", "仍", "除非", "無論", "若", 
    "除了...也", "透過", "以維持", "評估", "脈絡", "偏誤", "然而", 
    "此外", "因此", "鑑於", "唯有", "與其", "不如"
}

CONNECTORS = {
    "因果複句": ["因為", "所以", "由於", "因此", "以致於", "導致"],
    "轉折複句": ["雖然", "但是", "不過", "卻", "可是", "然而"],
    "假設複句": ["如果", "要是", "假如", "的話", "若", "即使"],
    "條件複句": ["只要", "只有", "當...時", "除了", "除非", "無論", "唯有"],
    "並列複句": ["同時", "一方面", "以及", "並且", "也", "又", "既...又..."],
    "遞進複句": ["不但", "而且", "甚至", "更", "不僅"],
    "承接複句": ["先", "然後", "接著", "於是", "才"],
    "目的複句": ["為了", "以免", "以便", "以維持"],
    "選擇複句": ["不是...就是", "與其...不如", "寧可...也不"]
}

ADVANCED_TERMS = {
    "演算法", "同溫層", "合力", "敘事觀點", "社會文化脈絡", "供給", "需求",
    "蒸發", "凝結", "光合作用", "分裂", "細胞", "生物體", "公義", "政治結構",
    "經濟條件", "環境影響", "社會公平", "單一因果", "偏誤", "多元觀點", "效率",
    "憲政體制", "權力分立", "變因", "共識", "職權", "濫用", "權益"
}

# ==========================================
# 1. 頁面基本設定
# ==========================================
st.set_page_config(
    page_title="台灣中小學試題句子難度檢測系統",
    page_icon="📚",
    layout="wide"
)

# ==========================================
# 2. 核心資源快取載入區
# ==========================================
@st.cache_resource(show_spinner="載入 NLP 模型中...")
def load_nlp():
    try:
        return spacy.load("zh_core_web_sm")
    except OSError:
        st.error("❌ 找不到 spaCy 模型！請確保 requirements.txt 包含正確的 whl 網址。")
        st.stop()

@st.cache_resource(show_spinner="載入評分模型中...")
def load_difficulty_model():
    model_path = "mdd_baseline_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# ==========================================
# 3. 難度特徵運算邏輯
# ==========================================
def analyze_clause_types(doc: spacy.tokens.Doc) -> str:
    text = doc.text
    detected_types = []
    
    if any(kw in text for kw in ADVANCED_KEYWORDS):
        detected_types.append("進階論述句")
        
    for clause_type, keywords in CONNECTORS.items():
        if any(kw in text for kw in keywords):
            detected_types.append(clause_type)
            
    if not detected_types:
        dep_labels = {token.dep_ for token in doc}
        if "advcl" in dep_labels or "conj" in dep_labels:
            detected_types.append("複雜修飾句")
        else:
            detected_types.append("簡單句")
            
    return ", ".join(detected_types)

def calculate_vocab_depth(doc: spacy.tokens.Doc) -> int:
    return sum(1 for token in doc if token.text in ADVANCED_TERMS)

def extract_features_from_doc(doc: spacy.tokens.Doc) -> Dict[str, Any]:
    word_count = len(doc)
    char_count = len(doc.text)
    
    nouns_count = sum(1 for token in doc if token.pos_ in ("NOUN", "PROPN"))
    verbs_count = sum(1 for token in doc if token.pos_ == "VERB")
    
    noun_ratio = nouns_count / word_count if word_count > 0 else 0.0
    verb_ratio = verbs_count / word_count if word_count > 0 else 0.0
    
    dep_distances = [abs(token.i - token.head.i) for token in doc if token.head != token]
    mdd = sum(dep_distances) / len(dep_distances) if dep_distances else 0.0
    
    return {
        "text": doc.text,
        "char_count": char_count,
        "word_count": word_count,
        "noun_ratio": noun_ratio,
        "verb_ratio": verb_ratio,
        "mdd": mdd,
        "clause_types": analyze_clause_types(doc),
        "vocab_depth": calculate_vocab_depth(doc)
    }

def predict_grade(features: Dict[str, Any], ml_model: Optional[Any]) -> str:
    score = 3.0 
    if ml_model is not None:
        try:
            df_features = pd.DataFrame([{
                "char_count": features["char_count"],
                "word_count": features["word_count"],
                "noun_ratio": features["noun_ratio"],
                "verb_ratio": features["verb_ratio"],
                "mdd": features["mdd"]
            }])
            raw_pred = ml_model.predict(df_features)[0]
            if isinstance(raw_pred, (int, float)):
                score = float(raw_pred)
        except Exception:
            pass

    if features["char_count"] <= 20: score -= 1.0
    elif features["char_count"] >= 35: score += 1.0
    elif features["char_count"] >= 45: score += 1.5
    
    if features["mdd"] < 2.6: score -= 0.5
    elif 3.2 <= features["mdd"] < 4.0: score += 1.0
    elif features["mdd"] >= 4.0: score += 2.0
    
    if features["noun_ratio"] < 0.20: score -= 0.5
    elif features["noun_ratio"] > 0.30: score += 0.5
    
    complex_clauses = ["進階論述句", "目的複句", "選擇複句", "遞進複句"]
    if any(c in features["clause_types"] for c in complex_clauses):
        score += 1.0
        
    if features["vocab_depth"] >= 1:
        score += 1.5

    if score >= 4.5: return "5-6 年級 (高年級) 或以上"
    elif score <= 2.5: return "1-2 年級 (低年級)"
    else: return "3-4 年級 (中年級)"

def run_batch_analysis(question_list: List[str], nlp_model, difficulty_model) -> pd.DataFrame:
    results = []
    progress_bar = st.progress(0)
    total = len(question_list)
    
    for i, doc in enumerate(nlp_model.pipe(question_list, batch_size=50)):
        feat = extract_features_from_doc(doc)
        grade = predict_grade(feat, difficulty_model)
        
        results.append({
            "題目內容": feat["text"],
            "預估適用年級": grade,
            "複句結構與句式": feat["clause_types"],
            "總字數": feat["char_count"],
            "名詞密度": f"{feat['noun_ratio']:.1%}",
            "MDD數值": round(feat["mdd"], 2)
        })
        progress_bar.progress((i + 1) / total)
        
    progress_bar.empty()
    return pd.DataFrame(results)

# ==========================================
# 3.5 視覺化統計圖表繪製邏輯
# ==========================================
def render_statistics_charts(df: pd.DataFrame):
    st.markdown("### 📊 批次分析統計儀表板")
    col1, col2, col3 = st.columns(3)
    
    # 1. 預估年級占比 (圓餅圖)
    grade_counts = df["預估適用年級"].value_counts().reset_index()
    grade_counts.columns = ["年級", "題數"]
    fig_grade = px.pie(grade_counts, names="年級", values="題數", hole=0.4, 
                       title="預估適用年級占比", 
                       color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_grade.update_traces(textposition='inside', textinfo='percent+label')
    fig_grade.update_layout(showlegend=False)
    col1.plotly_chart(fig_grade, use_container_width=True)
    
    # 2. 句式占比 (圓餅圖) - 自動拆解包含多重句式的欄位
    clause_series = df["複句結構與句式"].str.split(", ").explode()
    clause_counts = clause_series.value_counts().reset_index()
    clause_counts.columns = ["句式", "出現次數"]
    fig_clause = px.pie(clause_counts, names="句式", values="出現次數", hole=0.4, 
                        title="複句句式出現占比",
                        color_discrete_sequence=px.colors.qualitative.Set3)
    fig_clause.update_traces(textposition='inside', textinfo='percent+label')
    fig_clause.update_layout(showlegend=False)
    col2.plotly_chart(fig_clause, use_container_width=True)
    
    # 3. MDD 區間分布 (長條圖)
    bins = [0, 1, 2, 3, 4, 5, 100]
    labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5以上']
    df_mdd = df.copy()
    # 將 MDD 切分為指定區間 (right=False 代表包含左邊界但不含右邊界，例如 [1, 2) )
    df_mdd['MDD區間'] = pd.cut(df_mdd['MDD數值'], bins=bins, labels=labels, right=False)
    mdd_counts = df_mdd['MDD區間'].value_counts().sort_index().reset_index()
    mdd_counts.columns = ["MDD區間", "題數"]
    
    fig_mdd = px.bar(mdd_counts, x="MDD區間", y="題數", 
                     title="MDD 依存距離區間分布", 
                     text_auto=True, 
                     color="MDD區間",
                     color_discrete_sequence=px.colors.sequential.Blues_r)
    fig_mdd.update_layout(showlegend=False, xaxis_title="MDD 數值區間", yaxis_title="題目數量")
    col3.plotly_chart(fig_mdd, use_container_width=True)


# ==========================================
# 4. 前端介面與互動
# ==========================================
with st.sidebar:
    st.header("⚙️ 系統狀態")
    nlp = load_nlp()
    st.success("✅ spaCy 中文模型已載入")
    
    model = load_difficulty_model()
    if model:
        st.success("✅ ML 基準模型已啟用")
    else:
        st.warning("⚠️ 啟用動態積分評分引擎 (未載入 pkl 模型)")
        
    st.divider()
    subject = st.selectbox("學科", ["國語文", "數學", "社會", "自然"])
    show_table = st.checkbox("顯示單題特徵明細表", value=True)

st.title("📚 台灣中小學試題句子難度檢測系統")
st.caption("支援單題檢測、句式特徵解析，以及多題文字貼上／檔案上傳的批次檢測。")

tab1, tab2 = st.tabs(["✍️ 單題檢測與複句分析", "📋 批次多題檢測與統計儀表板"])

# --- TAB 1: 單題檢測 ---
with tab1:
    question_text = st.text_area("題目文字", height=130, placeholder="請輸入試題...")

    if st.button("🚀 開始檢測單題", type="primary"):
        if not question_text.strip():
            st.warning("請先輸入題目文字！")
        else:
            with st.spinner("分析中..."):
                doc = nlp(question_text)
                features = extract_features_from_doc(doc)
                predicted_grade = predict_grade(features, model)
                
                st.divider()
                cols = st.columns(4)
                cols[0].metric("🎯 預估年級", str(predicted_grade))
                cols[1].metric("📏 總字數", f"{features['char_count']} 字")
                cols[2].metric("🧠 依存距離 (MDD)", f"{features['mdd']:.2f}")
                cols[3].metric("🔗 複句結構", features["clause_types"])
                    
                if show_table:
                    st.subheader("📋 試題特徵明細")
                    st.dataframe({
                        "特徵名稱": ["總詞數", "名詞比例", "動詞比例", "進階術語計數"],
                        "數值": [
                            features['word_count'], 
                            f"{features['noun_ratio']:.1%}", 
                            f"{features['verb_ratio']:.1%}", 
                            f"{features['vocab_depth']} 個"
                        ]
                    }, use_container_width=True)

# --- TAB 2: 批次查詢與統計儀表板 ---
with tab2:
    batch_mode = st.radio("輸入方式：", ["📋 貼上多行文字", "📂 上傳檔案"], horizontal=True)
    
    if batch_mode == "📋 貼上多行文字":
        batch_text = st.text_area("每行一題：", height=250)
        if st.button("⚡ 開始批次分析與產生圖表", type="primary"):
            q_list = [line.strip() for line in batch_text.split("\n") if line.strip()]
            if q_list:
                res_df = run_batch_analysis(q_list, nlp, model)
                # 繪製圖表
                st.divider()
                render_statistics_charts(res_df)
                
                # 顯示表格與下載
                st.markdown("### 📝 詳細題目檢測報表")
                st.dataframe(res_df, use_container_width=True)
                st.download_button("📥 下載 CSV 報告", res_df.to_csv(index=False).encode("utf-8-sig"), "文字批次報告.csv", "text/csv")
            else:
                st.warning("請貼上有效的題目內容！")
                
    else:
        uploaded_file = st.file_uploader("請選擇 CSV 或 Excel 檔案", type=["csv", "xlsx"])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                text_col = next((c for c in df.columns if str(c).strip().lower() in ["題目", "question", "text", "試題", "內容"]), None)
                
                if not text_col:
                    st.error(f"❌ 找不到題目欄位！現有欄位：{', '.join(df.columns)}")
                else:
                    q_list = df[text_col].dropna().astype(str).str.strip()
                    q_list = q_list[q_list != ""].tolist()
                    
                    st.info(f"成功擷取 {len(q_list)} 題，準備分析。")
                    if st.button("⚡ 開始批次分析與產生圖表", type="primary"):
                        res_df = run_batch_analysis(q_list, nlp, model)
                        
                        # 繪製圖表
                        st.divider()
                        render_statistics_charts(res_df)
                        
                        # 顯示表格與下載
                        st.markdown("### 📝 詳細題目檢測報表")
                        st.dataframe(res_df, use_container_width=True)
                        st.download_button("📥 下載結果", res_df.to_csv(index=False).encode("utf-8-sig"), "檔案分析報告.csv", "text/csv")
            except Exception as e:
                st.error(f"讀取失敗：{e}")
