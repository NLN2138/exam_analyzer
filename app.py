import streamlit as st
import spacy
import re
import pandas as pd
import joblib
import os

# ==========================================
# 1. 頁面配置與模型快取 (雲端優化)
# ==========================================
st.set_page_config(
    page_title="國小測驗卷語句難度與語意分析系統",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def load_nlp_model():
    """直接載入已安裝的 spaCy 中文模型"""
    try:
        return spacy.load("zh_core_web_sm")
    except Exception as e:
        return None

@st.cache_resource
def load_ml_model():
    """載入預測模型 (.pkl)，若不存在則回傳 None"""
    if os.path.exists('mdd_baseline_model.pkl'):
        try:
            return joblib.load('mdd_baseline_model.pkl')
        except Exception:
            return None
    return None

nlp = load_nlp_model()
ml_model = load_ml_model()

# ==========================================
# 2. 核心 NLP 引擎與特徵萃取
# ==========================================
class QuestionPreprocessor:
    def __init__(self):
        # 國小測驗卷常見的「東方語意標記」關聯詞正則字典
        self.semantic_markers = {
            "因果": r"(因為|所以|由於|因此|導致|以致於)",
            "假設": r"(如果|假使|要是|若|則|假如|一旦)",
            "轉折": r"(但是|可是|卻|雖然|然而|不過|只是)",
            "條件": r"(只要|只有|除非|無論|不管|任憑)",
            "並列": r"(一邊|同時|以及|和|跟|與|既)",
            "目的": r"(為了|以便|以免|用以)",
            "遞進": r"(不但|而且|甚至|更|何況)",
            "選擇": r"(或者|還是|與其|不如|寧可)"
        }
    
    def _calculate_mdd(self, doc):
        """計算單一句子的平均依存距離 (MDD)"""
        if not doc: 
            return 0.0
        
        total_distance = 0
        valid_tokens = 0
        
        for token in doc:
            # 排除標點符號與根節點 (ROOT)
            if token.dep_ != "ROOT" and not token.is_punct and token.pos_ != "PUNCT":
                distance = abs(token.i - token.head.i)
                total_distance += distance
                valid_tokens += 1
                
        return total_distance / valid_tokens if valid_tokens > 0 else 0.0

    def _classify_semantic_marker(self, text):
        """依據正則表達式判定複句語意邏輯"""
        for marker, pattern in self.semantic_markers.items():
            if re.search(pattern, text):
                return marker
        return "連貫／承接" # 預設無明顯標記的承接句

    def analyze(self, text: str, subject: str) -> dict:
        """執行完整的文字分析，回傳特徵字典"""
        word_count = len(text.strip())
        doc = nlp(text) if nlp else None
        mdd_value = self._calculate_mdd(doc)
        semantic_type = self._classify_semantic_marker(text)
        
        return {
            "MDD(平均依存距離)": round(mdd_value, 3),
            "字數": word_count,
            "學科": subject,
            "東方語意標記": semantic_type
        }

# ==========================================
# 3. 預測邏輯 (整合機器學習與統計基線降級)
# ==========================================
def predict_grade(analysis_result: dict) -> str:
    """根據分析特徵預測適配年級"""
    if ml_model is not None:
        features_df = pd.DataFrame([analysis_result])
        try:
            prediction = ml_model.predict(features_df)[0]
            return str(prediction)
        except Exception:
            pass 
    
    # 統計基準線備用邏輯
    mdd = analysis_result["MDD(平均依存距離)"]
    subj = analysis_result["學科"]
    if mdd > 3.6 and subj in ["自然", "社會"]:
        return "5年級或6年級 (統計基準推估)"
    elif mdd > 3.5:
        return "4年級或5年級 (統計基準推估)"
    else:
        return "3年級 (統計基準推估)"

# ==========================================
# 4. Streamlit 網頁介面設計
# ==========================================
st.title("📚 國小測驗卷語句難度與語意分析系統")
st.markdown("本系統基於量化語料庫與依存句法分析（MDD），評估國小各學科試題的句法負擔與適配年級。")

# 系統健康狀態提示
if nlp is None:
    st.error("🚨 嚴重錯誤：無法載入 spaCy 中文模型 (`zh_core_web_sm`)。請檢查 requirements.txt 是否正確安裝。")
    st.stop()

if ml_model is None:
    st.info("💡 提示：目前未檢測到 `.pkl` 模型檔案，系統自動採用基於統計特徵的基準線引擎進行預測。")

engine = QuestionPreprocessor()

# 頁面左右分欄
col1, col2 = st.columns([1, 1.8], gap="large")

with col1:
    st.subheader("📝 試題輸入區")
    subject = st.selectbox(
        "選擇所屬學科：", 
        ["自然", "社會", "國語"],
        help="不同學科的語句負荷基準不同，社會與自然科通常具有更高的句法複雜度。"
    )
    
    default_text = "如果我們不控制碳排放，全球暖化將導致極端氣候常態化。"
    question_text = st.text_area(
        "請輸入或貼上試題文本：", 
        height=180, 
        value=default_text,
        help="請輸入單一試題或複句文本。"
    )
    
    analyze_btn = st.button("🚀 開始深度分析", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 分析結果與儀表板")
    
    if analyze_btn:
        if not question_text.strip():
            st.warning("⚠️ 請先輸入有效的試題文本！")
        else:
            with st.spinner("正在進行依存句法剖析與語意特徵萃取..."):
                result = engine.analyze(question_text, subject)
                predicted_grade = predict_grade(result)
            
            # 顯示預測年級大卡片
            st.success(f"### 🎯 預測適配年級：**{predicted_grade}**")
            
            # 三項核心指標數據卡片
            m1, m2, m3 = st.columns(3)
            m1.metric(
                label="MDD (平均依存距離)", 
                value=result["MDD(平均依存距離)"],
                help="句子中詞語與支配詞距離的平均值，數值越高代表句法結構越長、樹狀越深、閱讀負荷越大。"
            )
            m2.metric(
                label="文本總字數", 
                value=f"{result['字數']} 字"
            )
            m3.metric(
                label="東方語意標記", 
                value=result["東方語意標記"],
                help="依據關聯詞自動判定的複句邏輯類型（如假設、因果、轉折等）。"
            )
            
            st.divider()
            
            # 專家解讀與建議
            st.markdown("#### 💡 教學與命題檢核建議")
            mdd_val = result["MDD(平均依存距離)"]
            if mdd_val > 3.6:
                st.warning(
                    f"此題的 MDD 為 **{mdd_val}**，句法結構較為複雜。若作為中年級（3-4年級）測驗，"
                    "學生可能需要花費額外認知資源在「解析句型」而非「理解學科知識」上，建議適度拆解子句。"
                )
            else:
                st.info(
                    f"此題的 MDD 為 **{mdd_val}**，句法結構適中，符合一般學童的認知發展負荷，具有良好的測驗親和性。"
                )
    else:
        st.info("👈 請在左側輸入試題並點擊「開始深度分析」以檢視結構指標。")
