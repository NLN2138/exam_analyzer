import os
import re
import joblib
import pandas as pd
import spacy
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple

# ==========================================
# 0. 靜態常數定義 (容易維護與擴充)
# ==========================================

# ✨ 擴充至 60+ 個進階論述與學術常規詞彙 (涵蓋邏輯、推論、學術動詞)
ADVANCED_KEYWORDS = {
    "由於", "導致", "以致於", "即使", "仍", "除非", "無論", "若", 
    "除了...也", "透過", "以維持", "評估", "脈絡", "偏誤", "然而", 
    "此外", "因此", "鑑於", "唯有", "與其", "不如", "據此", "綜上所述", 
    "探討", "釐清", "闡述", "剖析", "歸納", "演繹", "驗證", "假說", 
    "旨在", "抑或", "縱使", "迄今", "趨勢", "顯著", "核心", "範疇", 
    "涉及", "奠定", "藉由", "促使", "衍伸", "闡明", "釐定", "審視", 
    "統整", "檢視", "探究", "辨析", "詮釋", "實踐", "蘊含", "突顯", 
    "綜觀", "舉凡", "毋寧", "端賴", "悖論", "機制", "框架", "準則"
}

# ✨ 各學科專業詞彙擴充至約 200 詞 (對齊台灣 108 課綱各科核心知識點)
SUBJECT_TERMS = {
    "國語文": {
        # 修辭與文體
        "修辭", "譬喻", "借代", "轉化", "擬人", "擬物", "誇飾", "排比", "層遞", "設問", 
        "對偶", "頂真", "映襯", "雙關", "象徵", "呼告", "倒裝", "韻文", "詞牌", "新詩", 
        "意象", "寓言", "絕句", "律詩", "古體詩", "近體詩", "樂府", "賦", "散文", "小說", 
        "記敘文", "抒情文", "說明文", "議論文", "應用文", "書信", "便條", "對聯", "題辭",
        # 國學與語文常識
        "敘事觀點", "文眼", "主旨", "大意", "段落", "伏筆", "懸念", "烘托", "借景抒情", 
        "托物言志", "六書", "象形", "指事", "會意", "形聲", "轉注", "假借", "部首", "筆畫", 
        "字形", "字音", "字義", "詞性", "名詞", "動詞", "形容詞", "副詞", "代詞", "介詞", 
        "連詞", "助詞", "量詞", "歎詞", "句型", "直述句", "疑問句", "祈使句", "感嘆句",
        "平仄", "押韻", "對仗", "經史子集", "唐宋八大家", "詩仙", "詩聖", "詞眼", "曲",
        "偏旁", "部首", "繁體", "簡體", "文言文", "白話文", "語錄體", "紀傳體", "編年體",
        "國音", "聲母", "韻母", "結合韻", "聲調", "破音字", "同音字", "多音字", "形近字",
        "成語", "諺語", "歇後語", "慣用語", "外來語", "敬辭", "謙辭", "稱謂", "文法",
        # 閱讀理解與文學鑑賞
        "文本", "情節", "人物", "背景", "衝突", "高潮", "結局", "起承轉合", "第一人稱",
        "第三人稱", "倒敘", "順敘", "插敘", "補敘", "白描", "寫實", "浪漫", "魔幻", "史詩",
        "神話", "傳說", "民間故事", "童話", "科幻", "武俠", "推理", "寓意", "絃外之音",
        "言外之意", "主觀", "客觀", "批判", "賞析", "鑑賞", "共鳴", "流派", "文學史"
    },
    
    "數學": {
        # 數與量
        "整數", "分數", "小數", "質數", "合數", "因數", "倍數", "公因數", "公倍數", 
        "最大公因數", "最小公倍數", "絕對值", "有理數", "無理數", "實數", "正數", "負數",
        "倒數", "相反數", "科學記號", "四捨五入", "無條件進位", "無條件捨去", "概數",
        "比例", "正比", "反比", "百分率", "千分率", "折現率", "利率", "本金", "利息",
        # 代數
        "演算法", "方程式", "函數", "未知數", "變數", "常數", "係數", "多項式", "單項式",
        "同類項", "指數", "底數", "對數", "根號", "平方根", "立方根", "一次方程式", 
        "二次方程式", "聯立方程式", "不等式", "等差數列", "等比數列", "級數", "公差", "公比",
        "公式", "代入", "展開", "因式分解", "配方法", "十字交乘", "公式解", "判別式",
        # 幾何
        "幾何", "點", "線", "面", "角", "度", "射線", "線段", "平行", "垂直", "相交",
        "三角形", "直角三角形", "等腰三角形", "正三角形", "鈍角三角形", "銳角三角形",
        "四邊形", "正方形", "長方形", "平行四邊形", "梯形", "菱形", "箏形", "多邊形",
        "圓", "半徑", "直徑", "圓周", "圓周率", "弧", "弦", "扇形", "弓形", "圓心角", "圓周角",
        "全等", "相似", "比例尺", "對稱", "線對稱", "點對稱", "旋轉", "平移", "翻轉",
        "面積", "體積", "表面積", "周長", "柱體", "錐體", "球體", "長方體", "正方體",
        "座標", "平面座標", "直角座標", "象限", "原點", "Ｘ軸", "Ｙ軸", "斜率", "畢氏定理",
        # 統計與機率
        "統計", "資料", "圖表", "長條圖", "折線圖", "圓形圖", "直方圖", "次數分配表",
        "平均數", "中位數", "眾數", "全距", "四分位數", "盒狀圖", "標準差", "變異數",
        "機率", "事件", "樣本空間", "期望值", "排列", "組合", "樹狀圖", "相對次數"
    },
    
    "社會": {
        # 公民與社會、經濟
        "社會文化脈絡", "供給", "需求", "公義", "政治結構", "經濟條件", "環境影響", 
        "社會公平", "單一因果", "偏誤", "多元觀點", "效率", "憲政體制", "權力分立", 
        "職權", "濫用", "權益", "共識", "市場", "人權", "民主", "法治", "憲法", "法律", 
        "立法院", "行政院", "司法院", "考試院", "監察院", "選舉", "政黨", "利益團體", 
        "媒體", "第四權", "機會成本", "比較利益", "絕對利益", "外部性", "通貨膨脹", 
        "GDP", "國內生產毛額", "全球化", "社會流動", "弱勢族群", "社會規範", "倫理", 
        "道德", "性別平權", "公共利益", "公民參與", "消費者", "生產者", "利潤", "誘因",
        "市場機能", "看不見的手", "政府干預", "稅收", "社會福利", "少子化", "高齡化",
        "多元文化", "文化位階", "原住民", "新住民", "基本權利", "救濟", "民法", "刑法",
        "行政法", "無罪推定", "少年事件處理法", "契約", "侵權行為", "財產權", "智慧財產權",
        # 歷史
        "史前時代", "舊石器時代", "新石器時代", "金屬器時代", "原住民", "大航海時代",
        "荷西時期", "鄭氏時期", "清領時期", "日治時期", "戰後時期", "解嚴", "戒嚴", 
        "民主化", "白色恐怖", "二二八事件", "朝代", "皇帝", "封建", "帝國", "殖民", 
        "條約", "不平等條約", "革命", "啟蒙運動", "文藝復興", "工業革命", "冷戰", 
        "第一次世界大戰", "第二次世界大戰", "聯合國", "十字軍東征", "資本主義", "共產主義",
        "文明古國", "四大發明", "絲路", "甲午戰爭", "馬關條約", "辛亥革命", "五四運動",
        # 地理
        "經度", "緯度", "赤道", "本初子午線", "時區", "國際換日線", "比例尺", "圖例",
        "等高線", "地形", "高山", "丘陵", "台地", "平原", "盆地", "火山", "海岸",
        "氣候", "天氣", "季風", "洋流", "溫室效應", "氣壓", "降水", "氣溫", "水文",
        "人口金字塔", "人口密度", "都市化", "鄉村", "聚落", "產業", "第一級產業", 
        "第二級產業", "第三級產業", "高科技產業", "自然資源", "永續發展", "板塊", "地震帶"
    },
    
    "自然": {
        # 生物
        "細胞", "細胞膜", "細胞壁", "細胞質", "細胞核", "葉綠體", "粒線體", "液胞",
        "光合作用", "呼吸作用", "酵素", "擴散作用", "滲透作用", "生物體", "組織", "器官",
        "系統", "消化系統", "循環系統", "呼吸系統", "排泄系統", "神經系統", "內分泌系統",
        "激素", "動脈", "靜脈", "微血管", "心臟", "神經元", "受器", "動器", "大腦", "小腦",
        "腦幹", "脊髓", "生殖", "無性生殖", "有性生殖", "分裂", "減數分裂", "細胞分裂",
        "遺傳", "基因", "染色體", "DNA", "顯性", "隱性", "突變", "演化", "天擇", "化石",
        "生態系", "食物鏈", "食物網", "生產者", "消費者", "分解者", "生物多樣性", "碳循環",
        "氮循環", "界門綱目科屬種", "植物界", "動物界", "真菌界", "原生生物界", "原核生物界",
        # 理化 (物理與化學)
        "質量", "體積", "密度", "元素", "化合物", "混合物", "純物質", "原子", "分子",
        "質子", "中子", "電子", "原子序", "質量數", "週期表", "金屬", "非金屬", "化學變化",
        "物理變化", "化學式", "化學反應", "氧化", "還原", "燃燒", "酸鹼", "中和", "濃度",
        "pH值", "指示劑", "電解質", "反應速率", "催化劑", "變因", "控制變因", "操縱變因",
        "位置", "位移", "路徑長", "速度", "速率", "加速度", "力", "合力", "重力", "摩擦力",
        "浮力", "壓力", "大氣壓力", "帕斯卡原理", "牛頓運動定律", "慣性", "作用力與反作用力",
        "功", "功率", "能", "動能", "位能", "力學能", "熱量", "比熱", "傳導", "對流", "輻射",
        "蒸發", "凝結", "沸騰", "熔化", "汽化", "昇華", "波", "頻率", "波長", "振幅", "週期",
        "聲音", "音調", "響度", "音色", "光", "反射", "折射", "透鏡", "面鏡", "色散",
        "電流", "電壓", "電阻", "歐姆定律", "串聯", "並聯", "靜電", "磁場", "電磁感應", "馬達",
        # 地球科學
        "岩石", "礦物", "火成岩", "沉積岩", "變質岩", "風化", "侵蝕", "搬運", "沉積",
        "岩漿", "板塊構造學說", "大陸漂移說", "海底擴張說", "地殼", "地函", "地核", "軟流圈",
        "地震", "震源", "震央", "規模", "震度", "斷層", "褶皺", "大氣", "對流層", "平流層",
        "臭氧層", "溫室氣體", "高氣壓", "低氣壓", "等壓線", "鋒面", "冷鋒", "暖鋒", "滯留鋒",
        "颱風", "季風", "海陸風", "潮汐", "滿潮", "乾潮", "大潮", "小潮", "洋流", "黑潮",
        "恆星", "行星", "衛星", "太陽系", "銀河系", "宇宙", "光年", "自轉", "公轉", "日食", "月食"
    }
}
ALL_SUBJECT_TERMS = set().union(*SUBJECT_TERMS.values())

# ==========================================
# 1. 頁面基本設定
# ==========================================
st.set_page_config(
    page_title="台灣中小學試題句子難度檢測系統（雛形）",
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
        
    # [更新] 改用 Regex 掃描匹配句式
    for clause_type, patterns in CONNECTORS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                detected_types.append(clause_type)
                break  # 該類型一旦配對成功，就換下一個句式檢查
            
    if not detected_types:
        dep_labels = {token.dep_ for token in doc}
        if "advcl" in dep_labels or "conj" in dep_labels:
            detected_types.append("複雜修飾句")
        else:
            detected_types.append("簡單句")
            
    return ", ".join(detected_types)

def calculate_vocab_depth(doc: spacy.tokens.Doc, term_set: set) -> int:
    return sum(1 for token in doc if token.text in term_set)

def extract_features_from_doc(doc: spacy.tokens.Doc, term_set: set) -> Dict[str, Any]:
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
        "vocab_depth": calculate_vocab_depth(doc, term_set)
    }

def predict_grade(features: Dict[str, Any], ml_model: Optional[Any]) -> Tuple[str, float]:
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
    
    # [更新] 將新版的進階邏輯句式加入加分條件
    complex_clauses = ["進階論述句", "目的複句", "選擇複句", "遞進複句", "推斷複句", "假轉複句", "取捨複句"]
    if any(c in features["clause_types"] for c in complex_clauses):
        score += 1.0
        
    if features["vocab_depth"] >= 1:
        score += 1.5

    if score >= 4.5: grade_str = "5-6 年級 (高年級) 或以上"
    elif score <= 2.5: grade_str = "1-2 年級 (低年級)"
    else: grade_str = "3-4 年級 (中年級)"
    
    return grade_str, score

def run_batch_analysis(question_list: List[str], nlp_model, difficulty_model, term_set: set) -> pd.DataFrame:
    results = []
    progress_bar = st.progress(0)
    total = len(question_list)
    
    for i, doc in enumerate(nlp_model.pipe(question_list, batch_size=50)):
        feat = extract_features_from_doc(doc, term_set)
        grade_str, raw_score = predict_grade(feat, difficulty_model)
        
        results.append({
            "題目內容": feat["text"],
            "預估適用年級": grade_str,
            "分數_hidden": raw_score,  
            "複句結構與句式": feat["clause_types"],
            "總字數": feat["char_count"],
            "名詞密度": f"{feat['noun_ratio']:.1%}",
            "MDD數值": round(feat["mdd"], 2)
        })
        progress_bar.progress((i + 1) / total)
        
    progress_bar.empty()
    return pd.DataFrame(results)

# ==========================================
# 3.5 視覺化統計與總覽 UI 繪製邏輯
# ==========================================
def render_overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    avg_score = df["分數_hidden"].mean()
    overall_grade = max(1, int(round(avg_score)))
    total_chars = int(df["總字數"].sum())
    avg_mdd = df["MDD數值"].mean()
    
    st.markdown("### 🌟 整體題庫評估總覽")
    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 綜合預估年級", f"{overall_grade} 年級")
    c2.metric("📏 總字數", f"{total_chars} 字")
    c3.metric("🧠 平均依存距離 (MDD)", f"{avg_mdd:.2f}")
    st.divider()
    
    return df.drop(columns=["分數_hidden"])

def render_statistics_charts(df: pd.DataFrame):
    st.markdown("### 📊 批次分析統計圖表")
    col1, col2, col3 = st.columns(3)
    
    grade_counts = df["預估適用年級"].value_counts().reset_index()
    grade_counts.columns = ["年級", "題數"]
    fig_grade = px.pie(grade_counts, names="年級", values="題數", hole=0.4, 
                       title="預估適用年級占比", 
                       color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_grade.update_traces(textposition='inside', textinfo='percent+label')
    fig_grade.update_layout(showlegend=False)
    col1.plotly_chart(fig_grade, use_container_width=True)
    
    clause_series = df["複句結構與句式"].str.split(", ").explode()
    clause_counts = clause_series.value_counts().reset_index()
    clause_counts.columns = ["句式", "出現次數"]
    fig_clause = px.pie(clause_counts, names="句式", values="出現次數", hole=0.4, 
                        title="複句句式出現占比",
                        color_discrete_sequence=px.colors.qualitative.Set3)
    fig_clause.update_traces(textposition='inside', textinfo='percent+label')
    fig_clause.update_layout(showlegend=False)
    col2.plotly_chart(fig_clause, use_container_width=True)
    
    bins = [0, 1, 2, 3, 4, 5, 100]
    labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5以上']
    df_mdd = df.copy()
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
    
    subject = st.selectbox("學科", ["全部學科", "國語文", "數學", "社會", "自然"])
    
    st.markdown("### 👁️ 介面顯示設定")
    show_table = st.checkbox("顯示資料明細表", value=True)
    show_charts = st.checkbox("顯示視覺化圖表", value=True)
    
    if subject == "全部學科":
        current_term_set = ALL_SUBJECT_TERMS
    else:
        current_term_set = SUBJECT_TERMS.get(subject, set())

st.title("📚 台灣中小學試題句子難度檢測系統（雛形）")
st.caption(f"目前分析學科模式：**{subject}** (將依據對應學科之進階詞庫進行難度加權)")

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
                features = extract_features_from_doc(doc, current_term_set)
                predicted_grade_str, predicted_raw_score = predict_grade(features, model)
                
                st.divider()
                cols = st.columns(4)
                cols[0].metric("🎯 預估年級", predicted_grade_str)
                cols[1].metric("📏 總字數", f"{features['char_count']} 字")
                cols[2].metric("🧠 依存距離 (MDD)", f"{features['mdd']:.2f}")
                cols[3].metric("🔗 複句結構", features["clause_types"])
                    
                if show_table:
                    st.subheader("📋 試題特徵明細")
                    st.dataframe({
                        "特徵名稱": ["總詞數", "名詞比例", "動詞比例", "該科進階術語計數"],
                        "數值": [
                            features['word_count'], 
                            f"{features['noun_ratio']:.1%}", 
                            f"{features['verb_ratio']:.1%}", 
                            f"{features['vocab_depth']} 個"
                        ]
                    }, use_container_width=True)

                if show_charts:
                    st.subheader("📊 單題視覺化圖表")
                    col_chart1, col_chart2 = st.columns(2)
                    
                    fig_mdd_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = features['mdd'],
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "MDD 依存距離指標<br><span style='font-size:0.8em;color:gray'>數值越高代表句法越複雜崎嶇</span>"},
                        gauge = {
                            'axis': {'range': [None, 6]},
                            'bar': {'color': "#1f77b4"},
                            'steps' : [
                                {'range': [0, 2.6], 'color': "#d9f0d3"},    
                                {'range': [2.6, 4.0], 'color': "#fff2ae"},  
                                {'range': [4.0, 6.0], 'color': "#fbb4ae"}   
                            ],
                        }
                    ))
                    col_chart1.plotly_chart(fig_mdd_gauge, use_container_width=True)
                    
                    other_ratio = max(0, 1.0 - features['noun_ratio'] - features['verb_ratio'])
                    df_pos = pd.DataFrame({
                        "詞性": ["名詞與專有名詞", "動詞", "其他附屬詞"],
                        "比例": [features['noun_ratio'], features['verb_ratio'], other_ratio]
                    })
                    fig_pos = px.pie(df_pos, names="詞性", values="比例", hole=0.4, 
                                     title="句子詞性結構占比", 
                                     color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_pos.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pos.update_layout(showlegend=False)
                    col_chart2.plotly_chart(fig_pos, use_container_width=True)

# --- TAB 2: 批次查詢與統計儀表板 ---
with tab2:
    batch_mode = st.radio("輸入方式：", ["📋 貼上多行文字", "📂 上傳檔案"], horizontal=True)
    
    if batch_mode == "📋 貼上多行文字":
        batch_text = st.text_area("每行一題：", height=250)
        if st.button("⚡ 開始批次分析", type="primary"):
            q_list = [line.strip() for line in batch_text.split("\n") if line.strip()]
            if q_list:
                res_df = run_batch_analysis(q_list, nlp, model, current_term_set)
                st.divider()
                
                display_df = render_overall_summary(res_df)
                
                if show_charts:
                    render_statistics_charts(display_df)
                
                if show_table:
                    st.markdown("### 📝 詳細題目檢測報表")
                    st.dataframe(display_df, use_container_width=True)
                
                st.download_button("📥 下載 CSV 報告", display_df.to_csv(index=False).encode("utf-8-sig"), "批次檢測報告.csv", "text/csv")
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
                    if st.button("⚡ 開始批次分析", type="primary"):
                        res_df = run_batch_analysis(q_list, nlp, model, current_term_set)
                        st.divider()
                        
                        display_df = render_overall_summary(res_df)
                        
                        if show_charts:
                            render_statistics_charts(display_df)
                        
                        if show_table:
                            st.markdown("### 📝 詳細題目檢測報表")
                            st.dataframe(display_df, use_container_width=True)
                        
                        st.download_button("📥 下載結果", display_df.to_csv(index=False).encode("utf-8-sig"), "檔案分析報告.csv", "text/csv")
            except Exception as e:
                st.error(f"讀取失敗：{e}")
