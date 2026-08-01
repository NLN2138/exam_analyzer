import os
import re
import joblib
import pandas as pd
import numpy as np
import spacy
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple

# ==========================================
# 0. 靜態常數與黑名單定義
# ==========================================

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

SUBJECT_TERMS = {
    "國語文": {
        "修辭", "譬喻", "借代", "轉化", "擬人", "擬物", "誇飾", "排比", "層遞", "設問", 
        "對偶", "頂真", "映襯", "雙關", "象徵", "呼告", "倒裝", "韻文", "詞牌", "新詩", 
        "意象", "寓言", "絕句", "律詩", "古體詩", "近體詩", "樂府", "賦", "散文", "小說", 
        "記敘文", "抒情文", "說明文", "議論文", "應用文", "書信", "便條", "對聯", "題辭",
        "敘事觀點", "文眼", "主旨", "大意", "段落", "伏筆", "懸念", "烘托", "借景抒情", 
        "托物言志", "六書", "象形", "指事", "會意", "形聲", "轉注", "假借", "部首", "筆畫", 
        "字形", "字音", "字義", "詞性", "名詞", "動詞", "形容詞", "副詞", "代詞", "介詞", 
        "連詞", "助詞", "量詞", "歎詞", "句型", "直述句", "疑問句", "祈使句", "感嘆句",
        "平仄", "押韻", "對仗", "經史子集", "唐宋八大家", "詩仙", "詩聖", "詞眼", "曲",
        "偏旁", "部首", "繁體", "簡體", "文言文", "白話文", "語錄體", "紀傳體", "編年體",
        "國音", "聲母", "韻母", "結合韻", "聲調", "破音字", "同音字", "多音字", "形近字",
        "成語", "諺語", "歇後語", "慣用語", "外來語", "敬辭", "謙辭", "稱謂", "文法",
        "文本", "情節", "人物", "背景", "衝突", "高潮", "結局", "起承轉合", "第一人稱",
        "第三人稱", "倒敘", "順敘", "插敘", "補敘", "白描", "寫實", "浪漫", "魔幻", "史詩",
        "神話", "傳說", "民間故事", "童話", "科幻", "武俠", "推理", "寓意", "絃外之音",
        "言外之意", "主觀", "客觀", "批判", "賞析", "鑑賞", "共鳴", "流派", "文學史"
    },
    "數學": {
        "整數", "分數", "小數", "質數", "合數", "因數", "倍數", "公因數", "公倍數", 
        "最大公因數", "最小公倍數", "絕對值", "有理數", "無理數", "實數", "正數", "負數",
        "倒數", "相反數", "科學記號", "四捨五入", "無條件進位", "無條件捨去", "概數",
        "比例", "正比", "反比", "百分率", "千分率", "折現率", "利率", "本金", "利息",
        "演算法", "方程式", "函數", "未知數", "變數", "常數", "係數", "多項式", "單項式",
        "同類項", "指數", "底數", "對數", "根號", "平方根", "立方根", "一次方程式", 
        "二次方程式", "聯立方程式", "不等式", "等差數列", "等比數列", "級數", "公差", "公比",
        "公式", "代入", "展開", "因式分解", "配方法", "十字交乘", "公式解", "判別式",
        "幾何", "點", "線", "面", "角", "度", "射線", "線段", "平行", "垂直", "相交",
        "三角形", "直角三角形", "等腰三角形", "正三角形", "鈍角三角形", "銳角三角形",
        "四邊形", "正方形", "長方形", "平行四邊形", "梯形", "菱形", "箏形", "多邊形",
        "圓", "半徑", "直徑", "圓周", "圓周率", "弧", "弦", "扇形", "弓形", "圓心角", "圓周角",
        "全等", "相似", "比例尺", "對稱", "線對稱", "點對稱", "旋轉", "平移", "翻轉",
        "面積", "體積", "表面積", "周長", "柱體", "錐體", "球體", "長方體", "正方體",
        "座標", "平面座標", "直角座標", "象限", "原點", "Ｘ軸", "Ｙ軸", "斜率", "畢氏定理",
        "統計", "資料", "圖表", "長條圖", "折線圖", "圓形圖", "直方圖", "次數分配表",
        "平均數", "中位數", "眾數", "全距", "四分位數", "盒狀圖", "標準差", "變異數",
        "機率", "事件", "樣本空間", "期望值", "排列", "組合", "樹狀圖", "相對次數"
    },
    "社會": {
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
        "史前時代", "舊石器時代", "新石器時代", "金屬器時代", "大航海時代",
        "荷西時期", "鄭氏時期", "清領時期", "日治時期", "戰後時期", "解嚴", "戒嚴", 
        "民主化", "白色恐怖", "二二八事件", "朝代", "皇帝", "封建", "帝國", "殖民", 
        "條約", "不平等條約", "革命", "啟蒙運動", "文藝復興", "工業革命", "冷戰", 
        "第一次世界大戰", "第二次世界大戰", "聯合國", "十字軍東征", "資本主義", "共產主義",
        "經度", "緯度", "赤道", "本初子午線", "時區", "國際換日線", "比例尺", "圖例",
        "等高線", "地形", "高山", "丘陵", "台地", "平原", "盆地", "火山", "海岸",
        "氣候", "天氣", "季風", "洋流", "溫室效應", "氣壓", "降水", "氣溫", "水文"
    },
    "自然": {
        "細胞", "細胞膜", "細胞壁", "細胞質", "細胞核", "葉綠體", "粒線體", "液胞",
        "光合作用", "呼吸作用", "酵素", "擴散作用", "滲透作用", "生物體", "組織", "器官",
        "系統", "消化系統", "循環系統", "呼吸系統", "排泄系統", "神經系統", "內分泌系統",
        "生殖", "無性生殖", "有性生殖", "分裂", "減數分裂", "細胞分裂",
        "遺傳", "基因", "染色體", "DNA", "顯性", "隱性", "突變", "演化", "天擇", "化石",
        "生態系", "食物鏈", "食物網", "生產者", "消費者", "分解者", "生物多樣性", "碳循環",
        "質量", "體積", "密度", "元素", "化合物", "混合物", "純物質", "原子", "分子",
        "質子", "中子", "電子", "原子序", "質量數", "週期表", "金屬", "非金屬", "化學變化",
        "物理變化", "化學式", "化學反應", "氧化", "還原", "燃燒", "酸鹼", "中和", "濃度",
        "pH值", "指示劑", "電解質", "反應速率", "催化劑", "變因", "控制變因", "操縱變因",
        "位置", "位移", "路徑長", "速度", "速率", "加速度", "力", "合力", "重力", "摩擦力",
        "浮力", "壓力", "大氣壓力", "帕斯卡原理", "牛頓運動定律", "慣性", "作用力與反作用力",
        "功", "功率", "能", "動能", "位能", "力學能", "熱量", "比熱", "傳導", "對流", "輻射",
        "波", "頻率", "波長", "振幅", "週期", "聲音", "音調", "響度", "音色", "光", "反射", "折射",
        "電流", "電壓", "電阻", "歐姆定律", "串聯", "並聯", "靜電", "磁場", "電磁感應",
        "岩石", "礦物", "火成岩", "沉積岩", "變質岩", "風化", "侵蝕", "搬運", "沉積",
        "板塊構造學說", "大陸漂移說", "地殼", "地函", "地核", "軟流圈", "地震", "震源", "震央"
    }
}
ALL_SUBJECT_TERMS = set().union(*SUBJECT_TERMS.values())

CONNECTORS = {
    "因果複句": [r"因為.*所以", r"由於", r"導致", r"以致於", r"因此", r"爰此"],
    "假轉複句": [r"雖然.*但", r"儘管", r"然而", r"卻", r"縱使", r"固然"],
    "目的複句": [r"為了", r"以便", r"以利", r"旨在", r"用以"],
    "選擇複句": [r"不是.*就是", r"或者.*或者", r"抑或", r"還是"],
    "遞進複句": [r"不但.*而且", r"不僅", r"甚至", r"更何況", r"尤有甚者"],
    "推斷複句": [r"既然.*就", r"可見", r"據此", r"推測", r"照理說"],
    "取捨複句": [r"與其.*不如", r"寧可.*也不", r"寧願"],
    "條件複句": [r"如果.*就", r"若.*則", r"只要.*就", r"只有.*才", r"除非"]
}

# 試題低難度常見指示詞/結構黑名單
INSTRUCTION_PATTERNS = [
    r'請[畫劃選填寫看算選]看', r'下列[何者|敘述]', r'正確的[打畫]', r'填入[適當|正確]',
    r'回答[下列|以下]問題', r'選出[一個|正確]', r'第.*題', r'每題.*分', r'共.*分',
    r'閱讀測驗', r'一、', r'二、', r'三、', r'四、', r'五、', r'六、', r'七、', r'八、',
    r'看圖[回答|填]', r'勾選', r'連連看', r'圈圈看'
]

DEFAULT_SINGLE_Q = "樹上的蘋果又紅又大，看起來非常好吃。"

DEFAULT_BATCH_Q = """樹上的蘋果又紅又大，看起來非常好吃。
放學回到家，我會先把手洗乾淨，然後才開始寫作業。
如果明天早上沒有下雨，我們就一起去公園騎腳踏車。
因為他每天都很認真練習書法，所以在這次的比賽中得到了第一名。
雖然這道數學題看起來非常複雜，但是只要畫圖仔細思考，就能找到答案。
閱讀課外讀物不但能幫助我們認識世界，而且能豐富我們的想像力。
在進行科學探究時，只有嚴格控制所有的實驗變因，才能確保最終數據的準確性。
面對團隊合作的意見分歧，我們與其互相爭論誰的點子最好，不如冷靜下來尋找共識。
現代民主國家設立了權力分立的憲政體制，以免少數掌權者濫用職權而侵害人民的基本權益。
藝術家嘗試運用跨領域的數位互動多媒體視覺效果與傳統水墨繪畫技法進行深度融合，進而在充滿未來感的展覽空間中營造出一種能夠誘導觀者進行深刻自我審視與哲學反思的沈浸式藝術體驗。
為了有效緩解因城市化進程迅速推進與車輛持有量暴增所帶來的市中心交通癱瘓與空氣品質惡化問題，市政府決定籌措巨額預算全面建構以軌道運輸為骨幹且低碳環保的大眾運輸系統。
這場關乎全人類未來生存命運的大氣與環境科學國際高峰研討會，集結了來自全球數十個國家在氣候變遷領域具有卓越學術貢獻的頂尖學者，共同針對全球暖化對亞熱帶地區糧食生產安全性所造成的嚴峻衝擊進行深度的研討與對策擬定。
基於現象學還原論針對主體間性所提出的解構性思維，學者們試圖透過重新建構個體在意識流演變過程中所經驗到的時空感知經驗，來回應當代存在主義哲學在面臨數位科技虛擬化浪潮時所遭遇到的本體論危機與價值轉向議題。
"""

DEFAULT_EXAM_PAPER = """臺北市立OO國民中學 113 學年度第一學期 綜合學科 期末定期評量試卷
班級：八年級 ___班  姓名：_________  座號：___ (每題4分，共100分)

一、 基礎題（請選出一個最適當的答案，並在答案卡上填塗）
1. (  ) 下列關於細胞構造的敘述，何者完全正確？
(A) 葉綠體是細胞進行呼吸作用的主要場所
(B) 細胞核內含有遺傳物質 DNA，能控制細胞的生理活動
(C) 所有植物細胞都具有大液胞與細胞壁，且均能進行光合作用
(D) 粒線體能將光能轉化為化學能以維持生命運作

2. (  ) 關於我國憲政體制的權力分立原則，下列何者說明正確？
(A) 行政院負責審議法律案與中央政府總預算案
(B) 監察院掌理彈劾、糾舉及審計權，以防止公務員濫用職權
(C) 司法院由總統直接指揮，負責審理民事與刑事訴訟
(D) 立法院得對行政院院長提出不信任案，並得隨時解散立法院

二、 閱讀理解與題組（閱讀下列文本後回答第 3-5 題）
    為了有效緩解因城市化進程迅速推進與車輛持有量暴增所帶來的市中心交通癱瘓與空氣品質惡化問題，市政府決定籌措巨額預算，全面建構以軌道運輸為骨幹且低碳環保的大眾運輸系統。
    在進行這項重大公共工程規劃時，專家團隊不僅需要精確評估各路線的運輸效率與機會成本，更必須審慎考量環境影響與社會公平。面對各界意見分歧，決策者與其互相爭論，不如透過建立多元觀點的溝通機制，尋找社會最大公義之共識。

3. (  ) 根據上文脈絡，市政府推動低碳大眾運輸系統的核心旨在解決何種課題？
(A) 促進地方觀光產業發展 (B) 緩解交通癱瘓與空氣品質惡化 (C) 降低車輛車牌考照門檻 (D) 提高汽機車銷售稅率

4. (  ) 文中所提及「評估各路線的機會成本」，在經濟學上所指的涵義為何？
(A) 興建軌道系統所消耗的電力總總支出 (B) 放棄掉的其他可能選擇中，價值最高者 (C) 政府向市民徵收的交通稅收總額 (D) 搭乘大眾運輸工具時的單程票價
"""

# ==========================================
# 1. 頁面設定
# ==========================================
st.set_page_config(
    page_title="台灣 K-12 與高中試題難度檢測系統",
    page_icon="📚",
    layout="wide"
)

# ==========================================
# 2. 載入模型
# ==========================================
@st.cache_resource(show_spinner="載入 NLP 模型中...")
def load_nlp():
    try:
        return spacy.load("zh_core_web_sm")
    except OSError:
        st.error("❌ 找不到 spaCy 中文模型！請確保以 python -m spacy download zh_core_web_sm 安裝。")
        st.stop()

@st.cache_resource(show_spinner="載入評分模型中...")
def load_difficulty_model():
    model_path = "mdd_baseline_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# ==========================================
# 3. 試題清洗與拆分引擎 (智慧降噪)
# ==========================================
def sanitize_exam_paper(raw_text: str, min_length: int = 14) -> Tuple[List[str], List[str]]:
    cleaned = re.sub(r'[(（][^()（）]*每[題字格分].*?[)）]', '', raw_text)
    cleaned = re.sub(r'[一二三四五六七八九十]+\s*[\u4e00-\u9fa5]+[：:]', '', cleaned)
    cleaned = re.sub(r'^\s*[\d\w]+\s*[\.、．]', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'[↓｜|]', '', cleaned)
    cleaned = re.sub(r'\([ 0-9A-Za-z\s]*\)|（[ 0-9A-Za-z\s]*）', '', cleaned)
    
    raw_sentences = re.split(r'[\n。！？!?]', cleaned)
    valid_sentences = []
    filtered_out = []
    
    for s in raw_sentences:
        s_strip = s.strip()
        s_strip = re.sub(r'^\s*\d+\s*', '', s_strip)
        if not s_strip:
            continue
            
        if len(s_strip) < min_length:
            filtered_out.append(f"[過短] {s_strip}")
            continue
            
        is_instruction = any(re.search(pat, s_strip) for pat in INSTRUCTION_PATTERNS)
        if is_instruction and len(s_strip) < 28:
            filtered_out.append(f"[指令句] {s_strip}")
            continue
            
        if re.search(r'^(選項|答案|選擇|填空|改錯字|配對)', s_strip):
            filtered_out.append(f"[結構標籤] {s_strip}")
            continue
            
        valid_sentences.append(s_strip)
                
    return valid_sentences, filtered_out

# ==========================================
# 4. 難度特徵運算邏輯
# ==========================================
def analyze_clause_types(doc: spacy.tokens.Doc) -> str:
    text = doc.text
    detected_types = []
    
    if any(kw in text for kw in ADVANCED_KEYWORDS):
        detected_types.append("進階論述句")
        
    for clause_type, patterns in CONNECTORS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                detected_types.append(clause_type)
                break 
            
    if not detected_types:
        dep_labels = {token.dep_ for token in doc}
        if "advcl" in dep_labels or "conj" in dep_labels:
            detected_types.append("複雜修飾句")
        else:
            detected_types.append("簡單句")
            
    return ", ".join(detected_types)

def calculate_vocab_depth(doc: spacy.tokens.Doc, term_set: set) -> int:
    text = doc.text
    matched = sum(1 for token in doc if token.text in term_set)
    matched_sub = sum(1 for term in term_set if term in text)
    return max(matched, matched_sub)

def extract_features_from_doc(doc: spacy.tokens.Doc, term_set: set) -> Dict[str, Any]:
    word_count = len(doc)
    char_count = len(doc.text)
    
    nouns_count = sum(1 for token in doc if token.pos_ in ("NOUN", "PROPN"))
    verbs_count = sum(1 for token in doc if token.pos_ == "VERB")
    
    noun_ratio = nouns_count / word_count if word_count > 0 else 0.0
    verb_ratio = verbs_count / word_count if word_count > 0 else 0.0
    
    valid_tokens = [t for t in doc if t.pos_ not in ("PUNCT", "SPACE")]
    
    if valid_tokens:
        token_to_valid_idx = {t.i: idx for idx, t in enumerate(valid_tokens)}
        dep_distances = []
        for t in valid_tokens:
            if t.head != t and t.head.i in token_to_valid_idx:
                dist = abs(token_to_valid_idx[t.i] - token_to_valid_idx[t.head.i])
                dep_distances.append(dist)
        base_mdd = sum(dep_distances) / len(dep_distances) if dep_distances else 0.0
    else:
        base_mdd = 0.0
    
    raw_text = doc.text
    sub_clauses = re.split(r'[，。；]', raw_text)
    max_clause_len = max(len(c) for c in sub_clauses) if sub_clauses else char_count
    clause_delimiters = raw_text.count("，") + raw_text.count("；")
    
    punct_factor = 0.10 * clause_delimiters
    long_chunk_factor = max(0.0, (max_clause_len - 35) / 10) * 0.15 if max_clause_len > 35 else 0.0
    noun_density_factor = (noun_ratio - 0.35) * 1.5 if noun_ratio > 0.35 else 0.0
    
    total_compensation = 1.0 + punct_factor + long_chunk_factor + noun_density_factor
    adjusted_mdd = base_mdd * total_compensation if char_count >= 40 else base_mdd

    return {
        "text": doc.text,
        "char_count": char_count,
        "word_count": word_count,
        "noun_ratio": noun_ratio,
        "verb_ratio": verb_ratio,
        "base_mdd": base_mdd,
        "mdd": adjusted_mdd,
        "clause_types": analyze_clause_types(doc),
        "vocab_depth": calculate_vocab_depth(doc, term_set)
    }

def predict_grade(features: Dict[str, Any], ml_model: Optional[Any]) -> Tuple[str, float]:
    # 🌟 1. 下修初始基準分 (從 3.0 下修至 1.5)
    score = 1.5
    
    # ML 模型輔助
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
                score = float(raw_pred) * 0.85  # ML 模型預測值同步打 85 折
        except Exception:
            pass

    # 🌟 2. 下修字數加成門檻
    char_len = features["char_count"]
    if char_len <= 15: score -= 1.0
    elif char_len <= 25: score -= 0.5
    elif 26 <= char_len <= 45: score += 0.5
    elif 46 <= char_len <= 70: score += 1.0     # 原本 +1.5
    elif 71 <= char_len <= 100: score += 1.5    # 原本 +2.5
    elif char_len > 100: score += 2.5           # 原本 +4.0
    
    # 🌟 3. 收斂 MDD 依存距離加分
    mdd = features["mdd"]
    if mdd < 2.0: score -= 1.0                  # 原本 -1.5
    elif 2.0 <= mdd < 2.8: score -= 0.5
    elif 3.5 <= mdd < 4.2: score += 0.5         # 原本 +1.0
    elif 4.2 <= mdd < 5.0: score += 1.2         # 原本 +2.5
    elif mdd >= 5.0: score += 2.0               # 原本 +4.0

    # 4. 名詞密度加成適度微調
    noun_r = features["noun_ratio"]
    if noun_r < 0.20: score -= 0.5
    elif 0.35 <= noun_r < 0.45: score += 0.5
    elif noun_r >= 0.45: score += 1.2           # 原本 +2.0

    # 5. 複句結構
    complex_clauses = ["進階論述句", "目的複句", "選擇複句", "遞進複句", "推斷複句", "假轉複句", "取捨複句"]
    if any(c in features["clause_types"] for c in complex_clauses):
        score += 1.0                              # 原本 +1.5

    # 🌟 6. 重設學科術語加分與保底門檻 (避免國中題被誤判為高中)
    v_depth = features["vocab_depth"]
    if v_depth == 1: score += 0.5
    elif v_depth == 2: score += 1.0
    elif v_depth >= 3: score += 2.0             # 原本 +3.5
    
    # 保底門檻修正：更高門檻才觸發高中
    if v_depth >= 5 and score < 8.5:
        score = max(score, 8.5)
    elif v_depth >= 3 and score < 6.5:
        score = max(score, 6.5)
    elif v_depth >= 2 and score < 4.5:
        score = max(score, 4.5)

    # 7. 短句壓低
    if char_len < 20 and v_depth == 0 and "簡單句" in features["clause_types"]:
        score = min(score, 2.5)

    # 🌟 8. 提高各年級評定門檻 (使整體判定往低年級挪移)
    if score >= 10.0: grade_str = "10-12 年級 (高中或以上)"   # 原本 9.5
    elif score >= 7.5: grade_str = "7-9 年級 (國中)"          # 原本 7.0
    elif score >= 5.0: grade_str = "5-6 年級 (國小高年級)"
    elif score >= 3.0: grade_str = "3-4 年級 (國小中年級)"
    else: grade_str = "1-2 年級 (國小低年級)"
    
    return grade_str, score


def map_score_to_grade_str(avg_score: float) -> str:
    """ 考卷整體分數映射門檻同步下修校正 """
    if avg_score >= 10.0: return "10-12 年級 (高中或以上)"
    elif avg_score >= 7.5: return "7-9 年級 (國中)"
    elif avg_score >= 5.0: return "5-6 年級 (國小高年級)"
    elif avg_score >= 3.0: return "3-4 年級 (國小中年級)"
    else: return "1-2 年級 (國小低年級)"

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
            "學科術語數": feat["vocab_depth"],
            "MDD數值": round(feat["mdd"], 2)
        })
        progress_bar.progress((i + 1) / total)
        
    progress_bar.empty()
    return pd.DataFrame(results)

# ==========================================
# 5. 視覺化繪圖函數 (含 TAB 1 單句與 TAB 3 加權)
# ==========================================
def render_single_sentence_charts(features: Dict[str, Any], raw_score: float):
    """
    單句分析專屬視覺化：難度積分儀表板與特徵強弱雷達圖
    """
    st.markdown("### 📊 單句難度特徵分析儀表板")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = min(12.0, max(1.0, raw_score)),
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "難度數值落點 (1-12年級)", 'font': {'size': 16}},
            gauge = {
                'axis': {'range': [1, 12], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#2A648E"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [1, 3], 'color': '#E8F5E9'},   # 國小低
                    {'range': [3, 5], 'color': '#C8E6C9'},   # 國小中
                    {'range': [5, 7], 'color': '#FFF9C4'},   # 國小高
                    {'range': [7, 9.5], 'color': '#FFE0B2'}, # 國中
                    {'range': [9.5, 12], 'color': '#FFCDD2'} # 高中
                ],
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c2:
        categories = ['字數長度', 'MDD依存距離', '名詞密度', '學科術語數']
        
        val_len = min(100, (features['char_count'] / 100) * 100)
        val_mdd = min(100, (features['mdd'] / 6.0) * 100)
        val_noun = min(100, features['noun_ratio'] * 200)
        val_term = min(100, (features['vocab_depth'] / 4) * 100)
        
        values = [val_len, val_mdd, val_noun, val_term]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='此單句特徵',
            line_color='#1E88E5'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            height=260,
            title={'text': "特徵強度多維雷達圖", 'font': {'size': 16}},
            margin=dict(l=40, r=40, t=40, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

def render_overall_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, float, int, float]:
    """
    🌟 核心調整：改採前 50% 最具鑑別度的語句 (P50 中位數以上) 作為整份考卷難度加權
    """
    scores = df["分數_hidden"].values
    if len(scores) >= 4:
        top_50_cutoff = np.percentile(scores, 50)
        hard_scores = [s for s in scores if s >= top_50_cutoff]
        overall_score = float(np.mean(hard_scores))
    else:
        overall_score = float(np.mean(scores))

    overall_grade_str = map_score_to_grade_str(overall_score)
    total_chars = int(df["總字數"].sum())
    
    if len(df) >= 4:
        top_mdd_cutoff = np.percentile(df["MDD數值"].values, 50)
        avg_mdd = df[df["MDD數值"] >= top_mdd_cutoff]["MDD數值"].mean()
    else:
        avg_mdd = df["MDD數值"].mean()
    
    st.markdown("### 🌟 整體考卷深度評估總覽")
    st.caption("💡 **評估加權機制**：考卷難度採 **前 50% 最具鑑別度的核心語句/長文** 進行加權計算（已過濾無意義結構與指示雜訊）。")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 考卷綜合預估年級", overall_grade_str)
    c2.metric("📏 採樣有效字數", f"{total_chars} 字")
    c3.metric("🧠 核心語句平均 MDD", f"{avg_mdd:.2f}")
    st.divider()
    
    display_df = df.drop(columns=["分數_hidden"])
    return display_df, overall_score, total_chars, avg_mdd

def render_statistics_charts(df: pd.DataFrame):
    st.markdown("### 📊 考卷難度分布與句式視覺化")
    col1, col2, col3 = st.columns(3)
    
    grade_counts = df["預估適用年級"].value_counts().reset_index()
    grade_counts.columns = ["年級", "題數"]
    fig_grade = px.pie(grade_counts, names="年級", values="題數", hole=0.4, 
                       title="採樣句年級分布占比", 
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
    
    bins = [0, 2.5, 3.5, 4.5, 6.0, 100]
    labels = ['<2.5', '2.5-3.5', '3.5-4.5', '4.5-6.0', '6.0+']
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
# 6. 前端介面與頁籤規劃
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

st.title("📚 台灣中小學與高中試題難度檢測系統")
st.caption(f"目前分析學科模式：**{subject}** (涵蓋國小低年級至高中或以上程度)")

tab1, tab2, tab3 = st.tabs(["✍️ 單句分析", "📋 多句分析", "📄 整份考題分析 (智慧降噪)"])

# --- TAB 1: 單句檢測 ---
with tab1:
    question_text = st.text_area(
        "題目文字", 
        height=130, 
        placeholder=f"請輸入單一試題...\n\n若未輸入內容點選分析，將自動載入預設範例題：\n{DEFAULT_SINGLE_Q}"
    )

    if st.button("🚀 開始檢測單句", type="primary"):
        target_text = question_text.strip()
        if not target_text:
            target_text = DEFAULT_SINGLE_Q
            st.info("💡 您未輸入內容，已自動載入**預設單句**進行分析。")

        with st.spinner("分析中..."):
            doc = nlp(target_text)
            features = extract_features_from_doc(doc, current_term_set)
            predicted_grade_str, predicted_raw_score = predict_grade(features, model)
            
            st.divider()
            cols = st.columns(4)
            cols[0].metric("🎯 預估年級", predicted_grade_str)
            cols[1].metric("📏 總字數", f"{features['char_count']} 字")
            cols[2].metric("🧠 依存距離 (MDD)", f"{features['mdd']:.2f}")
            cols[3].metric("🔗 複句結構", features["clause_types"])
            
            # 單句視覺化圖表
            if show_charts:
                render_single_sentence_charts(features, predicted_raw_score)
                
            if show_table:
                st.subheader("📋 試題特徵明細")
                st.dataframe({
                    "特徵名稱": ["總詞數 (含標點)", "名詞比例", "動詞比例", "該科進階術語計數", "原始 MDD", "修正 MDD"],
                    "數值": [
                        features['word_count'], 
                        f"{features['noun_ratio']:.1%}", 
                        f"{features['verb_ratio']:.1%}", 
                        f"{features['vocab_depth']} 個",
                        f"{features['base_mdd']:.2f}",
                        f"{features['mdd']:.2f}"
                    ]
                }, use_container_width=True)

# --- TAB 2: 多句批次查詢 ---
with tab2:
    batch_mode = st.radio("輸入方式：", ["📋 貼上多行文字", "📂 上傳檔案"], horizontal=True)
    
    if batch_mode == "📋 貼上多行文字":
        batch_text = st.text_area("每行一題：", height=280, placeholder=f"請貼上多行試題...\n\n預設範例：\n{DEFAULT_BATCH_Q}")
        if st.button("⚡ 開始批次分析", type="primary"):
            target_batch_text = batch_text.strip() or DEFAULT_BATCH_Q
            q_list = [line.strip() for line in target_batch_text.split("\n") if line.strip()]
            
            if q_list:
                res_df = run_batch_analysis(q_list, nlp, model, current_term_set)
                st.divider()
                display_df, avg_score, total_chars, avg_mdd = render_overall_summary(res_df)
                if show_charts: render_statistics_charts(display_df)
                if show_table: st.dataframe(display_df, use_container_width=True)

# --- TAB 3: 整份考題分析 (智慧降噪與 Top 50% 鑑別度加權) ---
with tab3:
    st.markdown("### 🧹 考題自動雜訊過濾與深度檢測")
    st.info("💡 **全卷檢測防稀釋原理**：\n1. **智慧降噪**：自動過濾「請回答下列問題」、「選出正確的...」等指示句、大題標頭與括號，避免無意義短句拉低難度。\n2. **Top 50% 鑑別度加權**：採考卷中最具鑑別度的前 50% 核心語句決定整份考卷適用年級。")
    
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        min_char_limit = st.slider("📏 採樣句數最低字數門檻", min_value=8, max_value=30, value=14, step=2, help="小於此字數的無意義短句或配分說明將被自動忽略。")
    
    raw_exam_paper = st.text_area(
        "請貼上整份考題文字：",
        height=320,
        placeholder=f"請在此直接貼上完整的考題內文...\n\n若未輸入內容點選分析，將自動載入預設試卷範例：\n{DEFAULT_EXAM_PAPER}"
    )
    
    if st.button("🔍 雜訊過濾並開始分析考題", type="primary"):
        exam_input = raw_exam_paper.strip()
        if not exam_input:
            exam_input = DEFAULT_EXAM_PAPER
            st.info("💡 您未輸入考題內容，已自動載入**預設考題範例**進行降噪與深度分析。")

        with st.spinner("正在進行文本降噪、結構切割與深度特徵提取..."):
            extracted_sentences, filtered_noise = sanitize_exam_paper(exam_input, min_length=min_char_limit)
            
        if not extracted_sentences:
            st.error("❌ 找不到符合字數門檻的有效句子，請嘗試降低採樣字數門檻！")
        else:
            st.success(f"✅ 成功從考題中過濾雜訊，擷取出 **{len(extracted_sentences)}** 個具代表性的有效試題語句！")
            
            res_df = run_batch_analysis(extracted_sentences, nlp, model, current_term_set)
            st.divider()
            
            display_df, overall_score, total_chars, avg_mdd = render_overall_summary(res_df)
            
            if show_charts:
                render_statistics_charts(display_df)
                
            if show_table:
                st.markdown("### 📝 擷取之有效試題句段檢測明細")
                st.dataframe(display_df, use_container_width=True)
            
            with st.expander("👁️ 檢視被自動過濾的考題雜訊與指示句（點擊展開）"):
                st.write(f"共過濾掉 **{len(filtered_noise)}** 個雜訊片段：")
                st.json(filtered_noise[:30])
            
            st.download_button(
                "📥 下載整份考題分析報告 CSV", 
                display_df.to_csv(index=False).encode("utf-8-sig"), 
                "考題分析報告.csv", 
                "text/csv"
            )
