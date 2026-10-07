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
from openai import OpenAI
import io
import PyPDF2
import docx

# ==========================================
# 0.1 OpenAI API 初始化設定
# ==========================================
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=openai_api_key) if openai_api_key else None

def get_smart_split_suggestion(sentence: str, mdd: float) -> str:
    """呼叫 ChatGPT API 進行智慧拆句建議"""
    if not client:
        return "⚠️ 未偵測到 OPENAI_API_KEY，請先至 Streamlit Secrets 設定金鑰。"
        
    prompt = f"""
    你是一位專業的華語文教材編輯與句法學專家。
    請根據以下高難度句子（MDD: {mdd}）進行「智慧拆句與改寫建議」。

    請依據以下原則給出具體建議：
    1. 依據標點、轉折/因果關聯詞或連動動詞節點進行斷句。
    2. 將長情境鋪陳或過長的修飾語獨立成句。
    3. 改寫後的句子應保持原意，但大幅降低學生的認知負荷（MDD）。

    原句：
    {sentence}

    請直接輸出以下格式（無須多餘問候）：
    **【複雜度分析】**：簡短一句話說明為何這句負擔大。
    **【改寫建議】**：(提供拆分或重組後的句子，並用亮點標示修改處)
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ API 呼叫失敗：{str(e)}"

# ==========================================
# 0.5 檔案解析 Helper 函數
# ==========================================
def extract_text_from_pdf(file_bytes) -> str:
    """從上傳的 PDF 檔案流中萃取文字"""
    try:
        reader = PyPDF2.PdfReader(file_bytes)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        return f"⚠️ PDF 讀取失敗: {str(e)}"

def extract_text_from_docx(file_bytes) -> str:
    """從上傳的 Word 檔案流中萃取文字"""
    try:
        doc = docx.Document(file_bytes)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"⚠️ Word 讀取失敗: {str(e)}"

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

INSTRUCTION_PATTERNS = [
    r'請[畫劃選填寫看算選]看', r'下列[何者|敘述]', r'正確的[打畫]', r'填入[適當|正確]',
    r'回答[下列|以下]問題', r'選出[一個|正確]', r'第.*題', r'每題.*分', r'共.*分',
    r'閱讀測驗', r'一、', r'二、', r'三、', r'四、', r'五、', r'六、', r'七、', r'八、',
    r'看圖[回答|填]', r'勾選', r'連連看', r'圈圈看'
]

# ==========================================
# 預設題庫內容
# ==========================================
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
縱使那些長期致力於推動跨領域人工智慧倫理規範與演算法透明度機制的頂尖學者，早已透過無數次嚴謹的實證研究，深刻揭示了缺乏監管的機器學習模型可能對當代民主憲政體制造成難以彌補的系統性破壞；然而高度依賴數據變現以維持資本指數型擴張的跨國科技巨頭，仍舊基於短期商業利益與壟斷市場的戰略考量，蓄意規避任何具備實質約束力的外部審查機制。
"""

DEFAULT_EXAM_PAPER = """
快樂市市立星辰國民小學 112學年度上學期 六年級自然科學領域期末評量  
六年   班   號   姓名：       

一、 是非題 （每題2 分，共20 分） 
 1.(   )颱風來臨時，我們應該到海邊觀浪，以親自了解天氣變化的劇烈程度。
 2.(   )使用定滑輪雖然無法省力，但可以改變施力的方向，讓工作更加方便。
 3.(   )電磁鐵的磁力強弱與線圈纏繞的圈數及串聯的電池數量完全無關。
 4.(   )流水對地表的侵蝕作用，在坡度越陡峭、水流越急的地方通常越明顯。
 5.(   )在槓桿原理中，當施力臂大於抗力臂時，這是一個費力但省時的機械裝置。
 6.(   )觀察地面天氣圖時，標示為「H」的符號代表高氣壓中心，通常會帶來晴朗的天氣。
 7.(   )臺灣位於歐亞板塊與菲律賓海板塊交界帶，因此地震頻繁，平時應確實做好防震準備。
 8.(   )堅硬的岩石受到風吹、日曬、雨淋等自然力量影響而逐漸碎裂的過程，我們稱為「搬運作用」。
 9.(   )自製簡易馬達時，漆包線兩端的絕緣漆必須完全刮除乾淨，才能產生持續轉動的效果。
 10.(   )在生態系中，綠色植物能夠利用陽光進行光合作用並自行製造養分，因此被稱為生產者。
 
二、 選擇題 （每題2 分，共20 分） 
 1.(   )下列哪一種簡單機械在日常生活中的應用，其主要目的是為了「省力」？
 ○１吃飯用的筷子 ○２夾取物品的鑷子 ○３打開汽水的開瓶器 ○４打掃用的掃把。
 2.(   )當冷鋒過境臺灣時，通常會帶來什麼樣的氣象變化？
 ○１氣溫急遽上升，天氣晴朗 ○２氣溫明顯下降，容易出現短暫陣雨 ○３氣溫保持不變，無風無雨 ○４氣溫急劇上升並伴隨颱風生成。
 3.(   )關於地震相關知識的敘述，下列何者正確？
 ○１芮氏規模主要用來表示地震破壞的程度 ○２震央是指地下岩層最先開始斷裂的實際位置 ○３地震發生時應立即搭乘電梯逃生以節省時間 ○４平時應固定好家中的大型家具以防止搖晃時傾倒。
 4.(   )如果在實驗中想讓電磁鐵的磁力變得更強，可以使用下列哪一種方法？
 ○１減少漆包線圈的纏繞圈數 ○２將原本串聯的電池改為並聯 ○３增加串聯的電池數量 ○４將內部的鐵釘替換成塑膠棒。
 5.(   )下列哪一種自然現象屬於「風化作用」的範疇？
 ○１颱風帶來的豪雨將山坡土石沖刷下來 ○２河流中游的鵝卵石逐漸沉積在河床底部 ○３岩石因長期熱脹冷縮而產生網狀裂縫 ○４冰川向下移動時將地表切割出U型谷。
 6.(   )使用輪軸這項簡單機械時，如果將力氣施加在「輪」上，會產生什麼樣的效果？
 ○１省力但費時 ○２費力但省時 ○３既省力又省時 ○４既費力又費時。
 7.(   )臺灣夏季最常遇到，且容易造成嚴重農業損失與土石流的氣象災害是哪一種？
 ○１寒害 ○２颱風 ○３沙塵暴 ○４龍捲風。
 8.(   )礦物在日常生活中應用非常廣泛，下列生活物品與其原料的配對何者錯誤？
 ○１石英：製造玻璃 ○２石墨：製作鉛筆芯 ○３滑石：製作嬰兒爽身粉 ○４硫磺：作為建築房屋用的堅硬石材。
 9.(   )指北針的指針平放靜止時會指向北方，主要是因為受到下列何者的影響？
 ○１太陽的引力 ○２地球的重力 ○３地球的磁場 ○４月球的引力。
 10.(  )關於維持生態多樣性與環境保護，下列哪一項人類行為是不適當的？
 ○１多搭乘大眾運輸工具以減少碳排放 ○２將工廠未經處理的廢水直接排入河川 ○３減少使用一次性塑膠製品 ○４確實做好日常的垃圾分類與資源回收。
 
三、 綜合題（每格2 分，共20 分） 
 1. 槓桿原理在生活中有許多應用，請根據以下物品的使用情境，判斷它們屬於哪一種槓桿，並填入正確的代號：
 A. 省力槓桿 (施力臂 > 抗力臂)
 B. 費力槓桿 (施力臂 < 抗力臂)
 C. 等臂槓桿 (施力臂 = 抗力臂)
 (    ) (1) 升旗時使用的定滑輪。
 (    ) (2) 拔釘器拔除木板上的鐵釘。
 (    ) (3) 用鑷子夾取微小的實驗樣本。
 (    ) (4) 坐在公園蹺蹺板兩端重量相同的小朋友。
 
 2. 天氣預報中常出現不同的氣象圖示，請根據描述填寫對應的天氣系統名稱 (冷鋒、暖鋒、颱風)：
 (    ) (1) 在天氣圖上標示為藍色三角形連線，經過時氣溫會下降。
 (    ) (2) 在衛星雲圖上呈現一個巨大的逆時針螺旋狀雲系，中心可能有清晰的眼。
 (    ) (3) 在天氣圖上標示為紅色半圓形連線，經過時氣溫會回升。
 
四、 簡答題 （每個答案4 分，共20 分） 
 1. 請寫出兩項電磁鐵與一般永久磁鐵「不同」的特性。
 答：
 
 2. 河流的流水作用會改變地表的地貌，請依序寫出河流在上游、中游與下游的主要流水作用名稱。
 答：
 
五、 閱讀素養 （每題4 分，共20 分） 
閱讀下面短文後，回答問題。
 
  面對全球暖化與極端氣候日益嚴峻的考驗，世界各國正積極尋求減少溫室氣體排放的有效策略。為了取代傳統燃燒化石燃料所造成的大量碳排放，發展如太陽能、風能與水力發電等乾淨且可再生的替代能源，已成為當今國際社會共同努力的永續目標。
  臺灣身處四面環海的特殊地理位置，擁有得天獨厚的豐沛風力資源，特別是在台灣海峽的西部沿海一帶，政府與民間企業近年來大力推動大規模的離岸風力發電廠建置工程。透過在海面上架設巨大的風力發電機組，將強勁的海風轉換為穩定的電力來源。
  然而，在積極推動綠能產業發展與能源轉型的同時，如何兼顧當地海洋生態環境的保育、鳥類遷徙路線的迴避，以及沿海漁民傳統捕撈生計的維持，則是政府決策單位與社會大眾必須透過多方溝通、審慎評估與共同面對的重大挑戰。
 
 (   )(１)根據上文內容，下列哪一種能源不屬於文中倡導的「可再生替代能源」？
 ○１太陽能 ○２天然氣 ○３風能 ○４水力發電。
 (   )(２)臺灣西部沿海適合發展離岸風力發電，主要是因為具備了下列哪一項優勢？
 ○１擁有廣大的無人平原 ○２具備強勁且豐沛的海風資源 ○３海底蘊藏豐富的化石燃料 ○４當地人口密度極高。
 (   )(３)根據文章最後一段的敘述，推動離岸風電建設時「不需要」考量下列哪一個問題？
 ○１工廠排放廢氣的處理 ○２海洋生態環境的保育 ○３鳥類遷徙路線的迴避 ○４漁民傳統捕撈生計的維持。
 
 4. 請從文章中找出一項臺灣目前正在發展的綠色能源名稱。
 答：
 
 5. 請簡要說明，為什麼世界各國要發展替代能源來取代傳統的化石燃料？
 答：
"""

MDD_NORM_DATA = [
    {"學制": "國小", "科目": "國語", "年級": "1年級", "區域": "六都", "平均MDD": 3.792, "標準差": 1.009},
    {"學制": "國小", "科目": "國語", "年級": "1年級", "區域": "非六都", "平均MDD": 3.745, "標準差": 1.114},
    {"學制": "國小", "科目": "國語", "年級": "2年級", "區域": "六都", "平均MDD": 3.270, "標準差": 1.135},
    {"學制": "國小", "科目": "國語", "年級": "2年級", "區域": "非六都", "平均MDD": 3.537, "標準差": 0.980},
    {"學制": "國小", "科目": "國語", "年級": "3年級", "區域": "六都", "平均MDD": 3.489, "標準差": 0.933},
    {"學制": "國小", "科目": "國語", "年級": "3年級", "區域": "非六都", "平均MDD": 3.287, "標準差": 0.992},
    {"學制": "國小", "科目": "國語", "年級": "4年級", "區域": "六都", "平均MDD": 3.328, "標準差": 0.919},
    {"學制": "國小", "科目": "國語", "年級": "4年級", "區域": "非六都", "平均MDD": 3.329, "標準差": 0.919},
    {"學制": "國小", "科目": "國語", "年級": "5年級", "區域": "六都", "平均MDD": 3.499, "標準差": 0.767},
    {"學制": "國小", "科目": "國語", "年級": "5年級", "區域": "非六都", "平均MDD": 3.492, "標準差": 0.763},
    {"學制": "國小", "科目": "國語", "年級": "6年級", "區域": "六都", "平均MDD": 3.578, "標準差": 0.838},
    {"學制": "國小", "科目": "國語", "年級": "6年級", "區域": "非六都", "平均MDD": 3.356, "標準差": 0.874},
    {"學制": "國小", "科目": "自然", "年級": "3年級", "區域": "六都", "平均MDD": 3.427, "標準差": 0.815},
    {"學制": "國小", "科目": "自然", "年級": "3年級", "區域": "非六都", "平均MDD": 3.297, "標準差": 0.757},
    {"學制": "國小", "科目": "自然", "年級": "4年級", "區域": "六都", "平均MDD": 3.565, "標準差": 0.695},
    {"學制": "國小", "科目": "自然", "年級": "4年級", "區域": "非六都", "平均MDD": 3.372, "標準差": 0.769},
    {"學制": "國小", "科目": "自然", "年級": "5年級", "區域": "六都", "平均MDD": 3.673, "標準差": 0.722},
    {"學制": "國小", "科目": "自然", "年級": "5年級", "區域": "非六都", "平均MDD": 3.725, "標準差": 0.742},
    {"學制": "國小", "科目": "自然", "年級": "6年級", "區域": "六都", "平均MDD": 3.574, "標準差": 0.657},
    {"學制": "國小", "科目": "自然", "年級": "6年級", "區域": "非六都", "平均MDD": 3.365, "標準差": 0.708},
    {"學制": "國小", "科目": "社會", "年級": "3年級", "區域": "六都", "平均MDD": 3.444, "標準差": 0.676},
    {"學制": "國小", "科目": "社會", "年級": "3年級", "區域": "非六都", "平均MDD": 3.315, "標準差": 0.728},
    {"學制": "國小", "科目": "社會", "年級": "4年級", "區域": "六都", "平均MDD": 3.644, "標準差": 0.714},
    {"學制": "國小", "科目": "社會", "年級": "4年級", "區域": "非六都", "平均MDD": 3.509, "標準差": 0.793},
    {"學制": "國小", "科目": "社會", "年級": "5年級", "區域": "六都", "平均MDD": 3.653, "標準差": 0.768},
    {"學制": "國小", "科目": "社會", "年級": "5年級", "區域": "非六都", "平均MDD": 3.478, "標準差": 0.729},
    {"學制": "國小", "科目": "社會", "年級": "6年級", "區域": "六都", "平均MDD": 3.755, "標準差": 0.735},
    {"學制": "國小", "科目": "社會", "年級": "6年級", "區域": "非六都", "平均MDD": 3.494, "標準差": 0.760},
    {"學制": "國中", "科目": "國文", "年級": "7年級", "區域": "六都", "平均MDD": 4.207, "標準差": 0.615},
    {"學制": "國中", "科目": "國文", "年級": "7年級", "區域": "非六都", "平均MDD": 4.169, "標準差": 0.858},
    {"學制": "國中", "科目": "國文", "年級": "8年級", "區域": "六都", "平均MDD": 4.193, "標準差": 0.596},
    {"學制": "國中", "科目": "國文", "年級": "8年級", "區域": "非六都", "平均MDD": 4.261, "標準差": 0.642},
    {"學制": "國中", "科目": "國文", "年級": "9年級", "區域": "六都", "平均MDD": 4.247, "標準差": 0.586},
    {"學制": "國中", "科目": "國文", "年級": "9年級", "區域": "非六都", "平均MDD": 4.308, "標準差": 0.699},
    {"學制": "國中", "科目": "社會", "年級": "7年級", "區域": "六都", "平均MDD": 4.301, "標準差": 0.654},
    {"學制": "國中", "科目": "社會", "年級": "7年級", "區域": "非六都", "平均MDD": 4.109, "標準差": 0.629},
    {"學制": "國中", "科目": "社會", "年級": "8年級", "區域": "六都", "平均MDD": 4.272, "標準差": 0.489},
    {"學制": "國中", "科目": "社會", "年級": "8年級", "區域": "非六都", "平均MDD": 4.172, "標準差": 0.661},
    {"學制": "國中", "科目": "社會", "年級": "9年級", "區域": "六都", "平均MDD": 4.333, "標準差": 0.508},
    {"學制": "國中", "科目": "社會", "年級": "9年級", "區域": "非六都", "平均MDD": 4.415, "標準差": 0.831},
    {"學制": "高中", "科目": "國文", "年級": "10年級", "區域": "六都", "平均MDD": 4.195, "標準差": 0.603},
    {"學制": "高中", "科目": "國文", "年級": "10年級", "區域": "非六都", "平均MDD": 3.864, "標準差": 0.971},
    {"學制": "高中", "科目": "國文", "年級": "11年級", "區域": "六都", "平均MDD": 4.308, "標準差": 0.517},
    {"學制": "高中", "科目": "國文", "年級": "11年級", "區域": "非六都", "平均MDD": 3.817, "標準差": 0.861},
    {"學制": "高中", "科目": "國文", "年級": "12年級", "區域": "六都", "平均MDD": 4.169, "標準差": 0.640},
    {"學制": "高中", "科目": "國文", "年級": "12年級", "區域": "非六都", "平均MDD": 3.968, "標準差": 0.801}
]
df_mdd_norm = pd.DataFrame(MDD_NORM_DATA)

# ==========================================
# 1. 頁面設定
# ==========================================
st.set_page_config(
    page_title="AI 華語文句法難度自動檢測系統 v1.0（雛形）",
    page_icon="📚",
    layout="wide"
)

# ==========================================
# 2. 載入模型 (不在側邊欄顯示狀態)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_nlp():
    try:
        return spacy.load("zh_core_web_sm")
    except OSError:
        st.error("❌ 找不到 spaCy 中文模型！請確保以 python -m spacy download zh_core_web_sm 安裝。")
        st.stop()

@st.cache_resource(show_spinner=False)
def load_difficulty_model():
    model_path = "mdd_baseline_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

nlp = load_nlp()
model = load_difficulty_model()

# ==========================================
# 3. 試題清洗與拆分引擎
# ==========================================
def sanitize_exam_paper(raw_text: str, min_length: int = 14) -> Tuple[List[str], List[str]]:
    filtered_out = []
    
    first_question_match = re.search(
        r'(^\s*(?:[一二三四五六七八九十壹貳參肆伍]|\d+)\s*[\.、．\)])|(^\s*一\s*[\u4e00-\u9fa5]+[：:])', 
        raw_text, 
        flags=re.MULTILINE
    )
    
    if first_question_match:
        header_text = raw_text[:first_question_match.start()].strip()
        cleaned_body = raw_text[first_question_match.start():]
        if header_text:
            clean_header_log = header_text.replace('\n', ' ')
            filtered_out.append(f"[試卷表頭區塊已切除] {clean_header_log}")
    else:
        cleaned_body = raw_text

    cleaned = re.sub(r'(?<=.)\s*(○[0-9１-９]|[①-⑨]|\([A-Da-dＡ-Ｄａ-ｄ]\)|[A-Da-dＡ-Ｄａ-ｄ][\.、])', r'\n\1', cleaned_body)
    cleaned = re.sub(r'[(（][^()（）]*每[題字格分].*?[)）]', '', cleaned)
    cleaned = re.sub(r'(?:班級|學號|座號|姓名|分數|得分|閱卷老師|家長簽章)\s*[:：_＿\s].*', '', cleaned)
    cleaned = re.sub(r'(?:市立|縣立|國中|高中|國民小學|學年度|評量試卷|期中|期末).*', '', cleaned)
    cleaned = re.sub(r'[一二三四五六七八九十]+\s*[\u4e00-\u9fa5]+[：:]', '', cleaned)
    cleaned = re.sub(r'[↓｜|]', '', cleaned)
    cleaned = re.sub(r'\([ 0-9A-Za-z\s]*\)|（[ 0-9A-Za-z\s]*）', '', cleaned)
    
    raw_sentences = re.split(r'[\n。！？!?]', cleaned)
    valid_sentences = []
    
    HEADER_KEYWORDS = [
        "學年度", "期末", "期中", "定期評量", "評量試卷", "綜合學科",
        "班級", "座號", "姓名", "得分", "配分", "總分", "國民中學", "國民小學"
    ]
    
    for s in raw_sentences:
        s_strip = s.strip()
        s_strip = re.sub(r'^(○[0-9１-９]|[①-⑨]|\([A-Da-dＡ-Ｄａ-ｄ]\)|[A-Da-dＡ-Ｄａ-ｄ][\.、]|\d+\s*[\.、．])\s*', '', s_strip)
        
        if not s_strip:
            continue
            
        if any(kw in s_strip for kw in HEADER_KEYWORDS):
            filtered_out.append(f"[表頭殘餘資訊] {s_strip}")
            continue
            
        if len(s_strip) < min_length:
            filtered_out.append(f"[過短或獨立選項] {s_strip}")
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
    
    max_dd = 0
    if valid_tokens:
        token_to_valid_idx = {t.i: idx for idx, t in enumerate(valid_tokens)}
        dep_distances = []
        
        for t in valid_tokens:
            if t.head != t and t.head.i in token_to_valid_idx:
                dist = abs(token_to_valid_idx[t.i] - token_to_valid_idx[t.head.i])
                dep_distances.append(dist)
                if dist > max_dd:
                    max_dd = dist
                    
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
        "max_dd": max_dd,
        "clause_types": analyze_clause_types(doc),
        "vocab_depth": calculate_vocab_depth(doc, term_set)
    }

def predict_grade(features: Dict[str, Any], ml_model: Optional[Any]) -> Tuple[str, float]:
    score = 1.5
    
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
                score = float(raw_pred) * 0.85  
        except Exception:
            pass

    char_len = features["char_count"]
    if char_len <= 15: score -= 1.0
    elif char_len <= 25: score -= 0.5
    elif 26 <= char_len <= 45: score += 0.5
    elif 46 <= char_len <= 70: score += 1.0     
    elif 71 <= char_len <= 100: score += 1.5    
    elif char_len > 100: score += 2.5            
    
    mdd = features["mdd"]
    if mdd < 2.0: score -= 1.0                  
    elif 2.0 <= mdd < 2.8: score -= 0.5
    elif 3.5 <= mdd < 4.2: score += 0.5          
    elif 4.2 <= mdd < 5.0: score += 1.2          
    elif mdd >= 5.0: score += 2.0                

    noun_r = features["noun_ratio"]
    if noun_r < 0.20: score -= 0.5
    elif 0.35 <= noun_r < 0.45: score += 0.5
    elif noun_r >= 0.45: score += 1.2            

    complex_clauses = ["進階論述句", "目的複句", "選擇複句", "遞進複句", "推斷複句", "假轉複句", "取捨複句"]
    if any(c in features["clause_types"] for c in complex_clauses):
        score += 1.0                              

    v_depth = features["vocab_depth"]
    if v_depth == 1: score += 0.5
    elif v_depth == 2: score += 1.0
    elif v_depth >= 3: score += 2.0              
    
    if v_depth >= 5 and score < 8.5:
        score = max(score, 8.5)
    elif v_depth >= 3 and score < 6.5:
        score = max(score, 6.5)
    elif v_depth >= 2 and score < 4.5:
        score = max(score, 4.5)

    if char_len < 20 and v_depth == 0 and "簡單句" in features["clause_types"]:
        score = min(score, 2.5)

    if score >= 10.0: grade_str = "10-12 年級 (高中以上)"   
    elif score >= 7.5: grade_str = "7-9 年級 (國中)"          
    elif score >= 5.0: grade_str = "5-6 年級 (國小)"
    elif score >= 3.0: grade_str = "3-4 年級 (國小)"
    else: grade_str = "1-2 年級 (國小低年級)"
    
    return grade_str, score

def map_score_to_grade_str(avg_score: float) -> str:
    if avg_score >= 10.0: return "10-12 年級 (高中以上)"
    elif avg_score >= 7.5: return "7-9 年級 (國中)"
    elif avg_score >= 5.0: return "5-6 年級 (國小)"
    elif avg_score >= 3.0: return "3-4 年級 (國小)"
    else: return "1-2 年級 (國小)"

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
            "MDD數值": round(feat["mdd"], 2),
            "最大依存距離": feat["max_dd"]
        })
        progress_bar.progress((i + 1) / total)
        
    progress_bar.empty()
    return pd.DataFrame(results)

# ==========================================
# 4.5 AI 改寫建議與降幅渲染共用模組
# ==========================================
def render_ai_suggestion_ui(original_text: str, old_mdd: float, old_max_dd: int, nlp_model, term_set, unique_key: str):
    if not client:
        st.info("💡 請在 Streamlit Secrets 設定 `OPENAI_API_KEY` 即可解鎖 AI 智慧拆句改寫建議！")
        return
        
    ai_state_key = f"ai_result_{unique_key}"
    
    if ai_state_key not in st.session_state:
        if st.button("✨ 點擊獲取 AI 智慧改寫建議", key=unique_key):
            with st.spinner("🤖 AI 正在進行智慧拆句分析..."):
                suggestion = get_smart_split_suggestion(original_text, round(old_mdd, 2))
                st.session_state[ai_state_key] = suggestion
            
    if ai_state_key in st.session_state:
        suggestion = st.session_state[ai_state_key]
        st.info(suggestion)
        
        match = re.search(r'【改寫建議】[：:]\s*(.*)', suggestion, re.DOTALL)
        if match:
            rewritten_text = match.group(1).replace('**', '').replace('*', '').replace('`', '').strip()
            if rewritten_text:
                new_doc = nlp_model(rewritten_text)
                new_feat = extract_features_from_doc(new_doc, term_set)
                
                st.markdown("##### 📉 改寫後難度降幅評估")
                c1, c2 = st.columns(2)
                
                mdd_diff = new_feat['mdd'] - old_mdd
                max_dd_diff = new_feat['max_dd'] - old_max_dd
                
                c1.metric("✨ 改寫後 MDD", f"{new_feat['mdd']:.2f}", f"{mdd_diff:.2f}", delta_color="inverse")
                c2.metric("✨ 改寫後最大距離", f"{new_feat['max_dd']}", f"{max_dd_diff}", delta_color="inverse")

# ==========================================
# 5. 視覺化繪圖函數 
# ==========================================
def render_single_sentence_charts(features: Dict[str, Any], raw_score: float):
    st.markdown("### 📊 難度特徵分析儀表板")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = min(12.0, max(1.0, raw_score)),
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "難度數值落點 (1-12年級)", 'font': {'size': 12}},
            gauge = {
                'axis': {'range': [1, 12], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#2A648E"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [1, 3], 'color': '#E8F5E9'},   
                    {'range': [3, 5], 'color': '#C8E6C9'},   
                    {'range': [5, 7], 'color': '#FFF9C4'},   
                    {'range': [7, 9.5], 'color': '#FFE0B2'}, 
                    {'range': [9.5, 12], 'color': '#FFCDD2'} 
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
            title={'text': "特徵強度多維雷達圖", 'font': {'size': 12}},
            margin=dict(l=40, r=40, t=40, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

def render_overall_summary(df: pd.DataFrame, norm_mean: Optional[float], norm_std: Optional[float]) -> Tuple[pd.DataFrame, float, int, float]:
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
    
    st.markdown("### 🌟 整體評估總覽")
    st.caption("💡 **評估加權機制**：考卷難度採 **前 50% 最具鑑別度的核心語句/長文** 進行加權計算（已過濾無意義結構與指示雜訊）。")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 綜合文本難度 (含字彙與長度)", overall_grade_str)
    c2.metric("📏 採樣有效字數", f"{total_chars} 字")
    
    if norm_mean is not None and norm_std is not None:
        mdd_diff = avg_mdd - norm_mean
        z_score = mdd_diff / norm_std
        
        if z_score > 0.5:
            delta_color = "inverse"
            difficulty_label = "偏難"
        elif z_score < -0.5:
            delta_color = "normal"
            difficulty_label = "偏易"
        else:
            delta_color = "off"
            difficulty_label = "適中"
            
        c3.metric(
            label="🧠 句法結構負擔 (MDD)", 
            value=f"{avg_mdd:.2f}", 
            delta=f"與 {ref_region}{ref_grade} 基準比: {difficulty_label} ({mdd_diff:+.2f})",
            delta_color=delta_color,
            help=f"基準平均: {norm_mean:.2f}, 標準差: {norm_std:.2f}"
        )
    else:
        c3.metric("🧠 句法結構負擔 (MDD)", f"{avg_mdd:.2f}", help="目前選擇的科目或年級無基準資料")
        
    st.divider()
    
    display_df = df.drop(columns=["分數_hidden"], errors='ignore')
    return display_df, overall_score, total_chars, avg_mdd

def render_statistics_charts(df: pd.DataFrame):
    st.markdown("### 📊 難度特徵分析儀表板")
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
    st.markdown("### 🎯 科目與參照基準設定")
    subject = st.selectbox("學科", ["全部學科", "國語文", "數學", "社會", "自然"])
    
    ref_school = st.selectbox("學制", ["國小", "國中", "高中"])
    
    if ref_school == "國小":
        grade_options = [f"{i}年級" for i in range(1, 7)]
    elif ref_school == "國中":
        grade_options = [f"{i}年級" for i in range(7, 10)]
    else:
        grade_options = [f"{i}年級" for i in range(10, 13)]
        
    ref_grade = st.selectbox("年級", grade_options)
    ref_region = st.selectbox("區域", ["六都", "非六都"])
    
    st.divider()
    
    st.markdown("### 👁 介面顯示設定")
    show_table = st.checkbox("顯示資料明細表", value=True)
    show_charts = st.checkbox("顯示視覺化圖表", value=True)
    
    if subject == "全部學科":
        current_term_set = ALL_SUBJECT_TERMS
    else:
        current_term_set = SUBJECT_TERMS.get(subject, set())

mapped_subject = subject
if subject in ["國語文", "全部學科"]:
    mapped_subject = "國語" if ref_school == "國小" else "國文"

norm_row = df_mdd_norm[
    (df_mdd_norm['學制'] == ref_school) & 
    (df_mdd_norm['科目'] == mapped_subject) & 
    (df_mdd_norm['年級'] == ref_grade) & 
    (df_mdd_norm['區域'] == ref_region)
]

if not norm_row.empty:
    norm_mean = norm_row.iloc[0]['平均MDD']
    norm_std = norm_row.iloc[0]['標準差']
    norm_text = f"基準: {norm_mean:.2f} (±{norm_std:.2f})"
else:
    norm_mean, norm_std = None, None
    norm_text = "無此科目/年級之基準資料 (如數學)"

# ==========================================
# 7. 置頂固定標題列設計
# ==========================================
st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 2rem !important; }}
    div[data-testid="stVerticalBlock"] > div:has(.sticky-header),
    div.element-container:has(.sticky-header) {{
        position: sticky;
        top: 2.875rem; 
        background-color: var(--background-color); 
        z-index: 999; 
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--secondary-background-color); 
    }}
    .sticky-header h1 {{ margin: 0; padding-bottom: 0.2rem; font-size: 2.25rem; font-weight: 700; }}
    .sticky-header p {{ margin: 0; font-size: 1rem; color: var(--text-color); opacity: 0.8; }}
    
    [data-testid="stMetricValue"] > div {{
        font-size: 1.35rem !important; 
        line-height: 1.2 !important;
    }}
    </style>
    
    <div class="sticky-header">
        <h1>📚 AI 華語文句法難度自動檢測系統 v1.0（雛形）</h1>
        <p>目前分析學科模式：<strong>{subject}</strong> | 當前參考標準：<strong>{ref_school} {ref_grade} ({ref_region})</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs(["✍️ 單句分析", "📋 多句分析", "📄 試卷分析", "📖 系統說明"])

# ==========================================
# TAB 1: 單句檢測 (Placeholder 屬性)
# ==========================================
with tab1:
    question_text = st.text_area("題目文字", placeholder=DEFAULT_SINGLE_Q, height=130)

    if st.button("🚀 開始檢測單句", type="primary"):
        target_text = question_text.strip()
        st.session_state['t1_run'] = True
        st.session_state['t1_warning'] = not question_text.strip()
        
        # 若無輸入，自動使用預設
        if not target_text:
            target_text = DEFAULT_SINGLE_Q
            
        with st.spinner("分析中..."):
            doc = nlp(target_text)
            features = extract_features_from_doc(doc, current_term_set)
            predicted_grade_str, predicted_raw_score = predict_grade(features, model)
            
            st.session_state['t1_feat'] = features
            st.session_state['t1_grade'] = predicted_grade_str
            st.session_state['t1_score'] = predicted_raw_score
            
            if "ai_result_btn_ai_single" in st.session_state:
                del st.session_state["ai_result_btn_ai_single"]

    if st.session_state.get('t1_run', False):
        if st.session_state.get('t1_warning'):
            st.info("💡 您未輸入內容，已自動載入**預設單句**進行分析。")
            
        features = st.session_state['t1_feat']
        predicted_grade_str = st.session_state['t1_grade']
        predicted_raw_score = st.session_state['t1_score']
        
        st.divider()
        st.markdown("### 🌟 整體評估總覽")
        cols = st.columns(4)
        cols[0].metric("🎯 綜合文本難度 (含字彙與長度)", predicted_grade_str)
        cols[1].metric("📏 採樣有效字數", f"{features['char_count']} 字")
        
        if norm_mean is not None and norm_std is not None:
            mdd_diff = features['mdd'] - norm_mean
            z = mdd_diff / norm_std
            difficulty_label = "偏難" if z > 0.5 else "偏易" if z < -0.5 else "適中"
            cols[2].metric("🧠 句法結構負擔 (MDD)", f"{features['mdd']:.2f}", 
                           delta=f"與 {ref_region}{ref_grade} 基準比: {difficulty_label} ({mdd_diff:+.2f})", 
                           delta_color="inverse" if z > 0.5 else "normal" if z < -0.5 else "off")
        else:
            cols[2].metric("🧠 句法結構負擔 (MDD)", f"{features['mdd']:.2f}", help="無基準資料")
            
        cols[3].metric("🔗 複句結構", features["clause_types"])
        st.write("")
        
        mdd_threshold = norm_mean if norm_mean is not None else 3.6
        if features['mdd'] > mdd_threshold:
            st.markdown("### 🚨 高難度警示與智慧拆句建議")
            st.warning(f"**系統偵測此句 MDD ({features['mdd']:.2f}) 高於當前標準 ({mdd_threshold:.2f})，認知負荷偏高！** \n* 最大依存距離: {features['max_dd']}")
            
            render_ai_suggestion_ui(
                original_text=features['text'], 
                old_mdd=features['mdd'], 
                old_max_dd=features['max_dd'], 
                nlp_model=nlp, 
                term_set=current_term_set,
                unique_key="btn_ai_single"
            )
        else:
            st.markdown("### ✨ 句法結構檢測通過")
            st.success(f"🎉 此句的 MDD ({features['mdd']:.2f}) 低於/等於當前難度標準 ({mdd_threshold:.2f})，句法負擔適中，無須進行拆句修改！")

        if show_charts:
            render_single_sentence_charts(features, predicted_raw_score)
            
        if show_table:
            st.markdown("### 📋 特徵明細")
            st.dataframe({
                "特徵名稱": ["總詞數 (含標點)", "名詞比例", "動詞比例", "該科進階術語計數", "最大依存距離", "原始 MDD", "修正 MDD"],
                "數值": [
                    features['word_count'], f"{features['noun_ratio']:.1%}", 
                    f"{features['verb_ratio']:.1%}", f"{features['vocab_depth']} 個",
                    features['max_dd'], f"{features['base_mdd']:.2f}", f"{features['mdd']:.2f}"
                ]
            }, use_container_width=True)

# ==========================================
# TAB 2: 多句批次查詢 (修復：已加上檔案上傳邏輯)
# ==========================================
with tab2:
    batch_mode = st.radio("輸入方式：", ["📋 貼上多行文字", "📂 上傳檔案"], horizontal=True)
    
    q_list = []
    run_analysis = False
    is_using_default_batch = False
    
    # 【模式 A】貼上多行文字
    if batch_mode == "📋 貼上多行文字":
        batch_text = st.text_area("每行一題：", placeholder=DEFAULT_BATCH_Q, height=280)
        
        if st.button("⚡ 開始批次分析", type="primary"):
            target_batch_text = batch_text.strip()
            if not target_batch_text:
                target_batch_text = DEFAULT_BATCH_Q
                is_using_default_batch = True
                
            q_list = [line.strip() for line in target_batch_text.split("\n") if line.strip()]
            run_analysis = True
            
    # 【模式 B】上傳檔案
    else:
        uploaded_batch_file = st.file_uploader("📂 選擇上傳試題檔案 (支援 PDF, Word, TXT)\n※ 檔案內容請保持「一行一題」的格式", type=["pdf", "docx", "txt"])
        
        if st.button("⚡ 讀取檔案並開始分析", type="primary"):
            if uploaded_batch_file is not None:
                with st.spinner(f"正在解析檔案：{uploaded_batch_file.name} ..."):
                    if uploaded_batch_file.name.lower().endswith(".pdf"):
                        file_text = extract_text_from_pdf(uploaded_batch_file)
                    elif uploaded_batch_file.name.lower().endswith(".docx"):
                        file_text = extract_text_from_docx(uploaded_batch_file)
                    elif uploaded_batch_file.name.lower().endswith(".txt"):
                        file_text = uploaded_batch_file.getvalue().decode("utf-8")
                    
                    q_list = [line.strip() for line in file_text.split("\n") if line.strip()]
                    run_analysis = True
                    
                    if not q_list:
                        st.error("❌ 檔案中沒有找到有效的文字行！")
            else:
                st.warning("⚠️ 請先選擇要上傳的檔案。")
                
    # 進行分析與渲染 (兩種模式共用)
    if run_analysis and q_list:
        st.session_state['t2_warning'] = is_using_default_batch
        res_df = run_batch_analysis(q_list, nlp, model, current_term_set)
        st.session_state['t2_run'] = True
        st.session_state['t2_res_df'] = res_df
        
        for k in list(st.session_state.keys()):
            if k.startswith("ai_result_btn_ai_batch_"):
                del st.session_state[k]

    if st.session_state.get('t2_run', False) and 't2_res_df' in st.session_state:
        if st.session_state.get('t2_warning'):
            st.info("💡 您未輸入內容，已自動載入**預設批次考題**進行分析。")
            
        res_df = st.session_state['t2_res_df']
        st.divider()
        
        display_df, avg_score, total_chars, avg_mdd = render_overall_summary(res_df, norm_mean, norm_std)
        
        if show_charts: 
            render_statistics_charts(display_df)
        
        if show_table: 
            st.markdown("### 📋 特徵明細")
            st.dataframe(display_df, use_container_width=True)

        mdd_threshold = norm_mean if norm_mean is not None else 3.6
        high_diff_df = display_df[display_df["MDD數值"] > mdd_threshold].copy()
        
        if not high_diff_df.empty:
            st.markdown("### 🚨 考題高難度警示與 AI 智慧拆句 (Top 2)")
            st.caption(f"為避免 API 濫用並聚焦重點，系統僅針對 **高於標準 ({mdd_threshold:.2f})** 且最具鑑別度的 **前 2 句** 提供 AI 改寫建議。")
            
            top_2_hardest = high_diff_df.sort_values(by="MDD數值", ascending=False).head(2)
            
            for idx, row in top_2_hardest.iterrows():
                with st.expander(f"⚠️ 高負載試題 (MDD: {row['MDD數值']} | 最大距離: {row['最大依存距離']})：{row['題目內容'][:15]}...", expanded=True):
                    st.write(f"**原句**：{row['題目內容']}")
                    
                    render_ai_suggestion_ui(
                        original_text=row['題目內容'], 
                        old_mdd=row['MDD數值'], 
                        old_max_dd=row['最大依存距離'], 
                        nlp_model=nlp, 
                        term_set=current_term_set,
                        unique_key=f"btn_ai_batch_{idx}"
                    )
        else:
            st.markdown("### ✨ 句法結構檢測通過")
            st.success(f"🎉 本次測試的試題 MDD 皆低於/等於當前標準門檻 ({mdd_threshold:.2f})，無須進行高負載句法拆句與修改！")

# ==========================================
# TAB 3: 整份考題分析 (Placeholder 屬性)
# ==========================================
with tab3:
    st.markdown("### 🧹 考題自動雜訊過濾與深度檢測")
    st.info("💡 **全卷檢測防稀釋原理**：\n1. **智慧降噪**：自動過濾無意義短句拉低難度。\n2. **Top 50% 鑑別度加權**：採考卷中最具鑑別度的前 50% 核心語句決定整體適用年級。")
    
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        min_char_limit = st.slider("📏 採樣句數最低字數門檻", min_value=8, max_value=30, value=14, step=2)
    
    uploaded_file = st.file_uploader("📂 選擇上傳試卷檔案 (支援 PDF, Word, TXT)", type=["pdf", "docx", "txt"])
    
    raw_exam_paper = st.text_area("或者直接貼上考題文字（優先使用上方的上傳檔案）：", placeholder=DEFAULT_EXAM_PAPER, height=350)
    
    if st.button("🔍 讀取檔案/文字並開始分析", type="primary"):
        exam_input = ""
        is_using_default = False
        
        if uploaded_file is not None:
            with st.spinner(f"正在解析上傳的檔案：{uploaded_file.name} ..."):
                if uploaded_file.name.lower().endswith(".pdf"):
                    exam_input = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.name.lower().endswith(".docx"):
                    exam_input = extract_text_from_docx(uploaded_file)
                elif uploaded_file.name.lower().endswith(".txt"):
                    exam_input = uploaded_file.getvalue().decode("utf-8")
        else:
            exam_input = raw_exam_paper.strip()
            # 🌟 若沒有輸入任何文字，即採用預設考題
            if not exam_input:
                exam_input = DEFAULT_EXAM_PAPER
                is_using_default = True

        st.session_state['t3_warning'] = is_using_default
        
        with st.spinner("正在進行文本降噪、結構切割與深度特徵提取..."):
            extracted_sentences, filtered_noise = sanitize_exam_paper(exam_input, min_length=min_char_limit)
            
            if extracted_sentences:
                res_df = run_batch_analysis(extracted_sentences, nlp, model, current_term_set)
                st.session_state['t3_run'] = True
                st.session_state['t3_res_df'] = res_df
                st.session_state['t3_noise'] = filtered_noise
                st.session_state['t3_extracted'] = len(extracted_sentences)
            else:
                st.session_state['t3_run'] = True
                st.session_state['t3_res_df'] = None
                
            for k in list(st.session_state.keys()):
                if k.startswith("ai_result_btn_ai_exam_"):
                    del st.session_state[k]

    if st.session_state.get('t3_run', False):
        if st.session_state.get('t3_warning'):
            st.info("💡 目前使用的分析內容為系統內建的**預設考題範例**。您可以在上方文字框貼上自己的考卷！")
            
        if st.session_state.get('t3_res_df') is not None:
            st.success(f"✅ 成功從考題中過濾雜訊，擷取出 **{st.session_state['t3_extracted']}** 個具代表性的有效試題語句！")
            res_df = st.session_state['t3_res_df']
            
            st.divider()
            display_df, overall_score, total_chars, avg_mdd = render_overall_summary(res_df, norm_mean, norm_std)
            
            if show_charts:
                render_statistics_charts(display_df)
                
            if show_table:
                st.markdown("### 📋 特徵明細")
                st.dataframe(display_df, use_container_width=True)

            mdd_threshold = norm_mean if norm_mean is not None else 3.6
            high_diff_df = display_df[display_df["MDD數值"] > mdd_threshold].copy()
            
            if not high_diff_df.empty:
                st.markdown("### 🚨 考題高難度警示與 AI 智慧拆句 (Top 2)")
                st.caption(f"為避免 API 濫用並聚焦重點，系統僅針對 **高於標準 ({mdd_threshold:.2f})** 且最具鑑別度的 **前 2 句** 提供 AI 改寫建議。")
                
                top_2_hardest = high_diff_df.sort_values(by="MDD數值", ascending=False).head(2)
                
                for idx, row in top_2_hardest.iterrows():
                    with st.expander(f"⚠️ 高負載試題 (MDD: {row['MDD數值']} | 最大距離: {row['最大依存距離']})：{row['題目內容'][:15]}...", expanded=True):
                        st.write(f"**原句**：{row['題目內容']}")
                        
                        render_ai_suggestion_ui(
                            original_text=row['題目內容'], 
                            old_mdd=row['MDD數值'], 
                            old_max_dd=row['最大依存距離'], 
                            nlp_model=nlp, 
                            term_set=current_term_set,
                            unique_key=f"btn_ai_exam_{idx}"
                        )
            else:
                st.markdown("### ✨ 句法結構檢測通過")
                st.success(f"🎉 全卷核心試題的 MDD 皆低於/等於當前標準門檻 ({mdd_threshold:.2f})，符合該年級認知發展，無須修改！")
            
            with st.expander("👁️ 檢視被自動過濾的考題雜訊與指示句（點擊展開）"):
                st.write(f"共過濾掉 **{len(st.session_state['t3_noise'])}** 個雜訊片段：")
                st.json(st.session_state['t3_noise'][:30])
            
            st.download_button("📥 下載整份考題分析報告 CSV", display_df.to_csv(index=False).encode("utf-8-sig"), "考題分析報告.csv", "text/csv")
        else:
            st.error("❌ 找不到符合字數門檻的有效句子，請嘗試降低採樣字數門檻，或確認上傳的檔案包含可讀取的純文字格式！")

# ==========================================
# TAB 4: 系統說明與文獻引用
# ==========================================
with tab4:
    st.markdown("## 📖 系統原理與使用說明")
    st.write("本系統專為教育工作者與學術研究人員設計，透過自然語言處理 (NLP) 技術，客觀評估華語文教材與試題的閱讀認知負荷。")
    
    st.divider()
    
    st.markdown("### 🧠 什麼是「平均依存距離」(MDD)?")
    st.write("平均依存距離（Mean Dependency Distance, MDD）是計量語言學中，用以衡量大腦處理句子結構時「認知負荷（Cognitive Load）」的核心指標。")
    st.write("在閱讀時，大腦必須將相關聯的詞彙連結起來才能理解句意。**如果兩個相關詞彙在句子中隔得越遠，大腦需要將前一個詞暫存在工作記憶（Working Memory）中的時間就越長，進而造成閱讀困難。**")
    
    st.markdown("#### 📐 MDD 難度對比範例")
    st.write("為什麼我們不能只看「句子字數長短」，而必須評估「依存距離」？請看以下兩個**字數完全相同**的句子對比：")
    
    st.markdown("""
    **🔴 句型 A：修飾語前置（結構糾結，閱讀吃力）**
    > 警方順利逮捕了那名涉嫌昨晚在銀行搶劫的嫌犯。
    
    * **大腦的痛點**：讀到動詞「逮捕」後，大腦必須把這個動作懸空記在工作記憶裡，跨越中間一大串修飾語，直到句末才終於讀到受詞「嫌犯」。
    * **系統診斷**：**MDD = 3.67**，最大依存距離 (**Max DD) = 10**。
    
    ---
    
    **🟢 句型 B：拆解為短句（結構線性，輕鬆易讀）**
    > 警方順利逮捕了那名嫌犯，他涉嫌昨晚在銀行搶劫。
    
    * **負荷的釋放**：動詞「逮捕」後面緊接著受詞「嫌犯」，語意瞬間完整，大腦暫存記憶得以清空釋放，接著才用另一個短句去吸收補充資訊。
    * **系統診斷**：**MDD = 2.33**，最大依存距離 (**Max DD) = 6**。
    """)
    
    st.info("💡 **結論**：即便字數完全相同，只要出題者習慣將「長串修飾語」夾在核心動詞與名詞之間，就會創造出極高的 MDD 與認知負擔。本系統能協助教師精準揪出這類「隱形的高負載試題」。")
    
    st.divider()
    
    st.markdown("### 🛠️ 系統使用方式")
    st.markdown("""
    1. **設定對標基準 (側邊欄)**：
        請先在左側邊欄設定您考卷的「目標對象」（如：國小 6年級 六都）。系統會自動載入該階段學生的平均 MDD 作為難度評估標準。
    2. **選擇分析模式**：
        * **✍️ 單句分析**：適合針對特定難懂的課文長句進行深度診斷。
        * **📋 多句分析**：支援**檔案上傳**與**多行貼上**。
        * **📄 試卷分析**：支援 **PDF、Word 檔案上傳**，或直接貼上整份期中/期末考卷，系統會自動啟動「智慧降噪」，濾除題號、配分、指示句（如「請選出正確答案」），專注評估核心試題的閱讀難度。
    3. **AI 智慧改寫建議**：
        若系統偵測到某題的 MDD 高於您設定的年級基準，系統會提供AI驅動的智慧拆句與改寫建議，協助教師在不改變題意的前提下降低閱讀門檻。
    """)
    
    st.divider()
    
    st.markdown("### 📑 本系統相關研究")   
    st.code("""周一銘 (2026)。〈初探華語文評量材料句法複雜度與自動檢測系統建置〉。2026東臺灣華語文教學論壇暨國際學術研討會。國立臺東大學。""", language="text")
    
    st.caption("※ 本系統之 MDD 基準數據取自台灣學生各年級與地區的實際測驗文本語料庫統計。")
    st.caption("※ 本系統目前仍為前導版本，仍持續測試調整中，相關數值和內容僅供學術討論與研發參考。")
    
    st.divider()
    
    st.markdown("### 📝 意見與回饋")
    st.write("歡迎填寫系統使用意見調查，您的回饋將幫助我們持續優化系統！")
    st.link_button("填寫意見調查表單", "https://forms.gle/ekaCib6ddXQWAhvaA")
