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

# ==========================================
# 0.1 OpenAI API 初始化設定
# ==========================================
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=openai_api_key) if openai_api_key else None

def get_smart_split_suggestion(sentence: str, bottleneck: str, mdd: float) -> str:
    """呼叫 ChatGPT API 進行智慧拆句建議"""
    if not client:
        return "⚠️ 未偵測到 OPENAI_API_KEY，請先至 Streamlit Secrets 設定金鑰。"
        
    prompt = f"""
    你是一位專業的華語文教材編輯與句法學專家。
    請根據以下高難度句子（MDD: {mdd}）進行「智慧拆句與改寫建議」。
    系統偵測到的句法瓶頸（最長依存距離弧）：{bottleneck}

    請依據以下原則給出具體建議：
    1. 依據標點、轉折/因果關聯詞或連動動詞節點進行斷句。
    2. 針對瓶頸處，將長情境鋪陳或過長的修飾語獨立成句。
    3. 改寫後的句子應保持原意，但大幅降低學生的認知負荷（MDD）。

    原句：
    {sentence}

    請直接輸出以下格式（無須多餘問候）：
    **【瓶頸分析】**：簡短一句話說明為何這句負擔大。
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

DEFAULT_EXAM_PAPER = """1 OO市OO國小OO學年度上學期六年級自然科學領域期末評量  
六年   班   號  姓名：       
一、 是非題 （每題2 分，共20 分） 
 1.(   )「自製簡易小馬達」實驗中，線圈兩端的漆包線的漆面要完全刮除乾淨，才不會接觸不良，影響實驗。
 2.(   )礦物是由一種或多種不同大小、顏色的岩石所組成。
 3.(   )河水長時間對地表進行侵蝕、搬運、堆積，使地表改變了原來的面貌。
 4.(   )地磁的 N 極在北極附近，因此可以吸引指北針的指北端。
 5.(   )地震除了會造成地表地貌改變外，也可能引發海嘯。
 6.(   )通電的線圈中間擺放木棒會比擺放鋁棒所產生的磁力更強。
 7.(   )電器產品運用電磁鐵原理運作時，都是將電流轉化為動能。
 8.(   )岩石風化後慢慢碎裂成小沙石、泥土，與動植物腐化分解的物質混合，在長時間作用下會形成化石。
 9.(   )自然界中可以改變地表的力量有風、水、地震、火山還有人類活動。
 10.(   )「自製簡易小馬達」實驗中，線圈纏繞圈數愈少，重量愈輕，線圈就一定會轉動得更快。
 
 二、 選擇題 （每題2 分，共20 分） 1.(   )自動鉛筆需要安裝筆芯才能使用。
下列哪一種岩石或礦物是筆芯的原料？
 ○１石墨 ○２石膏 ○３石英       ○４石灰。
 2.(   )小芯暑假和家人去澎湖旅遊，他們逛到古厝保存區時，發現這區的古宅牆體都是用咾咕石砌成的，它是下列哪一種生物遺骸形成的石灰岩？
 ○１恐龍 ○２珊瑚 ○３穿山甲 ○４鯨魚。
 3.(   )下列哪一種方式無法確認通電的電磁鐵線圈具有磁性？
 ○１附近的指北針產生偏轉 ○２可以吸起迴紋針 ○３通電的線圈會發熱 ○４灑鐵粉附近會產生磁力線。
 4.(   )地球內部就像是一個大磁鐵，具有磁力，稱為     ○１重力 ○２磁鐵 ○３磁極 ○４地磁。
 5.(   )下列哪一個物品沒有運用到電磁鐵的原理？
 ○１電動車 ○２鬧鐘 ○３微波爐 ○４手電筒。
 6.(   )石灰岩主要是下列哪一種礦物所組成？
 ○１石英   ○２長石   ○３方解石 ○４黑雲母。
 7.(   )使用指北針時，周圍不能出現下列何種物品，才不會影響到指針指北端指向北方呢？
 ○１玻璃杯 ○２鐵製長尾夾 ○３書本 ○４塑膠尺。
 8.(   )有一種礦物通常具有透明且良好的結晶，常被加工製成飾品，它會是 ○１石英 ○２石墨 ○３硫黃 ○４石膏 9.(   )下列哪一種人類的行為模式不是土壤保育的方法？
 ○１植樹造林 ○２過量施肥 ○３開發前的環境評估   ○４在坡地堆砌石頭。
 10.(  )下列哪一個物品使用時不會釋出電磁波？
 ○１電視 ○２平板電腦 ○３手機 ○４課本。
 
 三、 綜合題（題1 每格1 分，其餘每題2 分，共24 分） 1. 請將表格完成(每格1 分)。
  上游 中游 下游 地勢(坡度)   作用  侵蝕及搬運 作用減弱  景觀   請參考下方資訊填入表格 (陡/緩/平坦/侵蝕/搬運/堆積/大石頭/鵝卵石/泥土細沙) 2. 下圖為懸吊的長條形磁鐵，靜止時會指向地磁方向，請根據指北針的指向，標示出磁鐵的 N 極和 S 極。
  ○１（    ）極 ○２（    ）極  3. 下圖是小怡去國立科學工藝博物館中拍下的某個展區的礦物硬度表，請依照內容回答下列的問題。
 摩氏礦物硬度表 硬度 礦石代表 相對硬度 1 滑石  2 石膏 指甲可刻劃 3 方解石 大頭針可刻劃 4 螢石  5 磷灰石 小刀或玻璃可刻劃 6 正長石 鋼刀不可刻劃 7 石英 鋼刀不可刻劃 8 黃玉  9 剛玉  10 金剛石  當鑑定硬度時，如果沒有以上的摩氏硬度計，可用其他東西代替，如刀片硬度約為 5.5；銅幣約為 3.5 至 4；指甲約為 2 至 3；鋼刀或玻璃硬度為 6。
 註：摩氏硬度表的數字是沒有定量的意義，只代表硬度的等級。
  （ ）(１)從上面的礦物硬度表，可得知下列哪一種礦物或岩石的硬度最低？
 ○１滑石 ○２方解石 ○３石膏 ○４石英。
 （ ）(２)接(１)題，下列哪一種礦物或岩石比螢石堅硬？
 ○１滑石 ○２磷灰石 ○３方解石 ○４石膏。
 （ ）(３)小彬抽了一個礦石禮包，得到一顆礦石。
這顆礦石用指甲刻劃沒有痕跡、但用大頭針可以留下痕跡，最有可能的礦物是 ○１石膏 ○２滑石 ○３方解石 ○４石英。
  4.榕榕進行「觀察通電的電線對指北針的影響」實驗，將通電的電線擺在指北針上方，指針順時針偏轉，回答下面問題。
  （ ）(１)如果其他條件不變，將電線移到指北針下方，指針會 ○１順時針偏轉 ○２指向北方 ○３逆時針偏轉 ○４指向南方。
 （ ）(２)如果其他條件不變，將電流改變方向，指針會 ○１順時針偏轉 ○２逆時針偏轉 ○３指向北方         ○４指向南方。
   （ ）(３)如果其他條件不變，將電線移到指北針下方，並       將電流改變方向，指針會 ○１順時針偏轉            ○２逆時針偏轉 ○３指向北方 ○４指向南方。
      
2 四、 簡答題 （每個答案2 分，共16 分） 1.請寫出兩種磁鐵和電磁鐵不同的特性。
    2.製作電磁鐵時，想要讓電磁鐵的磁力變強可以怎麼做？
請寫出兩種方法。
    3.電磁鐵被運用在日常生活中的物品上，請舉出兩種應用電磁鐵原理製作的物品。
    4.地震來臨前，隨時做好防災準備可以保障生命安全。
請寫出兩種平時可以做到的防災準備。
     五、 閱讀素養 （每格2 分，共20 分） 閱讀下面短文後，回答問題。
 1.    永安國小六年級學生在113 年12 月舉行戶外教育，第二天到了野柳地質公園，野柳位在新北市東北角地區，突出的狹長海岬地形長期受到強烈東北季風吹拂，風化作用加上海浪侵蝕生成了大家口中的「類火星地形」，岩石種類相當豐富，甚至還包含漸層色彩，野柳為大屯山餘脈伸出海中的岬角，從金山遠眺，猶如潛入海中的巨龜，故又名為「野柳龜」。
特殊的地理景觀因波浪侵蝕、岩石風化及地殼運動等作用，造就了海蝕洞溝、燭狀石、蕈狀岩、豆腐岩、蜂窩岩、壺穴、溶蝕盤等各種奇特景觀。
女王頭、仙女鞋、燭臺石等，更是聞名國際的海蝕奇觀。
其中女王頭是典型的蕈狀岩，蕈狀岩是野柳最常見的岩石，它的上層是含鈣質的砂岩，質地遠較下層堅硬，因此會出現差別侵蝕的現象，久而久之，就形成上粗下細、狀似蕈菇的蕈狀岩。
   （ ）(１)女王頭屬於蕈狀岩，係由兩層不同的岩層所形成，我們可推測，女王頭最有可能由哪種岩石所組成？
 ○１板岩 ○２大理岩 ○３安山岩 ○４砂岩。
 （ ）(２)女王頭的頭部區域布滿大小不一的孔洞，造成這些密密麻麻的構造原因為何？
 ○１缺乏使用保養品 ○２變質作用 ○３侵蝕作用 ○４沉積作用。
   2.臺灣位於環太平洋地震帶上，地震發生的次數相當頻繁，並且經常有強烈的地震發生。
根據下圖回答問題。
   （ ）(１)震央在哪裡？
 ○１臺灣東部海域 ○２臺灣西部海域 ○３臺灣北部海域 ○４臺灣南部海域。
 （ ）(２)此地震的芮氏規模有多大？
 ○１ 7.2 級 ○２ 7.2 ○３ 4 級 ○４ 4 。
 （ ）(３)此地震搖晃程度最严重是在下列哪一個地區？
 ○１南投縣 南投市 ○２臺東縣 長濱 ○３花蓮縣 和平 ○４高雄市 永安。
 3.    基隆瑞芳的水湳洞、金瓜石、九份蘊含100 多種礦物，幾乎佔臺灣礦石種類的三分之一，有「礦山版的亞馬遜森林」之美譽。
作為臺灣金銅礦產開採的原鄉，因其成礦作用產生一系列獨特的地質地形，深深影響礦業設施與聚落的分布及擴展，而其礦床型態及礦物組合也影響礦業開採手法與臺灣礦物科學發展。
 （ ）(１)九份金瓜石是臺灣著名的礦石開採地區，在開採礦物後，下列哪一種不是辨別岩石種類方法？
            ○１顏色 ○２硬度 ○３結晶外型 ○４岩石顆粒大小。
 （ ）(２)此區域能形成許多礦物的原因，跟當地岩層中礦物質的多樣性與地底下高溫作用影響有關。
因此造成礦物富集的原因，與哪一個作用最相關？
          ○１風化作用 ○２侵蝕作用 ○３火山作用 ○４搬運作用。
 （ ）(３)該地因為有許多地熱排氣孔，是個適合發展發電的環境，相對於臺灣其他地區，何處有類似的環境？
 ○１墾丁國家公園 ○２陽明山國家公園          ○３壽山國家自然公園 ○４台江國家公園。
  4.    地球本身就是一個巨大磁鐵，產生的磁場得以讓指北針持續指向北方。
能感受地磁的動物，其體內都有奈米級的生物羅盤，可使其辨認方位，而這些磁性粒子在動物身上的位置也不盡相同，如：鴿子在上喙、海龜在頭部、蜜蜂則在其腹部的滋養細胞等。
因此，這些動物的歸巢與遷徙行為會受地磁的影響，磁場方向可提供方向訊息，而磁場的強度及磁傾角可提供位置訊息，動物依這些資訊可以在腦中形成磁場地圖，幫助自己找到回家的路或遷徙的方向。
例如：臺灣西部的紫斑蝶利用地磁與太陽的位置作為方向判斷依據，每年都會南來北往的遷徙。
 （ ）(１)動物體內奈米級的磁性粒子能感應地球的磁場，這裡的奈米指的是什麼？
 ○１長度單位 ○２人名 ○３重量單位 ○４面積單位。
 （ ）(２)根據上文，下列敘述哪一項錯誤？
 ○１鴿子的生物羅盤位在上喙 ○２地球是一個巨大的磁鐵          ○３生物體內的磁性粒子稱為生理時鐘 ○４有些動物可依據地球的磁場方向來尋找回家的方向。
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

    cleaned = re.sub(r'[(（][^()（）]*每[題字格分].*?[)）]', '', cleaned_body)
    cleaned = re.sub(r'(?:班級|學號|座號|姓名|分數|得分|閱卷老師|家長簽章)\s*[:：_＿\s].*', '', cleaned)
    cleaned = re.sub(r'(?:市立|縣立|國中|高中|國民小學|學年度|評量試卷|期中|期末).*', '', cleaned)
    cleaned = re.sub(r'[一二三四五六七八九十]+\s*[\u4e00-\u9fa5]+[：:]', '', cleaned)
    cleaned = re.sub(r'^\s*[\d\w]+\s*[\.、．]', '', cleaned, flags=re.MULTILINE)
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
        s_strip = re.sub(r'^\s*\d+\s*', '', s_strip)
        if not s_strip:
            continue
            
        if any(kw in s_strip for kw in HEADER_KEYWORDS):
            filtered_out.append(f"[表頭殘餘資訊] {s_strip}")
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
    
    max_dd = 0
    bottleneck_info = "無明顯瓶頸"
    if valid_tokens:
        token_to_valid_idx = {t.i: idx for idx, t in enumerate(valid_tokens)}
        dep_distances = []
        max_dist_tokens = None
        
        for t in valid_tokens:
            if t.head != t and t.head.i in token_to_valid_idx:
                dist = abs(token_to_valid_idx[t.i] - token_to_valid_idx[t.head.i])
                dep_distances.append(dist)
                if dist > max_dd:
                    max_dd = dist
                    max_dist_tokens = (t, t.head)
                    
        base_mdd = sum(dep_distances) / len(dep_distances) if dep_distances else 0.0
        if max_dist_tokens:
            bottleneck_info = f"「{max_dist_tokens[0].text}」指向「{max_dist_tokens[1].text}」 (距離 {max_dd})"
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
        "bottleneck": bottleneck_info,
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

@st.cache_data(show_spinner=False)
def run_batch_analysis_cached(question_list: List[str], current_term_set: set) -> pd.DataFrame:
    """為了避免重複運算，將分析引擎加入快取或使用獨立的運算函數"""
    # 這裡實作用於批次與試卷分析
    pass # 稍後我們直接在流程內使用 session_state 來取代 cache

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
            "最大依存距離": feat["max_dd"],   
            "瓶頸定位": feat["bottleneck"]    
        })
        progress_bar.progress((i + 1) / total)
        
    progress_bar.empty()
    return pd.DataFrame(results)

# ==========================================
# 4.5 AI 改寫建議與降幅渲染共用模組 (含按鈕狀態保存)
# ==========================================
def render_ai_suggestion_ui(original_text: str, bottleneck: str, old_mdd: float, old_max_dd: int, nlp_model, term_set, unique_key: str):
    """處理 AI 呼叫並將改寫結果重新算 MDD 以顯示降幅的共用 UI 模組"""
    if not client:
        st.info("💡 請在 Streamlit Secrets 設定 `OPENAI_API_KEY` 即可解鎖 AI 智慧拆句改寫建議！")
        return
        
    ai_state_key = f"ai_result_{unique_key}"
    
    # 如果還沒產生建議，顯示按鈕
    if ai_state_key not in st.session_state:
        if st.button("✨ 點擊獲取 AI 智慧改寫建議", key=unique_key):
            with st.spinner("🤖 AI 正在進行智慧拆句分析..."):
                suggestion = get_smart_split_suggestion(original_text, bottleneck, round(old_mdd, 2))
                st.session_state[ai_state_key] = suggestion
            # 按鈕被按下的當下，session state 更新完成，Streamlit 會自動重新渲染畫面
            
    # 如果 session state 裡面已經有結果，直接顯示 (不用重新算)
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
            delta=f"與 {ref_region}{ref_grade} 常模比: {difficulty_label} ({mdd_diff:+.2f})",
            delta_color=delta_color,
            help=f"常模平均: {norm_mean:.2f}, 標準差: {norm_std:.2f}"
        )
    else:
        c3.metric("🧠 句法結構負擔 (MDD)", f"{avg_mdd:.2f}", help="目前選擇的科目或年級無常模資料")
        
    st.divider()
    
    display_df = df.drop(columns=["分數_hidden"])
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
    st.markdown("### 🎯 科目與參照常模設定")
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
    norm_text = f"常模: {norm_mean:.2f} (±{norm_std:.2f})"
else:
    norm_mean, norm_std = None, None
    norm_text = "無此科目/年級之常模資料 (如數學)"

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
    
    /* 👇 就是加在這裡：強制縮小 metric 數值的字體 👇 */
    [data-testid="stMetricValue"] > div {{
        font-size: 1.35rem !important; /* 預設為 1.8rem，改為 1.35rem 左右較為精緻 */
        line-height: 1.2 !important;
    }}
    /* 👆 加到這裡結束 👆 */
    
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
# TAB 1: 單句檢測 (結合 Session State 暫存架構)
# ==========================================
with tab1:
    question_text = st.text_area("題目文字", height=130, placeholder=f"請輸入單一試題...\n\n若未輸入內容點選分析，將自動載入預設範例題：\n{DEFAULT_SINGLE_Q}")

    if st.button("🚀 開始檢測單句", type="primary"):
        target_text = question_text.strip() or DEFAULT_SINGLE_Q
        st.session_state['t1_run'] = True
        st.session_state['t1_warning'] = not question_text.strip()
        
        with st.spinner("分析中..."):
            doc = nlp(target_text)
            features = extract_features_from_doc(doc, current_term_set)
            predicted_grade_str, predicted_raw_score = predict_grade(features, model)
            
            # 將運算結果存入 session_state
            st.session_state['t1_feat'] = features
            st.session_state['t1_grade'] = predicted_grade_str
            st.session_state['t1_score'] = predicted_raw_score
            
            # 清除舊的 AI 結果
            if "ai_result_btn_ai_single" in st.session_state:
                del st.session_state["ai_result_btn_ai_single"]

    # --- 以下為渲染區塊 ---
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
                           delta=f"與 {ref_region}{ref_grade} 常模比: {difficulty_label} ({mdd_diff:+.2f})", 
                           delta_color="inverse" if z > 0.5 else "normal" if z < -0.5 else "off")
        else:
            cols[2].metric("🧠 句法結構負擔 (MDD)", f"{features['mdd']:.2f}", help="無常模資料")
            
        cols[3].metric("🔗 複句結構", features["clause_types"])
        st.write("")
        
        # --- 單句：高難度警示與 AI 建議 ---
        mdd_threshold = norm_mean if norm_mean is not None else 3.6
        if features['mdd'] > mdd_threshold:
            st.markdown("### 🚨 高難度警示與智慧拆句建議")
            st.warning(f"**系統偵測此句 MDD ({features['mdd']:.2f}) 高於當前標準 ({mdd_threshold:.2f})，認知負荷偏高！** \n* 最大依存距離: {features['max_dd']} \n* 最長瓶頸弧: {features['bottleneck']}")
            
            render_ai_suggestion_ui(
                original_text=features['text'], 
                bottleneck=features['bottleneck'], 
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
# TAB 2: 多句批次查詢 (結合 Session State 暫存架構)
# ==========================================
with tab2:
    batch_mode = st.radio("輸入方式：", ["📋 貼上多行文字", "📂 上傳檔案"], horizontal=True)
    
    if batch_mode == "📋 貼上多行文字":
        batch_text = st.text_area("每行一題：", height=280, placeholder=f"請貼上多行試題...\n\n預設範例：\n{DEFAULT_BATCH_Q}")
        if st.button("⚡ 開始批次分析", type="primary"):
            target_batch_text = batch_text.strip() or DEFAULT_BATCH_Q
            q_list = [line.strip() for line in target_batch_text.split("\n") if line.strip()]
            
            if q_list:
                res_df = run_batch_analysis(q_list, nlp, model, current_term_set)
                st.session_state['t2_run'] = True
                st.session_state['t2_res_df'] = res_df
                
                # 清除舊的 AI 結果
                for k in list(st.session_state.keys()):
                    if k.startswith("ai_result_btn_ai_batch_"):
                        del st.session_state[k]

        # --- 以下為渲染區塊 ---
        if st.session_state.get('t2_run', False) and 't2_res_df' in st.session_state:
            res_df = st.session_state['t2_res_df']
            st.divider()
            
            display_df, avg_score, total_chars, avg_mdd = render_overall_summary(res_df, norm_mean, norm_std)
            
            if show_charts: 
                render_statistics_charts(display_df)
            
            if show_table: 
                st.markdown("### 📋 特徵明細")
                st.dataframe(display_df, use_container_width=True)

            # --- 批次：抓出大於常模標準的最難 Top 2 ---
            mdd_threshold = norm_mean if norm_mean is not None else 3.6
            high_diff_df = display_df[display_df["MDD數值"] > mdd_threshold].copy()
            
            if not high_diff_df.empty:
                st.markdown("### 🚨 考題高難度警示與 AI 智慧拆句 (Top 2)")
                st.caption(f"為避免 API 濫用並聚焦重點，系統僅針對 **高於標準 ({mdd_threshold:.2f})** 且最具鑑別度的 **前 2 句** 提供 AI 改寫建議。")
                
                top_2_hardest = high_diff_df.sort_values(by="MDD數值", ascending=False).head(2)
                
                for idx, row in top_2_hardest.iterrows():
                    with st.expander(f"⚠️ 高負載試題 (MDD: {row['MDD數值']} | 最大距離: {row['最大依存距離']})：{row['題目內容'][:15]}...", expanded=True):
                        st.write(f"**原句**：{row['題目內容']}")
                        st.write(f"**瓶頸弧**：{row['瓶頸定位']}")
                        
                        render_ai_suggestion_ui(
                            original_text=row['題目內容'], 
                            bottleneck=row['瓶頸定位'], 
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
# TAB 3: 整份考題分析 (結合 Session State 暫存架構)
# ==========================================
with tab3:
    st.markdown("### 🧹 考題自動雜訊過濾與深度檢測")
    st.info("💡 **全卷檢測防稀釋原理**：\n1. **智慧降噪**：自動過濾無意義短句拉低難度。\n2. **Top 50% 鑑別度加權**：採考卷中最具鑑別度的前 50% 核心語句決定整體適用年級。")
    
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        min_char_limit = st.slider("📏 採樣句數最低字數門檻", min_value=8, max_value=30, value=14, step=2)
    
    raw_exam_paper = st.text_area("請貼上整份考題文字：", height=320, placeholder=f"請在此直接貼上完整的考題內文...\n\n若未輸入內容點選分析，將自動載入預設試卷範例：\n{DEFAULT_EXAM_PAPER}")
    
    if st.button("🔍 雜訊過濾並開始分析考題", type="primary"):
        exam_input = raw_exam_paper.strip() or DEFAULT_EXAM_PAPER
        st.session_state['t3_warning'] = not raw_exam_paper.strip()
        
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
                
            # 清除舊的 AI 結果
            for k in list(st.session_state.keys()):
                if k.startswith("ai_result_btn_ai_exam_"):
                    del st.session_state[k]

    # --- 以下為渲染區塊 ---
    if st.session_state.get('t3_run', False):
        if st.session_state.get('t3_warning'):
            st.info("💡 您未輸入考題內容，已自動載入**預設考題範例**進行降噪與深度分析。")
            
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

            # --- 試卷：抓出大於常模標準的最難 Top 2 ---
            mdd_threshold = norm_mean if norm_mean is not None else 3.6
            high_diff_df = display_df[display_df["MDD數值"] > mdd_threshold].copy()
            
            if not high_diff_df.empty:
                st.markdown("### 🚨 考題高難度警示與 AI 智慧拆句 (Top 2)")
                st.caption(f"為避免 API 濫用並聚焦重點，系統僅針對 **高於標準 ({mdd_threshold:.2f})** 且最具鑑別度的 **前 2 句** 提供 AI 改寫建議。")
                
                top_2_hardest = high_diff_df.sort_values(by="MDD數值", ascending=False).head(2)
                
                for idx, row in top_2_hardest.iterrows():
                    with st.expander(f"⚠️ 高負載試題 (MDD: {row['MDD數值']} | 最大距離: {row['最大依存距離']})：{row['題目內容'][:15]}...", expanded=True):
                        st.write(f"**原句**：{row['題目內容']}")
                        st.write(f"**瓶頸弧**：{row['瓶頸定位']}")
                        
                        render_ai_suggestion_ui(
                            original_text=row['題目內容'], 
                            bottleneck=row['瓶頸定位'], 
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
            st.error("❌ 找不到符合字數門檻的有效句子，請嘗試降低採樣字數門檻！")

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
    
    st.markdown("#### 📐 MDD 計算範例")
    st.write("以下方簡單句子 **「他 昨天 吃 了 蘋果」** 為例，核心動詞是「吃」。我們計算每個詞彙與其核心詞之間的絕對位置距離：")
    
    st.markdown("""
    | 詞彙位置 | 詞彙 | 詞性 | 與哪個詞有關連 (Head) | 距離計算 | 絕對距離 |
    | :---: | :--- | :--- | :--- | :--- | :---: |
    | 1 | 他 | 代名詞 | 吃 (3) | \| 1 - 3 \| | **2** |
    | 2 | 昨天 | 時間詞 | 吃 (3) | \| 2 - 3 \| | **1** |
    | 3 | **吃** | **動詞** | (核心 Root) | - | **0** |
    | 4 | 了 | 助詞 | 吃 (3) | \| 4 - 3 \| | **1** |
    | 5 | 蘋果 | 名詞 | 吃 (3) | \| 5 - 3 \| | **2** |
    """)
    st.info("**此句的 MDD 計算：** \n總依存距離 (2 + 1 + 1 + 2) ÷ 總相依關係數 (4) = **1.50**")
    
    st.divider()
    
    st.markdown("### 🛠️ 系統使用方式")
    st.markdown("""
    1. **設定對標常模 (側邊欄)**：
       請先在左側邊欄設定您考卷的「目標對象」（如：國小 6年級 六都）。系統會自動載入該階段學生的平均 MDD 作為難度評估標準。
    2. **選擇分析模式**：
       * **✍️ 單句分析**：適合針對特定難懂的課文長句進行深度診斷。
       * **📋 多句分析**：可貼上多行單獨的句子，可進行批次分析。
       * **📄 試卷分析**：可直接貼上整份期中/期末考卷，系統會自動啟動「智慧降噪」，濾除題號、配分、指示句（如「請選出正確答案」），專注評估核心試題的閱讀難度。
    3. **AI 智慧改寫建議**：
       若系統偵測到某題的 MDD 高於您設定的年級常模，系統會自動抓出該句最難理解的「瓶頸弧」，並提供 **ChatGPT 驅動的智慧拆句與改寫建議**，協助教師在不改變題意的前提下降低閱讀門檻。
    """)
    
    st.divider()
    
    st.markdown("### 📑 引用方式")
    st.write("若本系統對您的教材編纂或學術研究有所助益，歡迎於參考文獻中引用本系統：")
    
    st.code("""周一銘 (2026)。初探華語文評量材料句法複雜度與自動檢測系統建置。2026東臺灣華語文教學論壇暨國際學術研討會。國立臺東大學。""", language="text")

    st.caption("※ 本系統之 MDD 常模數據取自台灣學生各年級與地區的實際測驗文本語料庫統計。")
