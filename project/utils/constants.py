from enum import Enum 
import pandas as pd 
import os

class CELL_35_CLASSES(Enum):
    Myb = (0, "原始粒细胞")
    pMy = (1, "早幼粒细胞")
    My = (2, "中性中幼粒细胞")
    mMy = (3, "中性晚幼粒细胞")
    bNe = (4, "中性杆状核粒细胞")
    sNe = (5, "中性分叶核粒细胞")
    Eom = (6, "嗜酸性中幼粒细胞")
    Eoe = (7, "嗜酸性晚幼粒细胞")
    bEo = (8, "嗜酸性杆状核粒细胞")
    sEo = (9, "嗜酸性分叶核粒细胞")
    Ba = (10, "嗜碱性粒细胞")
    pEr = (11, "原始红细胞")
    eEr = (12, "早幼红细胞")
    iEr = (13, "中幼红细胞")
    lEr = (14, "晚幼红细胞")
    eEr_me = (15, "早巨幼红细胞")
    iEr_me = (16, "中巨幼红细胞")
    lEr_me = (17, "晚巨幼红细胞")
    Lyb = (18, "原始淋巴细胞")
    pLy = (19, "幼稚淋巴细胞")
    Ly = (20, "成熟淋巴细胞")
    rLy = (21, "反应性异型淋巴细胞")
    Mob = (22, "原始单核细胞")
    pMo = (23, "幼稚单核细胞")
    Mo = (24, "单核细胞")
    Plb = (25, "原始浆细胞")
    iPl = (26, "幼稚浆细胞")
    Pl = (27, "成熟浆细胞")
    Hist = (28, "组织细胞")
    BrS = (29, "破碎细胞及杂质")
    Phag = (30, "吞噬细胞")
    Mast = (31, "肥大细胞")
    Er = (32, "成熟红细胞")
    Mito = (33, "分裂相")
    Plat = (34, "血小板")

file_path = os.path.abspath(os.path.dirname(__file__))
tolerance_csv_path = os.path.join(file_path, "resources/confusion_tolerance.csv")
print(tolerance_csv_path)
serverid_txt = os.path.join(file_path, "resources/Server_to_database.txt")

CELL_35_DICT = { c.value[0]: c.value[1] for c in CELL_35_CLASSES }
CM_TOL_DF = pd.read_csv(tolerance_csv_path, index_col=0)

SERVERID_TXT = os.path.abspath(serverid_txt)
with open(SERVERID_TXT, 'r') as f:
    server_id = f.readlines()
    server_id = [s.strip().split(',') for s in server_id]
    server_id = [(int(s[0]), int(s[1])) for s in server_id]
IDCHANGE_SERVER_TO_DATABASE = {s[0]: s[1] for s in server_id}
IDCHANGE_DATABASE_40_TO_35 = [
    -1,0,1,2,3,4,5,6,7,8,
    9,10,10,10,10,11,12,13,14,15,
    16,17,18,19,20,21,22,23,24,25,
    26,27,28,29,33,34,30,31,1,32
]

HOSPITAL_ID = {
    "044C0025": "复旦大学附属中山医院",
    "044C0027": "空军军医大学西京医院",
    "044C002E": "迪安诊断",
    "044C0028": "苏州大学附属第一医院",
    "044C0015": "陆军军医大学新桥医院全军血液病中心",
    "044C0033": "23楼演示医院5版",
    "044C0049": "武汉康圣环球",
    "044C003A": "巡回展示医院",
    "044C0016": "浙江大学医学院附属邵逸夫医院",
    "044C001D": "长海医院",
    "044C0022": "武汉中南医院",
    "044C002F": "金域医学检验中心",
    "03E80002": "浙江大学医学院附属邵逸夫医院",
    "03E80003": "浙江大学附属第一医院",
    "03E80005": "上海长海医院",
    "03E80007": "陆军军医大学第二附属医院-新桥医院",
    "03E80009": "上海长征医院",
    "03E80010": "吉林大学白求恩第一医院",
    "03E80011": "大连医科大学附属第一医院",
    "03E80012": "哈尔滨医科大学附属第二医院",
    "03E80013": "中南大学湘雅医院",
    "03E80015": "中山大学附属第一医院",
}

HOSPITAL_ABBR = {
    "044C0025": "ZhongShan",
    "044C0027": "XiJing",
    "044C002E": "DiAn",
    "044C0028": "SuYi",
    "044C0015": "XinQiaoXueYeBing",
    "044C0033": "23LZhanShi",
    "044C0049": "KangSheng",
    "044C003A": "XunHuiZhanShi",
    "044C0016": "ShaoYiFu",
    "044C001D": "ChangHai",
    "044C0022": "ZhongNan",
    "044C002F": "JinYu",
    "03E80002": "ShaoYiFu",
    "03E80003": "ZheYi",
    "03E80005": "ChangHai",
    "03E80007": "XinQiaoYiYuan",
    "03E80009": "ChangZheng",
    "03E80010": "BaiQiuEn",
    "03E80011": "DaLianYiYuan",
    "03E80012": "HaErBinYiYuan",
    "03E80013": "XiangYa",
    "03E80015": "ZhongShanYiYuan",
}

def convert_label(x):
    if x not in IDCHANGE_SERVER_TO_DATABASE: return -1
    x = IDCHANGE_SERVER_TO_DATABASE[x]
    return IDCHANGE_DATABASE_40_TO_35[x]