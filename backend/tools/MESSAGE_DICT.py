# -*- coding: utf-8 -*-
from enum import Enum


class RetCode(Enum):
    """
    返回码枚举（数值）
    """
    TASK_FINISHED = 100  # 任务已完成
    TASK_RUNNING = 101  # 任务进行中
    TASK_TIMEOUT = 102  # 任务超时
    IMAGE_ALREADY_UPLOADED = 103  # 图片已上传，请勿重复上传
    IMAGE_DAMAGED = 104  # 图片损坏，请检查上传图片是否完整
    TASK_IN_PROGRESS = 105  # 当前有任务正在进行

    API_SUCCESS = 200  # 接口调用成功
    CLIENT_ERROR = 201  # 参数错误、文件类型错误、相关条目不存在等客户端错误
    ROI_ERROR = 202  # ROI选区错误


class RetDesc(Enum):
    """
    返回码枚举（描述）
    """
    TASK_FINISHED = "Task completed"
    TASK_RUNNING = "Task in progress"
    TASK_TIMEOUT = "Task timeout"
    IMAGE_ALREADY_UPLOADED = "Image already uploaded, do not upload again"
    IMAGE_DAMAGED = "Image damaged, please check if uploaded image is complete"
    TASK_IN_PROGRESS = "Task in progress"

    API_SUCCESS = "API success"
    CLIENT_ERROR = "Client error: invalid params, file type error, or entry not found"
    ROI_ERROR = "ROI selection error"


class TaskType(Enum):
    """
    任务类型枚举
    """
    BM_WBC = 0
    PB_WBC = 0
    CF_WBC = 0
    BM_MEG = 1
    BM_RBC = 2
    PB_RBC = 2


CELL_TYPES_X40 = {
    100000: ("Unclassified_WBC", "未分类白细胞"),
    100001: ("Unclassified_MEG", "未分类巨核细胞"),
    100002: ("Unclassified_RBC", "未分类红细胞"),
    100003: ("Broken_Cell", "破碎细胞"),
    100004: ("Unclassified_PLT", "未分类血小板"),
    100005: ("RBC", "已分类红细胞"),
    100006: ("PLT", "已分类血小板"),
    100007: ("CSF", "未分类脑脊液细胞"),
}

CELL_TYPES_X100 = {
    200000: ("Myeloblast", "原始粒细胞"),
    200001: ("Promyelocyte", "早幼粒细胞"),
    200002: ("Neutrophilic_Myelocyte", "中性中幼粒细胞"),
    200003: ("Metamyelocyte", "中性晚幼粒细胞"),
    200004: ("Band_Neutrophil", "中性杆状核粒细胞"),
    200005: ("Segmented_Neutrophil", "中性分叶核粒细胞"),
    200006: ("Eosinophilic_Myelocyte", "嗜酸性中幼粒细胞"),
    200007: ("Eosinophilic_Metamyelocyte", "嗜酸性晚幼粒细胞"),
    200008: ("Band_Eosinophil", "嗜酸性杆状核粒细胞"),
    200009: ("Segmented_Eosinophil", "嗜酸性分叶核粒细胞"),
    200010: ("Basophil", "嗜碱性粒细胞"),
    200011: ("Proerythroblast", "原始红细胞"),
    200012: ("Early_Erythroblast", "早幼红细胞"),
    200013: ("Intermediate_Erythroblast", "中幼红细胞"),
    200014: ("Late_Erythroblast", "晚幼红细胞"),
    200015: ("Early_Megaloblastic_Erythroblast", "早巨幼红细胞"),
    200016: ("Intermediate_Megaloblastic_Erythroblast", "中巨幼红细胞"),
    200017: ("Late_Megaloblastic_Erythroblast", "晚巨幼红细胞"),
    200018: ("Lymphoblast", "原始淋巴细胞"),
    200019: ("Prolymphocyte", "幼稚淋巴细胞"),
    200020: ("Mature_Lymphocyte", "成熟淋巴细胞"),
    200021: ("Reactive_Lymphocyte", "反应性异型淋巴细胞"),
    200022: ("Monoblast", "原始单核细胞"),
    200023: ("Promonocyte", "幼稚单核细胞"),
    200024: ("Monocyte", "单核细胞"),
    200025: ("Plasmablast", "原始浆细胞"),
    200026: ("Immature_Plasma_Cell", "幼稚浆细胞"),
    200027: ("Mature_Plasma_Cell", "成熟浆细胞"),
    200028: ("Histiocyte", "组织细胞"),
    200029: ("Broken_Cell", "破碎细胞及杂质"),
    200030: ("Phagocyte", "吞噬细胞"),
    200031: ("Mast_Cell", "肥大细胞"),
    200032: ("Erythrocyte", "成熟红细胞"),
    200033: ("Mitotic_Cell", "分裂相"),
    200034: ("Platelet", "血小板"),
}

CELL_TYPES_MEG = {
    300000: ("Unclassified_MEG", "颗粒型巨核细胞"),
    300001: ("Unclassified_RBC", "产板型巨核细胞"),
    300002: ("Broken_Cell", "巨核细胞裸核"),
    300003: ("Broken_Cell", "幼稚型巨核细胞"),
    300004: ("Broken_Cell", "非巨核细胞"),
    300005: ("Broken_Cell", "分类不明/无法分类巨核细胞"),
}

CELL_TYPE_CSF = {
    400000: ("Neutrophil", "中性粒细胞"),
    400001: ("Monocyte", "单核细胞"),
    400002: ("Macrophage", "吞噬细胞"),
    400003: ("Basophil", "嗜碱性粒细胞"),
    400004: ("Eosinophil", "嗜酸性粒细胞"),
    400005: ("Plasma_Cell", "浆细胞"),
    400006: ("Mature_Lymphocyte", "成熟淋巴细胞"),
    400007: ("Broken_Cell", "破碎细胞"),
    400008: ("Tumor_Cell", "肿瘤细胞"),
    400009: ("Cryptococcus", "隐球菌"),
    400010: ("Leukemic_Cell", "白血病细胞"),
    400011: ("Lymphoma_Cell", "淋巴瘤细胞"),
}

CELL_TYPE_RBC = {
    500000: ("RBC", "正常成熟红细胞"),
    500003: ("ERBC", "球形红细胞"),
    500004: ("ERBC", "椭圆形红细胞"),
    500005: ("ERBC", "靶形红细胞"),
    500006: ("ERBC", "镰状红细胞"),
    500007: ("ERBC", "泪滴形红细胞"),
    500008: ("ERBC", "口形红细胞"),
    500009: ("ERBC", "裂红细胞"),
    500010: ("ERBC", "棘形红细胞"),
    500011: ("ERBC", "皱缩红细胞"),
    500012: ("ERBC", "咬痕红细胞"),
    500013: ("ERBC", "水泡红细胞"),
    500014: ("ERBC", "盔形红细胞"),
    500015: ("ERBC", "新月形红细胞"),
    500017: ("ERBC", "不规则红细胞"),
    500019: ("ERBC", "大红细胞"),
    500020: ("ERBC", "小红细胞"),
    500021: ("ERBC", "巨大红细胞"),
    500023: ("ERBC", "低色素"),
    500024: ("ERBC", "高色素"),
    500025: ("ERBC", "啫多色性红细胞"),
    500027: ("ERBC", "啫碱性点彩红细胞"),
    500028: ("ERBC", "豪-乔小体"),
    500029: ("ERBC", "Heinz小体"),
    500030: ("ERBC", "卡波环"),
    500031: ("ERBC", "Pappenheimer小体"),
    500032: ("ERBC", "寄生虫"),
    500034: ("ERBC", "缗钱状红细胞"),
    500035: ("ERBC", "凝集红细胞"),
}

CELL_TYPE_PLT = {
    600000: ("PLAT", "正常成熟血小板"),
    600001: ("EPLAT", "片状"),
    600002: ("EPLAT", "小簇"),
    600003: ("EPLAT", "大簇"),
    600004: ("EFC", "杂质"),
    600005: ("EPLAT", "畸形血小板"),
    600006: ("EPLAT", "颗粒减少血小板"),
    600007: ("EPLAT", "灰色血小板"),
    600008: ("EPLAT", "大血小板"),
    600009: ("EPLAT", "小血小板"),
    600010: ("EPLAT", "巨大血小板"),
}

# 根据细胞ID返回计数类别
# 不参与计数：Excluded from the count
# 有核细胞：Nucleated_Cell
# 巨核细胞：Megakaryocyte
def get_counting_cell_type(cell_id, smear_type="BM"):
    if cell_id in CELL_TYPES_X40:
        if cell_id == 100000:
            return "WBC"
        elif cell_id == 100001:
            return "MEG"
        return "EFC"
    elif cell_id in CELL_TYPES_X100:
        if smear_type == "BM":
            if cell_id in list(range(200000, 200028)):
                return "WBC"
            else:
                return "EFC"
        elif smear_type == "PB":
            if cell_id in list(range(200000, 200011)) + list(range(200018, 200028)):
                return "WBC"
            else:
                return "EFC"
    elif cell_id in CELL_TYPES_MEG:
        if cell_id in list(range(300000, 300004)):
            return "MEG"
        else:
            return "EFC"
    elif cell_id in CELL_TYPE_CSF:
        if cell_id == 400007:
            return 'EFC'
        return "WBC"
    elif cell_id in CELL_TYPE_RBC:
        return CELL_TYPE_RBC[cell_id][0]
    elif cell_id in CELL_TYPE_PLT:
        return CELL_TYPE_PLT[cell_id][0]
    else:
        return "EFC"

# 目前开放的DPI
dpi_list = [40, 50, 100]
# 允许上传的图片格式
allow_extensions = ['jpg', 'jpeg', 'gif', 'png']

# =============================================================================
# 模型目录（新增模型直接在此追加）
# dpi_range: (min, max) 或 None（不限制，分类模型）
#   请求 DPI 落在区间内才允许，并可缩放到 actual_dpi；否则接口返回「DPI不合适」
# output: bboxes=定位, tops=分类, scores=评分
# vram_gb: 预估显存（GB）
# camera=flir 时 DPIALL_BM_PB_WBC_classifier 会被替换为 DPIALL_FLIR_BM_PB_WBC_classifier
# =============================================================================
MODEL_TABLE = [
    {
        "name_zh": "5倍脑脊液有核细胞定位",
        "name": "DPI35000_CSF_cell_detection",
        "dpi_range": (30000, 45000),
        "actual_dpi": 35000,
        "targets": "WBC",
        "smear_types": "CSF",
        "output": "bboxes",
        "vram_gb": 1.5,
    },
    {
        "name_zh": "10倍脑脊液有核细胞定位",
        "name": "DPI71000_CSF_cell_detection",
        "dpi_range": (45001, 90000),
        "actual_dpi": 71000,
        "targets": "WBC",
        "smear_types": "CSF",
        "output": "bboxes",
        "vram_gb": 1.5,
    },
    {
        "name_zh": "低倍骨髓/外周血有核细胞定位",
        "name": "DPI147246_BM_PB_WBC_cell_detection",
        "dpi_range": (73623, 294492),
        "actual_dpi": 147246,
        "targets": "WBC",
        "smear_types": "BM/PB",
        "output": "bboxes",
        "vram_gb": 1.5,
    },
    {
        "name_zh": "低倍骨髓巨核细胞定位",
        "name": "DPI147246_BM_PB_MEG_cell_detection",
        "dpi_range": (73623, 294492),
        "actual_dpi": 147246,
        "targets": "MEG",
        "smear_types": "BM",
        "output": "bboxes",
        "vram_gb": 1.5,
    },
    {
        "name_zh": "高倍巨核细胞定位",
        "name": "DPI357378_BM_MEG_cell_detection",
        "dpi_range": (321640, 750493),
        "actual_dpi": 357378,
        "targets": "MEG",
        "smear_types": "BM",
        "output": "bboxes",
        "vram_gb": 1.5,
    },
    {
        "name_zh": "高倍有核细胞定位",
        "name": "DPI714756_BM_PB_CSF_WBC_detector",
        "dpi_range": (321640, 750493),
        "actual_dpi": 714756,
        "targets": "WBC",
        "smear_types": "BM/PB/CSF",
        "output": "bboxes",
        "vram_gb": 1.5,
    },
    {
        "name_zh": "高倍红细胞定位",
        "name": "DPI714756_BM_PB_RED_cell_detection",
        "dpi_range": (321640, 750493),
        "actual_dpi": 714756,
        "targets": "RBC",
        "smear_types": "PB",
        "output": "bboxes",
        "vram_gb": 1.5,
    },
    {
        "name_zh": "高倍血小板定位",
        "name": "DPI714756_BM_PB_PLAT_detection",
        "dpi_range": (321640, 750493),
        "actual_dpi": 714756,
        "targets": "PLAT",
        "smear_types": "PB",
        "output": "bboxes",
        "vram_gb": 1.5,
    },
    {
        "name_zh": "高倍巨核细胞分类",
        "name": "DPIALL_BM_MEG_cell_classifier",
        "dpi_range": None,
        "actual_dpi": 357378,
        "targets": "MEG",
        "smear_types": "BM",
        "output": "tops",
        "vram_gb": 1.5,
    },
    {
        "name_zh": "高倍骨髓/外周血有核细胞分类",
        "name": "DPIALL_BM_PB_WBC_classifier",
        "dpi_range": None,
        "actual_dpi": 714756,
        "targets": "WBC",
        "smear_types": "BM/PB",
        "output": "tops",
        "vram_gb": 1.5,
    },
    {
        "name_zh": "高倍脑脊液有核细胞分类",
        "name": "DPIALL_CSF_WBC_cell_classifier",
        "dpi_range": None,
        "actual_dpi": 714756,
        "targets": "WBC",
        "smear_types": "CSF",
        "output": "tops",
        "vram_gb": 1.5,
    },
    {
        "name_zh": "高倍红细胞分类",
        "name": "DPIALL_BM_PB_RED_cell_classifier",
        "dpi_range": None,
        "actual_dpi": 714756,
        "targets": "RBC",
        "smear_types": "PB",
        "output": "tops",
        "vram_gb": 1.5,
    },
    {
        "name_zh": "高倍血小板分类",
        "name": "DPIALL_BM_PB_PLAT_classifier",
        "dpi_range": None,
        "actual_dpi": 714756,
        "targets": "PLAT",
        "smear_types": "PB",
        "output": "tops",
        "vram_gb": 1.5,
    },
    {
        "name_zh": "低倍骨髓区域评分",
        "name": "DPI147246_BM_constituency_score",
        "dpi_range": (73623, 294492),
        "actual_dpi": 147246,
        "targets": "不限制",
        "smear_types": "BM",
        "output": "scores",
        "vram_gb": 1.5,
    },
    {
        "name_zh": "低倍血片区域评分",
        "name": "DPI147246_PB_constituency_score",
        "dpi_range": (73623, 294492),
        "actual_dpi": 147246,
        "targets": "不限制",
        "smear_types": "PB",
        "output": "scores",
        "vram_gb": 1.5,
    },
]

_OUTPUT_KIND = {
    "bboxes": "detection",
    "tops": "classification",
    "scores": "score",
}

DEFAULT_MODEL_VRAM_GB = 1.5
DPI_NOT_SUITABLE = "DPI不合适"


def model_dpi_ranges() -> dict[int, tuple[int, int]]:
    """actual_dpi → 并集后的 DPI 适用范围。"""
    ranges: dict[int, tuple[int, int]] = {}
    for row in MODEL_TABLE:
        dr = row.get("dpi_range")
        if not dr or dr == "不限制":
            continue
        actual = int(row["actual_dpi"])
        low, high = int(dr[0]), int(dr[1])
        if actual in ranges:
            lo, hi = ranges[actual]
            ranges[actual] = (min(lo, low), max(hi, high))
        else:
            ranges[actual] = (low, high)
    return ranges



