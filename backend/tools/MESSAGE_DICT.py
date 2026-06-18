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

# 根据细胞ID返回计数类别
# 不参与计数：Excluded from the count
# 有核细胞：Nucleated_Cell
# 巨核细胞：Megakaryocyte
def get_counting_cell_type(cell_id):
    if cell_id in CELL_TYPES_X40:
        return "Excluded from the count"
    elif cell_id in CELL_TYPES_X100:
        if cell_id in list(range(200000, 200028)):
            return "Nucleated_Cell"
        else:
            return "Excluded from the count"
    elif cell_id in CELL_TYPES_MEG:
        if cell_id in list(range(300000, 300004)):
            return "Megakaryocyte"
        else:
            return "Excluded from the count"
    elif cell_id in CELL_TYPE_CSF:
        return "Excluded from the count"
    else:
        return "Excluded from the count"

# 目前开放的DPI
dpi_list = [40, 50, 100]
# 允许上传的图片格式
allow_extensions = ['jpg', 'jpeg', 'gif', 'png']
