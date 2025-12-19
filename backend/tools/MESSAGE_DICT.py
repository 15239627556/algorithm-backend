# -*- coding: utf-8 -*-
from enum import Enum

class TaskStatus(Enum):
    """
    任务状态（对应文档中的 100/101/102 等状态码）
    """
    COMPLETED = 100  # 任务已完成
    RUNNING = 101  # 任务进行中
    TIMEOUT = 102  # 任务超时

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
    TASK_FINISHED = "任务已完成"
    TASK_RUNNING = "任务进行中"
    TASK_TIMEOUT = "任务超时"
    IMAGE_ALREADY_UPLOADED = "图片已上传，请勿重复上传"
    IMAGE_DAMAGED = "图片损坏，请检查上传图片是否完整"
    TASK_IN_PROGRESS = "当前有任务正在进行"

    API_SUCCESS = "接口调用成功"
    CLIENT_ERROR = "参数错误、文件类型错误、相关条目不存在等客户端错误"


class TaskType(Enum):
    """
    任务类型枚举
    """
    WBC = 0  #
    MEG = 1  #
    RBC = 2  #


CELL_TYPES_X40 = {
    100000: ("Unclassified_WBC", "未分类白细胞"),
    100001: ("Unclassified_MEG", "未分类巨核细胞"),
    100002: ("Unclassified_RBC", "未分类红细胞"),
    100003: ("Broken_Cell", "破碎细胞"),
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

# 目前开放的DPI
dpi_list = [40, 100]
# 允许上传的图片格式
allow_extensions = ['jpg', 'jpeg', 'gif', 'png']
