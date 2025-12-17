# -*- coding: utf-8 -*-
from enum import Enum


class RET_CODE(Enum):
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


class RET_DESC(Enum):
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


class Task_Type(Enum):
    """
    任务类型枚举
    """
    WBC = 0  # 任务类型0
    MEG = 1  # 任务类型1
    RBC = 2  # 任务类型2


CELL_TYPES = {
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

RET_CODE_OK = 200
RET_DESC_OK = 'success'
RET_DESC_TASK_CREATED = '任务创建成功'
RET_DESC_UPLOAD_OK = '图片上传成功'
RET_DESC_RESULT_OK = '结果获取成功'

RET_CODE_ERROR = -1
RET_DESC_EXISTS = '相关条目已存在'
RET_DESC_NOT_EXISTS = '相关条目不存在'
RET_DESC_PARAMS_ERROR = '参数错误'
RET_DESC_FILE_TYPE_ERROR = '文件类型错误'

RET_CODE_TASK_OK = 100
RET_DESC_TASK_OK = "任务已完成"

RET_CODE_TASK_RUN = 101
RET_DESC_TASK_RUN = '任务正在进行中'

RET_CODE_TASK_TIMEOUT = 102
RET_DESC_TASK_TIMEOUT = "任务超时"

RET_CODE_IMAGE_ERROR = 103
RET_DESC_IMAGE_UPLOADED = "图片已上传,请勿重复上传"

RET_CODE_IMAGE_MISSING = 104
RET_DESC_IMAGE_MISSING = "图片已损坏,请检查上传图片是否完整"

RET_CODE_TASK_BUSY = 105
RET_DESC_TASK_BUSY = "当前有任务正在进行，请稍后再试"

RET_DESC_TASK_UNKNOWN = '未知任务状态'

allow_extensions = ['jpg', 'jpeg', 'gif', 'png']
