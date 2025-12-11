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
