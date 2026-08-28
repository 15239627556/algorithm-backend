from io import BytesIO
from typing import Any, Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from backend.services.task_service import TaskService
from backend.tools.MESSAGE_DICT import RetCode, RetDesc

_BINARY_CONTENT_TYPES = frozenset({
    "application/octet-stream",
    "image/jpeg",
    "image/jpg",
    "image/png",
})

taskService = TaskService()

task = APIRouter(prefix="/api/v1/smear_analysis", tags=["任务相关接口"])


def _edge_cell_filter_form(value: Optional[str]) -> bool:
    """form 中 edge_cell_filter 常为字符串，不能用 type=bool（bool('false') 为 True）。"""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("0", "false", "no", "off", ""):
        return False
    return True


class _UploadAdapter:
    """兼容 TaskService 对 FileStorage 的 .read() / .filename 用法。"""

    __slots__ = ("filename", "_file")

    def __init__(self, upload: UploadFile) -> None:
        self.filename = upload.filename
        self._file = upload.file
        try:
            self._file.seek(0)
        except Exception:
            pass

    def read(self, *args, **kwargs):
        return self._file.read(*args, **kwargs)


# ---------- OpenAPI / 请求体模型（extra=allow，兼容客户端多余字段）----------

class CreateTaskBody(BaseModel):
    model_config = ConfigDict(extra="allow")
    smear_type: str = Field(default="BM", description="涂片类型:BM, PB, CF")
    dpi: int = Field(default=144750, description="DPI，模型据此选择: 144750/357378/714756")
    tile_width: int = Field(default=2448, description="拼图块宽度")
    tile_height: int = Field(default=2048, description="拼图块高度")
    target_cell_types: str = Field(
        default="",
        description="目标细胞类型如 WBC,MEG / WBC,RBC，供任务模式使用",
    )


class TileMsg(BaseModel):
    model_config = ConfigDict(extra="allow")
    row_index: int
    col_index: int
    position_x: int
    position_y: int


class UpdateCoordinatesBody(BaseModel):
    model_config = ConfigDict(extra="allow")
    task_id: str
    tiles_msg: list[TileMsg]


class TaskIdBody(BaseModel):
    model_config = ConfigDict(extra="allow")
    task_id: str


class GetTaskResultBody(BaseModel):
    model_config = ConfigDict(extra="allow")
    task_id: str
    roi_xmin: int = 0
    roi_ymin: int = 0
    roi_xmax: Optional[int] = None
    roi_ymax: Optional[int] = None
    index_offset: int = 0
    request_task_num: int = 100


class UserChoiceArea(BaseModel):
    model_config = ConfigDict(extra="allow")
    x_min: Optional[int] = None
    y_min: Optional[int] = None
    x_max: Optional[int] = None
    y_max: Optional[int] = None


class WbcPoint(BaseModel):
    model_config = ConfigDict(extra="allow")
    x: int
    y: int
    w: int
    h: int


class RoiKwargs(BaseModel):
    model_config = ConfigDict(extra="allow")
    index_offset: int = 0
    request_task_num: int = 100
    wbc_points: Optional[list[WbcPoint]] = None


class RequiredNum(BaseModel):
    model_config = ConfigDict(extra="allow")
    WBC: Optional[int] = None
    MEG: Optional[int] = None
    RBC: Optional[int] = None


class RoiSelectionBody(BaseModel):
    model_config = ConfigDict(extra="allow")
    task_id: str
    task_type: str
    user_choice_area: Optional[UserChoiceArea] = None
    view_width: int
    view_height: int
    kwargs: Optional[RoiKwargs] = None
    required_num: Optional[RequiredNum] = None


class GenerateViewsBody(BaseModel):
    model_config = ConfigDict(extra="allow")
    rects: list[list[float]]
    view_width: int = 384
    view_height: int = 283
    pad: int = 100


class AnalyzeSlideBody(BaseModel):
    model_config = ConfigDict(extra="allow")
    task_id: Optional[str] = None
    analyze_names: list[Any] = Field(default_factory=list)


class ModelControlBody(BaseModel):
    model_config = ConfigDict(extra="allow")
    smear_type: str = Field(default="BM", description="涂片类型: BM/PB/CSF（CF 视为 CSF）")
    dpi: int = Field(..., description="DPI，如 714756/357378/147246/35000")
    target_cell_types: str = Field(
        ...,
        description="目标检测类型，如 WBC,RBC / WBC,MEG / WBC,PLAT",
    )
    gpu_id: Optional[int] = Field(default=None, description="指定 GPU，默认轮询单卡")
    all_gpus: bool = Field(default=False, description="是否对所有 Triton 端点操作")


ALLOWED_ANALYZE_NAMES = {"cellularity"}


def _is_binary_tile_upload(request: Request) -> bool:
    """判断是否为二进制流上传（兼容 multipart 原格式）。"""
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if "multipart/form-data" in content_type:
        return False
    if content_type in _BINARY_CONTENT_TYPES:
        return True
    # Content-Type 未声明时：有 query 参数且 content-length>0，视为裸流
    return (
        bool(request.query_params.get("task_id"))
        and request.headers.get("content-length") not in (None, "0")
    )


@task.post("/create_task", summary="X40创建任务")
def create_task(body: CreateTaskBody):
    result = taskService.create_task(body.model_dump())
    return result


@task.post(
    "/upload_tile",
    summary="上传图片（任务模式）",
    description=(
        "task_id+row_index+col_index必填。支持两种格式："
        "1) multipart/form-data：form 字段 + tile_image 文件；"
        "2) 二进制流：Body 为图像字节，task_id/row_index/col_index 走 URL Query，"
        "Content-Type: application/octet-stream"
    ),
)
async def upload_tile(request: Request):
    # 同一路由需兼容 multipart 与裸二进制，故手动解析，避免 Form 抢占 body
    if _is_binary_tile_upload(request):
        tid = request.query_params.get("task_id")
        ridx_raw = request.query_params.get("row_index")
        cidx_raw = request.query_params.get("col_index")
        try:
            ridx = int(ridx_raw) if ridx_raw is not None else None
            cidx = int(cidx_raw) if cidx_raw is not None else None
        except (TypeError, ValueError):
            ridx, cidx = None, None
        image_bytes = await request.body()
        if not tid or ridx is None or cidx is None:
            return {
                "ret_code": RetCode.CLIENT_ERROR.value,
                "ret_desc": "Binary upload requires query params: task_id, row_index, col_index",
                "reason": "Binary upload requires query params: task_id, row_index, col_index",
            }
        if not image_bytes:
            return {
                "ret_code": RetCode.CLIENT_ERROR.value,
                "ret_desc": "Empty image body",
                "reason": "Empty image body",
            }
        file_obj: Any = BytesIO(image_bytes)
    else:
        form = await request.form()
        tid = form.get("task_id")
        ridx_raw = form.get("row_index")
        cidx_raw = form.get("col_index")
        tile_image = form.get("tile_image")
        try:
            ridx = int(ridx_raw) if ridx_raw is not None else None
            cidx = int(cidx_raw) if cidx_raw is not None else None
        except (TypeError, ValueError):
            ridx, cidx = None, None
        if not tid or ridx is None or cidx is None or tile_image is None:
            return {
                "ret_code": RetCode.CLIENT_ERROR.value,
                "ret_desc": "multipart upload requires form: task_id, row_index, col_index, tile_image",
                "reason": "multipart upload requires form: task_id, row_index, col_index, tile_image",
            }
        # form 文件字段可能是 starlette UploadFile（勿仅用 isinstance，跨包时会失败）
        if hasattr(tile_image, "file") and hasattr(tile_image, "filename"):
            file_obj = _UploadAdapter(tile_image)
        elif isinstance(tile_image, (bytes, bytearray)):
            file_obj = BytesIO(tile_image)
        elif hasattr(tile_image, "read"):
            file_obj = tile_image
        else:
            return {
                "ret_code": RetCode.CLIENT_ERROR.value,
                "ret_desc": "Invalid tile_image",
                "reason": "Invalid tile_image",
            }
    # 同步推理/写盘放到线程池，避免阻塞事件循环
    return await run_in_threadpool(taskService.upload_image, tid, ridx, cidx, file_obj)

@task.post("/update_coordinates", summary="更新拼图块坐标信息")
def update_coordinates(body: UpdateCoordinatesBody):
    result = taskService.update_coordinates(
        body.task_id,
        [m.model_dump() for m in body.tiles_msg],
    )
    return result


@task.post("/check_missing_tiles", summary="检查图片是否上传完毕")
def check_missing_tiles(body: TaskIdBody):
    result = taskService.check_image(body.task_id)
    return result


@task.post("/check_task_status", summary="获取任务状态")
def check_task_status(body: TaskIdBody):
    result = taskService.task_status(body.task_id)
    return result


@task.post("/get_task_result", summary="获取任务结果")
def get_task_result(body: GetTaskResultBody):
    result = taskService.get_result(
        body.task_id,
        body.roi_xmin,
        body.roi_ymin,
        body.roi_xmax,
        body.roi_ymax,
        body.index_offset,
        body.request_task_num,
    )
    return result


@task.post("/roi_selection", summary="获取X100任务列表")
def roi_selection(body: RoiSelectionBody):
    kwargs = body.kwargs.model_dump() if body.kwargs else {}
    required_num = body.required_num.model_dump(exclude_none=True) if body.required_num else {}
    user_choice_area = body.user_choice_area.model_dump() if body.user_choice_area else None
    result = taskService.get_task_list_x100(
        task_id=body.task_id,
        task_type=body.task_type,
        user_choice_area=user_choice_area,
        view_width=body.view_width,
        view_height=body.view_height,
        kwargs=kwargs,
        required_num=required_num,
    )
    return result


@task.post(
    "/generate_views",
    summary="Generate view boxes to cover rects",
    description="Generate minimum number of view boxes to cover all rects (set cover).",
)
def generate_views(body: GenerateViewsBody):
    result = taskService.generate_views(
        rects=body.rects,
        view_width=body.view_width,
        view_height=body.view_height,
        pad=body.pad,
    )
    return result


@task.post("/analyze_slide", summary="玻片分析")
def analyze_slide(body: AnalyzeSlideBody):
    task_id = body.task_id
    analyze_names = body.analyze_names
    if not task_id:
        return {
            "ret_code": RetCode.CLIENT_ERROR.value,
            "ret_desc": "task_id cannot be empty",
            "result": {},
        }
    if not isinstance(analyze_names, list):
        return {
            "ret_code": RetCode.CLIENT_ERROR.value,
            "ret_desc": RetDesc.CLIENT_ERROR.value,
            "result": {},
        }
    if not analyze_names:
        return {
            "ret_code": RetCode.CLIENT_ERROR.value,
            "ret_desc": "analyze_names cannot be empty",
            "result": {},
        }
    invalid = [n for n in analyze_names if n not in ALLOWED_ANALYZE_NAMES]
    if invalid:
        return {
            "ret_code": RetCode.CLIENT_ERROR.value,
            "ret_desc": f"Unsupported analyze item: {invalid}, only supported: {list(ALLOWED_ANALYZE_NAMES)}",
            "result": {},
        }
    result = taskService.analyze_slide(task_id, analyze_names)
    return result


@task.post("/load_models", summary="按需加载模型")
def load_models_api(body: ModelControlBody):
    from backend.tools.combo_validator import validate_combo
    from backend.tools.model_control import load_models, resolve_models

    ok_combo, err = validate_combo(body.dpi, body.smear_type, body.target_cell_types)
    if not ok_combo:
        return {
            "ret_code": RetCode.CLIENT_ERROR.value,
            "ret_desc": err,
            "reason": err,
        }
    resolved = resolve_models(body.dpi, body.smear_type, body.target_cell_types)
    models = resolved.names
    ok, msg, _ = load_models(
        body.dpi,
        body.smear_type,
        body.target_cell_types,
        gpu_id=body.gpu_id,
        all_gpus=body.all_gpus,
    )
    if not ok:
        return {
            "ret_code": RetCode.CLIENT_ERROR.value,
            "ret_desc": msg,
            "reason": msg,
            "models": models,
            "detection": [s.name for s in resolved.detection],
            "classification": [s.name for s in resolved.classification],
            "score": [s.name for s in resolved.score],
        }
    return {
        "ret_code": RetCode.API_SUCCESS.value,
        "ret_desc": RetDesc.API_SUCCESS.value,
        "models": models,
        "detection": [s.name for s in resolved.detection],
        "classification": [s.name for s in resolved.classification],
        "score": [s.name for s in resolved.score],
        "warning": err or resolved.warning,
    }


@task.post("/unload_models", summary="按需卸载模型")
def unload_models_api(body: ModelControlBody):
    from backend.tools.combo_validator import validate_combo
    from backend.tools.model_control import resolve_models, unload_models

    ok_combo, err = validate_combo(body.dpi, body.smear_type, body.target_cell_types)
    if not ok_combo:
        return {
            "ret_code": RetCode.CLIENT_ERROR.value,
            "ret_desc": err,
            "reason": err,
        }
    resolved = resolve_models(body.dpi, body.smear_type, body.target_cell_types)
    models = resolved.names
    ok, msg, _ = unload_models(
        body.dpi,
        body.smear_type,
        body.target_cell_types,
        gpu_id=body.gpu_id,
        all_gpus=body.all_gpus,
    )
    if not ok:
        return {
            "ret_code": RetCode.CLIENT_ERROR.value,
            "ret_desc": msg,
            "reason": msg,
            "models": models,
            "detection": [s.name for s in resolved.detection],
            "classification": [s.name for s in resolved.classification],
            "score": [s.name for s in resolved.score],
        }
    return {
        "ret_code": RetCode.API_SUCCESS.value,
        "ret_desc": RetDesc.API_SUCCESS.value,
        "models": models,
        "detection": [s.name for s in resolved.detection],
        "classification": [s.name for s in resolved.classification],
        "score": [s.name for s in resolved.score],
        "warning": err or resolved.warning,
    }


@task.post(
    "/analyze_cell_image",
    summary="细胞图像分析",
    description="任务模式：task_id+position；单张识别：dpi+algorithm_types必填",
)
def analyze_cell_image(
    image_file: UploadFile = File(..., description="图像文件（.jpg格式）"),
    task_id: Optional[str] = Form(None),
    position_xmin: Optional[int] = Form(None),
    position_ymin: Optional[int] = Form(None),
    position_xmax: Optional[int] = Form(None),
    position_ymax: Optional[int] = Form(None),
    dpi: int = Form(..., description="DPI，模型据此选择: 144750/357378/714756"),
    target_cell_types: str = Form(
        ...,
        description="目标细胞类型如 WBC,MEG / WBC,RBC,PLAT / MEG 等",
    ),
    smear_type: Optional[str] = Form(
        None,
        description="涂片类型BM/PB/CF，单张识别时使用，有task_id时从任务取",
    ),
    edge_cell_filter: Optional[str] = Form(
        "true",
        description="是否过滤边缘细胞，默认 true；可传 false/0/off",
    ),
):
    result = taskService.get_task_result_x100(
        task_id,
        _UploadAdapter(image_file),
        target_cell_types,
        dpi,
        _edge_cell_filter_form(edge_cell_filter),
        smear_type,
        position_xmin,
        position_ymin,
        position_xmax,
        position_ymax,
    )
    return result
