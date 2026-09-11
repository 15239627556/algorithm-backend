# triton_client.py
"""multipart 对齐 multi_pipeline_server：细胞检测 POST /infer（dpi+slide_type+task）；
全局图 POST /global/infer；滤镜 POST /{image_enhance|opencv_enhance}/infer（裸流）。按 actual_dpi 与结果解析。"""
from __future__ import annotations

import os
import sys

# 直接运行本文件时（python -m backend.tools.triton_client）将项目根加入 path
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)

import json
import logging
import threading
import time
from io import BytesIO
from urllib.parse import quote, urlparse
from typing import Any, Callable, List, Optional

import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from project.cells import Cell
from backend.tools.MESSAGE_DICT import (
    CELL_TYPES_X40,
    CELL_TYPES_X100,
    CELL_TYPES_MEG,
    CELL_TYPE_CSF,
    CELL_TYPE_RBC,
    CELL_TYPE_PLT,
    get_counting_cell_type,
)
from backend.tools.combo_validator import normalize_smear_type
from config import next_triton_endpoint, get_triton_endpoint
from backend.tools.model_control import (
    ensure_model_loaded,
    resolve_models,
    ResolvedModels,
)
from backend.tools.filter_edge_incomplete_cells import um_per_pixel_from_dpi

logger = logging.getLogger(__name__)

# 细胞检测走 multi_pipeline_server：POST /infer，Form 传 actual_dpi。
# actual_dpi 来自 MODEL_TABLE 命中的定位模型，不是 Triton 仓库模型名。
DPI_147246 = 147246
DPI_357378 = 357378
DPI_714756 = 714756
DPI_35000 = 35000
DPI_71000 = 71000
DPI_144750 = 144750  # 遗留请求值，落在 147246 区间

# CF 检测统一输出未分类脑脊液细胞
CSF_UNCLASSIFIED_CELL_TYPE = 100007
# 图片增强/滤镜（x40 超分辨率滤镜深度学习模式）
MODEL_IMAGE_ENHANCE = "Image_enhance_pipeline"

_PIPELINE_DPI_VALUES = frozenset({147246, 357378, 714756, 35000, 71000})
_FILTER_PIPELINE_TARGETS = frozenset({"image_enhance", "opencv_enhance"})
GLOBAL_IMAGE_ANALYSIS_MODEL = "GLOBAL-IMAGE-ANALYSIS"
GLOBAL_IMAGE_HEAD_DIR_MODEL = "GLOBAL-IMAGE-HEAD-DIR"
_GLOBAL_TASK_MODELS = {
    "roi": GLOBAL_IMAGE_ANALYSIS_MODEL,
    "dir": GLOBAL_IMAGE_HEAD_DIR_MODEL,
}
_VALID_GLOBAL_SMEAR_TYPES = frozenset({"BM", "PB"})
_VALID_GLOBAL_TASKS = frozenset(_GLOBAL_TASK_MODELS.keys())
_GLOBAL_INFER_SKIP_RESULT_KEYS = frozenset(
    {
        "error",
        "type",
        "trace_tail",
        "code",
        "_timing_models",
        "_plugin_timing",
        "timing",
        "latency_ms",
        "payload_bytes",
        "image_format",
        "payload_read_ms",
        "request_total_ms",
        "ran_models",
        "models",
        "dpi",
        "slide_type",
        "task",
    }
)
_PIPELINE_INFER_URL_RAW = (
    os.environ.get("PIPELINE_INFER_URL", "").strip().rstrip("/")
    or os.environ.get("PIPELINE_147246_INFER_URL", "").strip().rstrip("/")
)

PIPELINE_HTTP_TIMEOUT_S = float(os.environ.get("PIPELINE_HTTP_TIMEOUT_S", "600"))

# cell_analysis 模型固定输入尺寸；不足时居中补 RGB(230,230,230)
_CELL_ANALYSIS_TARGET_W = 2448
_CELL_ANALYSIS_TARGET_H = 2048
_CELL_ANALYSIS_PAD_RGB = (230, 230, 230)

# 连接建立阶段的超时（秒）。读取阶段用 PIPELINE_HTTP_TIMEOUT_S，推理耗时较长故单独区分。
PIPELINE_HTTP_CONNECT_TIMEOUT_S = float(os.environ.get("PIPELINE_HTTP_CONNECT_TIMEOUT_S", "10"))
# 仅对“连接建立失败”做有限重试；推理 POST 非幂等，故不重试已发出的请求（read/status 不重试）。
_PIPELINE_HTTP_CONNECT_RETRIES = int(os.environ.get("PIPELINE_HTTP_CONNECT_RETRIES", "2"))
# pipeline 裸流 POST 连接层失败时的应用层重试次数（含首次，默认最多 3 次）。
_PIPELINE_HTTP_POST_MAX_ATTEMPTS = int(os.environ.get("PIPELINE_HTTP_POST_MAX_ATTEMPTS", "3"))
# 连接池：按 host 复用；需 ≥ Web 侧并发（双端点轮询时每端各占一半）。
# 默认与 THREAD_POOL_SIZE 同量级，否则线程多了也只会在 urllib3 池里排队。
_PIPELINE_HTTP_POOL_CONNECTIONS = int(os.environ.get("PIPELINE_HTTP_POOL_CONNECTIONS", "32"))
_PIPELINE_HTTP_POOL_MAXSIZE = int(os.environ.get("PIPELINE_HTTP_POOL_MAXSIZE", "64"))

# 注意：requests.Session 不是线程安全的。进程内共享一个 Session 在高并发下会
# 把连接池打坏，表现为「一开始并发高，随后掉到几十」。改为 thread-local。
_thread_local = threading.local()
_pool_config_logged = False
_pool_config_log_lock = threading.Lock()


def _new_pipeline_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=_PIPELINE_HTTP_CONNECT_RETRIES,
        connect=_PIPELINE_HTTP_CONNECT_RETRIES,
        read=0,
        status=0,
        redirect=0,
        backoff_factor=0.5,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        pool_connections=_PIPELINE_HTTP_POOL_CONNECTIONS,
        pool_maxsize=_PIPELINE_HTTP_POOL_MAXSIZE,
        max_retries=retry,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _get_pipeline_session() -> requests.Session:
    """每线程独立 Session（连接池复用且线程安全）。"""
    global _pool_config_logged
    session = getattr(_thread_local, "pipeline_session", None)
    if session is None:
        session = _new_pipeline_session()
        _thread_local.pipeline_session = session
        if not _pool_config_logged:
            with _pool_config_log_lock:
                if not _pool_config_logged:
                    logger.info(
                        "pipeline HTTP pool: connections=%s maxsize=%s (thread-local sessions)",
                        _PIPELINE_HTTP_POOL_CONNECTIONS,
                        _PIPELINE_HTTP_POOL_MAXSIZE,
                    )
                    _pool_config_logged = True
    return session


def _reset_pipeline_session() -> None:
    """丢弃当前线程的 Session，避免重试时复用已断开的脏连接。"""
    session = getattr(_thread_local, "pipeline_session", None)
    if session is not None:
        try:
            session.close()
        except Exception:
            pass
        _thread_local.pipeline_session = None


def _normalize_http_url(url_or_hostport: str) -> str:
    s = url_or_hostport.strip().rstrip("/")
    if not s:
        return s
    if not s.lower().startswith(("http://", "https://")):
        s = f"http://{s}"
    return s


def _strip_plain_infer_suffix(url: str) -> str | None:
    """若为 http(s)://host[:port]/infer（无 DPI 前缀），返回 scheme://netloc，否则 None。"""
    u = _normalize_http_url(url)
    p = urlparse(u)
    if not p.scheme or not p.netloc:
        return None
    path_norm = ((p.path or "").rstrip("/")).lower()
    if path_norm != "/infer":
        return None
    return f"{p.scheme}://{p.netloc}"


def _pipeline_infer_url(endpoint: dict | None = None) -> str:
    """统一细胞检测推理：POST /infer（multipart）。"""
    ov_raw = _PIPELINE_INFER_URL_RAW
    if ov_raw:
        ov = _normalize_http_url(ov_raw)
        path_norm = (urlparse(ov).path or "").rstrip("/").lower()
        if path_norm.endswith("/infer"):
            return ov.rstrip("/")
        root = _strip_plain_infer_suffix(ov)
        if root is not None:
            return f"{root}/infer"
        return ov

    ep = endpoint or get_triton_endpoint()
    bs = _normalize_http_url(ep.get("pipeline_base_url") or "")
    return f"{bs.rstrip('/')}/infer"


def _global_pipeline_infer_url(endpoint: dict | None = None) -> str:
    """全局图分析：POST /global/infer（multipart）。"""
    ep = endpoint or get_triton_endpoint()
    bs = _normalize_http_url(ep.get("pipeline_base_url") or "")
    return f"{bs.rstrip('/')}/global/infer"


def _filter_pipeline_infer_url(target: str, endpoint: dict | None = None) -> str:
    """滤镜接口：POST /{image_enhance|opencv_enhance}/infer（裸流，响应为图片字节）。"""
    if target not in _FILTER_PIPELINE_TARGETS:
        raise ValueError(f"invalid filter pipeline target: {target!r}")
    ep = endpoint or get_triton_endpoint()
    bs = _normalize_http_url(ep.get("pipeline_base_url") or "")
    return f"{bs.rstrip('/')}/{target}/infer"


def _post_filter_pipeline_infer(
    url: str,
    image_bytes: bytes,
    timeout_s: float,
) -> tuple[bytes, str]:
    """滤镜推理：发送裸流，响应为 image/jpeg 或 image/png 字节。"""
    if not url.lower().startswith("http"):
        url = f"http://{url}"

    last_error: requests.exceptions.RequestException | None = None
    resp: requests.Response | None = None
    for attempt in range(1, _PIPELINE_HTTP_POST_MAX_ATTEMPTS + 1):
        try:
            resp = _get_pipeline_session().post(
                url,
                data=image_bytes,
                headers={"Content-Type": "application/octet-stream"},
                timeout=(PIPELINE_HTTP_CONNECT_TIMEOUT_S, timeout_s),
            )
            last_error = None
            break
        except requests.exceptions.ConnectionError as e:
            last_error = e
            logger.warning(
                "filter pipeline 连接失败 (attempt %d/%d): %s",
                attempt,
                _PIPELINE_HTTP_POST_MAX_ATTEMPTS,
                e,
            )
            _reset_pipeline_session()
            if attempt < _PIPELINE_HTTP_POST_MAX_ATTEMPTS:
                time.sleep(0.5 * attempt)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"filter pipeline 请求失败: {e}") from e

    if last_error is not None:
        raise RuntimeError(
            f"filter pipeline 请求失败（已重试 {_PIPELINE_HTTP_POST_MAX_ATTEMPTS} 次）: {last_error}"
        ) from last_error

    assert resp is not None
    if resp.status_code >= 400:
        try:
            err_json = resp.json()
            error = err_json.get("error", err_json)
        except ValueError:
            error = resp.text[:500]
        raise RuntimeError(f"filter pipeline HTTP {resp.status_code}: {error}")

    if not resp.content:
        raise RuntimeError("滤镜接口返回了空图片")

    content_type = resp.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type not in {"image/jpeg", "image/png"}:
        raise RuntimeError(f"滤镜接口返回类型异常: {content_type or '(empty)'}")
    return resp.content, content_type


def _ensure_filter_model_loaded(endpoint: dict, gpu_id: int, target: str) -> None:
    """
    推理前经 model_control 加载滤镜模型。
    opencv_enhance 纯 CPU，无需加载 Triton 模型。
    """
    if target != "image_enhance":
        return

    ok, err = ensure_model_loaded(MODEL_IMAGE_ENHANCE, gpu_id=gpu_id)
    if not ok:
        raise RuntimeError(f"filter model load failed on gpu={gpu_id}: {err}")
    logger.info(
        "filter model ensured gpu_id=%s target=%s endpoint=%s",
        gpu_id,
        target,
        _normalize_http_url(endpoint.get("pipeline_base_url") or ""),
    )


# X50 14 类 → 200000-200013
X50_CLASS_NAMES = [f"类{i}" for i in range(14)]

# 714756 RED 分类 head：label_id <-> class_id（与 pipeline 训练 meta 一致）
RED_COLOR_MAP = {"1": 0, "23": 1, "24": 2, "25": 3}
RED_STRUCT_MAP = {"1": 0, "27": 1, "28": 2, "29": 3, "30": 4, "31": 5, "32": 6}
RED_MORPH_MAP = {
    "1": 0,
    "3": 1, "4": 2, "5": 3, "6": 4, "7": 5,
    "8": 6, "9": 7, "10": 8, "11": 9, "12": 10,
    "13": 11, "14": 12, "15": 13, "17": 14,
}
# 单个细胞 agg=0；凝集 34->1，35->2
RED_AGG_MAP = {"1": 0, "34": 1, "35": 2}

RED_STRUCT_LABELS = {
    "1": "无结构异常",
    "27": "嗜碱性点彩红细胞",
    "28": "豪-乔小体",
    "29": "Heinz小体",
    "30": "卡波环",
    "31": "Pappenheimer小体",
    "32": "寄生虫",
}
RED_COLOR_LABELS = {
    "1": "无颜色异常",
    "23": "低色素",
    "24": "高色素",
    "25": "嗜多色性红细胞",
}
RED_MORPH_LABELS = {
    "1": "无形状异常",
    "3": "球形红细胞",
    "4": "椭圆形红细胞",
    "5": "靶形红细胞",
    "6": "镰状红细胞",
    "7": "泪滴形红细胞",
    "8": "口形红细胞",
    "9": "裂红细胞",
    "10": "棘形红细胞",
    "11": "皱缩红细胞",
    "12": "咬痕红细胞",
    "13": "水泡红细胞",
    "14": "盔形红细胞",
    "15": "新月形红细胞",
    "17": "不规则红细胞",
}
RED_AGG_LABELS = {
    "1": "单个细胞",
    "34": "缗钱状红细胞",
    "35": "凝集红细胞",
}
RED_SIZE_LABELS = {
    "1": "无大小异常",
    "19": "大红细胞",
    "20": "小红细胞",
    "21": "巨大红细胞",
}

PLAT_MORPH_LABELS = {
    '0': '无形态异常',
    '1': '畸形血小板'
}

PLAT_DIST_LABELS = {
    '0': '无分布异常',
    '1': '片状',
    '2': '小簇',
    '3': '大簇',
    '4': '杂志'
}

PLAT_COLOR_LABELS = {
    '0': '无颜色异常',
    "1": "颗粒减少血小板",
    "2": "灰色血小板"
}

PLAT_SEZ_LABELS = {
    '0': '无大小异常',
    '1': '大血小板',
    '2': '小血小板',
    '3': '巨大血小板'
}

# 成熟红细胞 TOP5：凝集 > 形状 > 结构 > 大小 > 颜色；label_id "1" 为无异常
_RED_TOP5_PRIORITY = ("AGG", "MORPH", "STRUCT", "SIZE", "COLOR")
_RED_NORMAL_LABEL_IDS = frozenset({"1"})
# 血小板 TOP5：形状 > 分布 > 大小 > 颜色；label_id "0" 为无异常
_PLAT_TOP5_PRIORITY = ("MORPH", "DIST", "SIZE", "COLOR")
_PLAT_NORMAL_LABEL_IDS = frozenset({"0"})
# 血小板 (head, label_id) → CELL_TYPE_PLT
_PLAT_HEAD_LABEL_TO_TYPE = {
    ("MORPH", "1"): 600005,
    ("DIST", "1"): 600001,
    ("DIST", "2"): 600002,
    ("DIST", "3"): 600003,
    ("DIST", "4"): 600004,
    ("SIZE", "1"): 600008,
    ("SIZE", "2"): 600009,
    ("SIZE", "3"): 600010,
    ("COLOR", "1"): 600006,
    ("COLOR", "2"): 600007,
}


def _red_inv_from_map(label_map: dict[str, int]) -> List[str]:
    """由 label_id->class_id 反查 class_id->label_id 列表。"""
    if not label_map:
        return []
    n = max(label_map.values()) + 1
    inv = [""] * n
    for label_id, idx in label_map.items():
        if 0 <= idx < n:
            inv[idx] = label_id
    return inv


RED_STRUCT_INV = _red_inv_from_map(RED_STRUCT_MAP)
RED_COLOR_INV = _red_inv_from_map(RED_COLOR_MAP)
RED_MORPH_INV = _red_inv_from_map(RED_MORPH_MAP)
RED_AGG_INV = _red_inv_from_map(RED_AGG_MAP)

# 按 gpu_id 缓存 gRPC 客户端（双容器各一个）
_triton_clients: dict[int, Any] = {}
_triton_client_lock = threading.Lock()


def _infer_route_dpi(resolved: ResolvedModels) -> int | None:
    """从 MODEL_TABLE 命中的模型取 /infer 的 dpi 参数（actual_dpi）。"""
    actuals = {spec.actual_dpi for spec in resolved.detection}
    if not actuals:
        actuals = {spec.actual_dpi for spec in resolved.score}
    if not actuals:
        return None
    for dpi in (DPI_147246, DPI_35000, DPI_71000, DPI_357378, DPI_714756):
        if dpi in actuals:
            return dpi
    return next(iter(actuals))


def get_model_by_dpi(
    dpi: int,
    smear_type: str = "BM",
    algorithm_types: str = "",
    *,
    return_warning: bool = False,
) -> str | tuple[str, str | None]:
    """返回 MODEL_TABLE 命中的定位模型名（逗号拼接）。"""
    resolved = resolve_models(dpi, smear_type, algorithm_types)
    names = ",".join(spec.name for spec in resolved.detection)
    if return_warning:
        return names, resolved.warning
    return names


def _pipeline_task_from_algorithm_types(
    algorithm_types: str,
    route_dpi: int,
    smear_type: str,
) -> str:
    """
    将 target_cell_types / algorithm_types 转为 multi_pipeline /infer 的 task 字段。
    与 model.json 中 task 对齐：wbc / meg / rbc / plat / cscore 等。
    """
    from backend.tools.combo_validator import _parse_cell_types

    task_map = {
        "WBC": "wbc",
        "MEG": "meg",
        "RBC": "rbc",
        "RED": "rbc",
        "PLAT": "plat",
        "PLT": "plat",
    }
    parts: list[str] = []
    seen: set[str] = set()
    for cell_type in _parse_cell_types(algorithm_types):
        task = task_map.get(cell_type, cell_type.lower())
        if task in seen:
            continue
        seen.add(task)
        parts.append(task)

    st = normalize_smear_type(smear_type)
    if (
        route_dpi == DPI_147246
        and st in ("BM", "PB")
        and seen.intersection({"wbc", "meg"})
        and "cscore" not in seen
    ):
        parts.append("cscore")

    return ",".join(parts)


def _post_raw_pipeline_infer(
    url: str,
    image_bytes: bytes,
    filename: str,  # 保留用于日志打印
    timeout_s: float,
    extra_form: dict[str, str] | None = None,
) -> dict[str, Any]:
    """直接发送二进制流（裸流），参数对齐 multi_pipeline_server（通过 URL Query 传递）。"""
    if not url.lower().startswith("http"):
        url = f"http://{url}"

    # 将原先的表单参数整理好，准备放到 URL 后面
    params = {name: value for name, value in extra_form.items()} if extra_form else {}

    # 把filename也放在params中
    params["filename"] = filename

    last_error: requests.exceptions.RequestException | None = None
    resp: requests.Response | None = None
    # logger.info(f"url: {url}")
    for attempt in range(1, _PIPELINE_HTTP_POST_MAX_ATTEMPTS + 1):
        # logger.info(
        #     "file_name=%s, 发送请求的时间：%s (attempt %d/%d)",
        #     filename,
        #     datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        #     attempt,
        #     _PIPELINE_HTTP_POST_MAX_ATTEMPTS,
        # )
        try:
            resp = _get_pipeline_session().post(
                url,
                params=params,        # 👈 核心改动 1：参数走 URL Query
                data=image_bytes,     # 👈 核心改动 2：直接塞入纯 bytes
                headers={"Content-Type": "application/octet-stream"}, # 👈 核心改动 3：明确告诉服务端这是纯二进制
                timeout=(PIPELINE_HTTP_CONNECT_TIMEOUT_S, timeout_s),
            )
            last_error = None
            break
        except requests.exceptions.ConnectionError as e:
            last_error = e
            logger.warning(
                "file_name=%s, pipeline 连接失败 (attempt %d/%d): %s",
                filename,
                attempt,
                _PIPELINE_HTTP_POST_MAX_ATTEMPTS,
                e,
            )
            _reset_pipeline_session()
            if attempt < _PIPELINE_HTTP_POST_MAX_ATTEMPTS:
                time.sleep(0.5 * attempt)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"pipeline_server 请求失败: {e}") from e

    if last_error is not None:
        raise RuntimeError(
            f"pipeline_server 请求失败（已重试 {_PIPELINE_HTTP_POST_MAX_ATTEMPTS} 次）: {last_error}"
        ) from last_error

    assert resp is not None

    if resp.status_code >= 400:
        err_body = resp.text
        try:
            err_json = resp.json()
        except ValueError:
            raise RuntimeError(f"pipeline_server HTTP {resp.status_code}: {err_body}")
        if isinstance(err_json, dict) and err_json.get("error") is not None:
            typ = err_json.get("type", "")
            suf = f" [{typ}]" if typ else ""
            raise RuntimeError(
                f"pipeline_server HTTP {resp.status_code}{suf}: {err_json['error']}"
            )
        raise RuntimeError(f"pipeline_server HTTP {resp.status_code}: {err_body}")

    if not resp.content:
        return {}
    try:
        return resp.json()
    except ValueError as e:
        raise RuntimeError(f"pipeline_server 返回非 JSON: {resp.content[:500]!r}") from e


def _post_multipart_pipeline_infer(
    url: str,
    image_bytes: bytes,
    filename: str,
    timeout_s: float,
    extra_form: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """multipart/form-data：image 必选；/infer 另需 dpi、slide_type、task 等 Form 字段。"""
    if not url.lower().startswith("http"):
        url = f"http://{url}"

    # requests 自动生成 boundary 并设置 Content-Type；普通字段走 data，文件走 files。
    data = {name: str(value) for name, value in extra_form.items()} if extra_form else None
    files = {"image": (filename, image_bytes, "image/jpeg")}
    try:
        resp = _get_pipeline_session().post(
            url,
            data=data,
            files=files,
            headers=headers or None,
            timeout=(PIPELINE_HTTP_CONNECT_TIMEOUT_S, timeout_s),
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"pipeline_server 请求失败: {e}") from e

    if resp.status_code >= 400:
        err_body = resp.text
        try:
            err_json = resp.json()
        except ValueError:
            raise RuntimeError(f"pipeline_server HTTP {resp.status_code}: {err_body}")
        if isinstance(err_json, dict) and err_json.get("error") is not None:
            typ = err_json.get("type", "")
            suf = f" [{typ}]" if typ else ""
            raise RuntimeError(
                f"pipeline_server HTTP {resp.status_code}{suf}: {err_json['error']}"
            )
        raise RuntimeError(f"pipeline_server HTTP {resp.status_code}: {err_body}")

    if not resp.content:
        return {}
    try:
        return resp.json()
    except ValueError as e:
        raise RuntimeError(f"pipeline_server 返回非 JSON: {resp.content[:500]!r}") from e


def _scalar_int(payload: dict[str, Any], *keys: str, default: int = 0) -> int:
    for k in keys:
        if k not in payload or payload[k] is None:
            continue
        v = payload[k]
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
            try:
                return int(np.asarray(v).reshape(-1)[0])
            except Exception:
                continue
        try:
            return int(v)
        except Exception:
            continue
    return default


def _as_float64_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def _res_get(res: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in res and res[k] is not None:
            return res[k]
    return None


def _pipeline_task_block(res: dict[str, Any], task: str) -> dict[str, Any]:
    block = res.get(task)
    return block if isinstance(block, dict) else {}


def _pipeline_task_det(res: dict[str, Any], task: str) -> dict[str, Any]:
    det = _pipeline_task_block(res, task).get("det")
    return det if isinstance(det, dict) else {}


def _pipeline_task_cls(res: dict[str, Any], task: str) -> dict[str, Any]:
    cls = _pipeline_task_block(res, task).get("cls")
    return cls if isinstance(cls, dict) else {}


def _xyxy_boxes_and_scores(det: dict[str, Any]) -> tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """从 task.det 提取 xyxy bboxes 与 scores，返回 (boxes, scores, count)。"""
    if not det:
        return None, None, 0
    boxes = _as_float64_array(det.get("bboxes"))
    if boxes is None or boxes.size == 0:
        return None, None, 0
    if boxes.ndim == 1 and boxes.size % 4 == 0:
        boxes = boxes.reshape(-1, 4)
    elif boxes.ndim < 2:
        return None, None, 0
    n = int(boxes.shape[0])
    scores = _as_float64_array(det.get("scores"))
    if scores is not None and scores.size >= n:
        scores = scores.flatten()[:n]
    else:
        scores = np.ones(n, dtype=np.float64)
    return boxes, scores, n


def _xyxy_to_xywh_rows(boxes: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    out = b.copy()
    out[:, 2] = b[:, 2] - b[:, 0]
    out[:, 3] = b[:, 3] - b[:, 1]
    return out


def _cls_head_field(cls_block: dict[str, Any], head: str, field: str) -> Optional[np.ndarray]:
    if not isinstance(cls_block, dict):
        return None
    head_block = cls_block.get(head)
    if not isinstance(head_block, dict):
        return None
    return _as_float64_array(head_block.get(field))


def _nested_cellularity_counts(res: dict[str, Any]) -> tuple[int, int]:
    block = _pipeline_task_block(res, "cellularity")
    wbc = block.get("wbc") if isinstance(block.get("wbc"), dict) else {}
    rbc = block.get("rbc") if isinstance(block.get("rbc"), dict) else {}
    wpc = _scalar_int(wbc, "count", default=-1)
    rpc = _scalar_int(rbc, "count", default=-1)
    if wpc < 0:
        wpc = _scalar_int(
            res,
            "cell_analysis_wbc_pixel_count",
            "wbc_pixel_count",
            "CELL_ANALYSIS_WBC_PIXEL_COUNT",
        )
    if rpc < 0:
        rpc = _scalar_int(
            res,
            "cell_analysis_red_pixel_count",
            "red_pixel_count",
            "CELL_ANALYSIS_RED_PIXEL_COUNT",
        )
    return wpc, rpc


def _parse_pipeline_json_147246(res: dict[str, Any]) -> tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    int,
    Optional[np.ndarray],
    Optional[np.ndarray],
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    """解析 147246 pipeline：wbc/meg.det（xyxy）、cscore.det/cls；兼容旧 flat 字段。"""
    if res.get("error"):
        raise RuntimeError(str(res.get("error")))

    wbc_boxes, wbc_scores, wbc_num = _xyxy_boxes_and_scores(_pipeline_task_det(res, "wbc"))
    meg_boxes, meg_scores, meg_num = _xyxy_boxes_and_scores(_pipeline_task_det(res, "meg"))

    if wbc_num <= 0:
        wbc_raw = _res_get(res, "wbc_detections", "WBC_DETECTIONS")
        wbc_arr = _as_float64_array(wbc_raw)
        if wbc_arr is not None and wbc_arr.size > 0:
            wbc_num = _scalar_int(res, "wbc_num", "WBC_NUM_DETECTIONS", default=len(wbc_arr))
            wbc_num = min(wbc_num, len(wbc_arr))
            wbc_boxes = wbc_arr[:wbc_num, :4] if wbc_arr.ndim >= 2 else wbc_arr.reshape(-1, 4)[:wbc_num]
            wbc_scores = (
                wbc_arr[:wbc_num, 4].flatten()
                if wbc_arr.ndim >= 2 and wbc_arr.shape[1] > 4
                else np.ones(wbc_num, dtype=np.float64)
            )

    if meg_num <= 0:
        meg_raw = _res_get(res, "meg_detections", "MEG_DETECTIONS")
        meg_arr = _as_float64_array(meg_raw)
        if meg_arr is not None and meg_arr.size > 0:
            meg_num = _scalar_int(res, "meg_num", "MEG_NUM_DETECTIONS", default=len(meg_arr))
            meg_num = min(meg_num, len(meg_arr))
            meg_boxes = meg_arr[:meg_num, :4] if meg_arr.ndim >= 2 else meg_arr.reshape(-1, 4)[:meg_num]
            meg_scores = (
                meg_arr[:meg_num, 4].flatten()
                if meg_arr.ndim >= 2 and meg_arr.shape[1] > 4
                else np.ones(meg_num, dtype=np.float64)
            )

    cscore_det = _pipeline_task_det(res, "cscore")
    cscore_cls = _pipeline_task_cls(res, "cscore")
    cscore_boxes, _, cscore_n = _xyxy_boxes_and_scores(cscore_det)
    if cscore_n > 0 and cscore_boxes is not None:
        # cscore.det.bboxes 为 xyxy；下游 meta["scores"] 约定 [lx, ly, w, h, score, grade]
        regions = _xyxy_to_xywh_rows(cscore_boxes)
        grades_a = np.asarray(cscore_cls.get("tops") if cscore_cls else [], dtype=np.float64).flatten()
        scores_a = np.asarray(cscore_cls.get("scores") if cscore_cls else [], dtype=np.float64).flatten()
    else:
        cr = _res_get(res, "constituency_regions", "CONSTITUENCY_REGIONS") or []
        cs = _res_get(res, "constituency_scores", "CONSTITUENCY_SCORES")
        cg = _res_get(res, "constituency_grades", "CONSTITUENCY_GRADES")
        regions = (
            np.asarray(cr, dtype=np.float64).reshape(-1, 4)
            if np.asarray(cr).size > 0
            else np.zeros((0, 4), dtype=np.float64)
        )
        scores_a = np.asarray(cs if cs is not None else [], dtype=np.float64).flatten()
        grades_a = np.asarray(cg if cg is not None else [], dtype=np.float64).flatten()

    wpc, rpc = _nested_cellularity_counts(res)
    return (
        wbc_boxes,
        wbc_scores,
        wbc_num,
        meg_boxes,
        meg_scores,
        meg_num,
        regions,
        scores_a,
        grades_a,
        wpc,
        rpc,
    )


def _infer_147246_finalize(
    algorithm_types: str,
    wbc_boxes: Optional[np.ndarray],
    wbc_scores: Optional[np.ndarray],
    wbc_num: int,
    meg_boxes: Optional[np.ndarray],
    meg_scores: Optional[np.ndarray],
    meg_num: int,
    con_regions: np.ndarray,
    con_scores: np.ndarray,
    con_grades: np.ndarray,
    wbc_pixel_count: int,
    red_pixel_count: int,
    smear_type: str,
) -> dict[str, Any]:
    regions = np.asarray(con_regions, dtype=np.float64)
    if regions.size:
        regions = regions.reshape(-1, 4)
    else:
        regions = np.zeros((0, 4), dtype=np.float64)
    n = regions.shape[0]
    scores_arr = np.asarray(con_scores, dtype=np.float64).flatten()
    grades_arr = np.asarray(con_grades, dtype=np.float64).flatten()
    if n > 0:
        if scores_arr.size < n:
            scores_arr = np.resize(scores_arr, n)
        if grades_arr.size < n:
            grades_arr = np.resize(grades_arr, n)
        constituency_scores_combined = np.column_stack([regions, scores_arr[:n], grades_arr[:n]]).tolist()
    else:
        constituency_scores_combined = []

    cells: List[Cell] = []
    if "WBC" in (algorithm_types or ""):
        if wbc_num > 0 and wbc_boxes is not None:
            s = (
                np.asarray(wbc_scores, dtype=np.float64).flatten()[:wbc_num]
                if wbc_scores is not None
                else np.ones(wbc_num, dtype=np.float64)
            )
            cells.extend(
                _boxes_xyxy_to_cells(
                    wbc_boxes[:wbc_num],
                    s,
                    np.zeros(wbc_num, dtype=np.int32),
                    100000,
                    ["unclassified"],
                    CELL_TYPES_X40,
                )
            )
    if "MEG" in (algorithm_types or ""):
        if meg_num > 0 and meg_boxes is not None:
            s = (
                np.asarray(meg_scores, dtype=np.float64).flatten()[:meg_num]
                if meg_scores is not None
                else np.ones(meg_num, dtype=np.float64)
            )
            cells.extend(
                _boxes_xyxy_to_cells(
                    meg_boxes[:meg_num],
                    s,
                    np.zeros(meg_num, dtype=np.int32),
                    100001,
                    ["unclassified"],
                    CELL_TYPES_X40,
                )
            )
    cell_list = _cells_to_cell_list_single(cells, smear_type)
    return {
        "cells": cells,
        "scores": constituency_scores_combined,
        "cell_list": cell_list,
        "wbc_pixel_count": wbc_pixel_count,
        "red_pixel_count": red_pixel_count,
    }


def _infer_357378_from_pipeline_json(res: dict[str, Any], smear_type: str) -> dict[str, Any]:
    """357378：meg.det（xyxy）+ meg.cls.tops/scores；兼容旧 flat boxes/scores。"""
    if res.get("error"):
        raise RuntimeError(str(res.get("error")))

    meg_det = _pipeline_task_det(res, "meg")
    meg_cls = _pipeline_task_cls(res, "meg")
    boxes, scores, num_det = _xyxy_boxes_and_scores(meg_det)

    if num_det <= 0:
        boxes_raw = _res_get(res, "boxes", "BOXES")
        boxes = _as_float64_array(boxes_raw)
        if boxes is None or boxes.size == 0:
            return {"cells": [], "scores": [], "cell_list": []}
        if boxes.ndim == 1 and boxes.size % 4 == 0:
            boxes = boxes.reshape(-1, 4)
        elif boxes.ndim < 2:
            return {"cells": [], "scores": [], "cell_list": []}
        navail = int(boxes.shape[0])
        num_det = _scalar_int(res, "num_detections", "NUM_DETECTIONS", default=navail)
        num_det = min(num_det, navail)
        if num_det <= 0:
            return {"cells": [], "scores": [], "cell_list": []}
        boxes = boxes[:num_det]
        scores_raw = _res_get(res, "scores", "SCORES")
        scores = (
            np.asarray(scores_raw, dtype=np.float64)[:num_det]
            if scores_raw is not None
            else np.ones(num_det, dtype=np.float64)
        )
        class_ids_raw = _res_get(res, "class_ids", "CLASS_IDS")
        class_probs_raw = _res_get(res, "class_probs", "CLASS_PROBS")
        class_ids = (
            np.asarray(class_ids_raw, dtype=np.int32)[:num_det]
            if class_ids_raw is not None
            else np.zeros(num_det, dtype=np.int32)
        )
        class_probs = np.asarray(class_probs_raw, dtype=np.float64) if class_probs_raw is not None else None
    else:
        tops_raw = meg_cls.get("tops") if meg_cls else None
        probs_raw = meg_cls.get("scores") if meg_cls else None
        if tops_raw is not None:
            class_ids = np.asarray(tops_raw, dtype=np.int32)
            if class_ids.ndim == 1:
                class_ids = class_ids.reshape(-1, 1)
            class_ids = class_ids[:num_det]
        else:
            class_ids = np.zeros((num_det, 1), dtype=np.int32)
        class_probs = np.asarray(probs_raw, dtype=np.float64) if probs_raw is not None else None
        if class_probs is not None:
            class_probs = class_probs.reshape(num_det, -1)[:num_det]

    cells = _boxes_xyxy_to_cells(
        boxes[:num_det],
        scores[:num_det],
        class_ids,
        300000,
        X50_CLASS_NAMES,
        CELL_TYPES_MEG,
        class_probs=class_probs,
    )
    scores_out = np.asarray(scores[:num_det]).flatten().tolist()
    cids = np.asarray(class_ids).reshape(num_det, -1)
    cprobs = (
        np.asarray(class_probs).reshape(num_det, -1)
        if class_probs is not None
        else np.ones((num_det, 5))
    )
    cell_list = _cells_to_cell_list_top5(cells, cids, cprobs, 300000, CELL_TYPES_MEG, X50_CLASS_NAMES, smear_type)
    return {"cells": cells, "scores": scores_out, "cell_list": cell_list}


def _infer_csf_from_pipeline_json(res: dict[str, Any], smear_type: str) -> dict[str, Any]:
    """35000 / 71000：wbc.det（xyxy）；统一映射为 100007 未分类脑脊液细胞。"""
    if res.get("error"):
        raise RuntimeError(str(res.get("error")))

    boxes, scores, num_det = _xyxy_boxes_and_scores(_pipeline_task_det(res, "wbc"))
    if num_det <= 0:
        boxes_raw = _res_get(res, "boxes", "BOXES")
        boxes = _as_float64_array(boxes_raw)
        if boxes is None or boxes.size == 0:
            return {"cells": [], "scores": [], "cell_list": []}
        if boxes.ndim == 1 and boxes.size % 4 == 0:
            boxes = boxes.reshape(-1, 4)
        elif boxes.ndim < 2:
            return {"cells": [], "scores": [], "cell_list": []}
        navail = int(boxes.shape[0])
        num_det = _scalar_int(res, "num_detections", "NUM_DETECTIONS", default=navail)
        num_det = min(num_det, navail)
        if num_det <= 0:
            return {"cells": [], "scores": [], "cell_list": []}
        boxes = boxes[:num_det]
        scores_raw = _res_get(res, "scores", "SCORES")
        scores = (
            np.asarray(scores_raw, dtype=np.float64)[:num_det]
            if scores_raw is not None
            else np.ones(num_det, dtype=np.float64)
        )

    class_ids = np.zeros(num_det, dtype=np.int32)
    cells = _boxes_xyxy_to_cells(
        boxes[:num_det],
        scores[:num_det],
        class_ids,
        CSF_UNCLASSIFIED_CELL_TYPE,
        ["CSF"],
        CELL_TYPES_X40,
    )
    scores_out = np.asarray(scores[:num_det]).flatten().tolist()
    cell_list = _cells_to_cell_list_single(cells, smear_type)
    return {"cells": cells, "scores": scores_out, "cell_list": cell_list}


def _prepare_xywh_detections(det_raw: Any, num_det: int) -> tuple[Optional[np.ndarray], int]:
    det = _as_float64_array(det_raw)
    if det is None or det.size == 0:
        return None, 0
    boxes = det.reshape(-1, det.shape[-1]) if det.ndim >= 2 else det.reshape(1, -1)
    if num_det <= 0:
        num_det = len(boxes)
    return boxes, min(num_det, len(boxes))


def _flatten_det_scores(scores_raw: Any) -> Optional[np.ndarray]:
    scores = _as_float64_array(scores_raw)
    if scores is None:
        return None
    return scores.flatten()


def _array_at(arr: Optional[np.ndarray], i: int) -> Optional[float]:
    if arr is None:
        return None
    flat = np.asarray(arr).flatten()
    if i < 0 or i >= len(flat):
        return None
    return float(flat[i])


def _cell_type_name(cell_type: int, default: str) -> str:
    type_info = CELL_TYPES_X40.get(cell_type)
    if type_info and isinstance(type_info, (tuple, list)):
        return type_info[1]
    return default


def _cells_from_xywh_detections(
    boxes: np.ndarray,
    count: int,
    det_scores: Optional[np.ndarray],
    cell_type: int,
    default_type_name: str,
    extra_builder: Optional[Callable[[int], dict[str, Any]]] = None,
) -> List[Cell]:
    type_name = _cell_type_name(cell_type, default_type_name)
    cells: List[Cell] = []
    for i in range(count):
        row = np.asarray(boxes[i]).flatten()
        if len(row) < 4:
            continue
        x, y, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        det_score = float(det_scores[i]) if det_scores is not None and i < len(det_scores) else 1.0
        extra = extra_builder(i) if extra_builder else {}
        cells.append(Cell(
            cell_xmin=int(x),
            cell_ymin=int(y),
            cell_xmax=int(x + w),
            cell_ymax=int(y + h),
            cell_type=cell_type,
            cell_type_name=type_name,
            class_confidence=det_score,
            bbox_confidence=det_score,
            extra=extra,
        ))
    return cells


def _red_label_name(inv: List[str], labels: dict[str, str], class_id: int) -> tuple[str, str]:
    label_id = inv[class_id] if 0 <= class_id < len(inv) else str(class_id)
    return label_id, labels.get(label_id, label_id)


def _append_red_abnormality_extra(
    extra: dict[str, Any],
    key: str,
    inv: List[str],
    labels: dict[str, str],
    class_arr: Optional[np.ndarray],
    prob_arr: Optional[np.ndarray],
    i: int,
) -> None:
    class_raw = _array_at(class_arr, i)
    if class_raw is None:
        return
    class_id = int(class_raw)
    # if class_id == 0:
    #     return
    label_id, name = _red_label_name(inv, labels, class_id)
    prob = _array_at(prob_arr, i)
    extra[key] = {
        "name": name,
        "label_id": label_id,
        "confidence": float(prob) if prob is not None else None,
    }


def _append_plat_abnormality_extra(
    extra: dict[str, Any],
    key: str,
    labels: dict[str, str],
    class_arr: Optional[np.ndarray],
    prob_arr: Optional[np.ndarray],
    i: int,
) -> None:
    """血小板分类 head：class_id 即 label_id，无需 RED 式 label 映射。"""
    class_raw = _array_at(class_arr, i)
    if class_raw is None:
        return
    label_id = str(int(class_raw))
    prob = _array_at(prob_arr, i)
    extra[key] = {
        "name": labels.get(label_id, label_id),
        "label_id": label_id,
        "confidence": float(prob) if prob is not None else None,
    }


def _classify_red_size_label_id(w_px: float, h_px: float, dpi: int) -> tuple[str, str]:
    """根据框宽高（像素）与 DPI 换算为微米后，按 (w+h)/2 判定大小异常。"""
    um_px = um_per_pixel_from_dpi(dpi)
    avg_um = (w_px * um_px + h_px * um_px) / 2.0
    if avg_um < 6:
        label_id = "20"
    elif avg_um < 10:
        label_id = "1"
    elif avg_um <= 15:
        label_id = "19"
    else:
        label_id = "21"
    return label_id, RED_SIZE_LABELS[label_id]


def _append_red_size_extra(extra: dict[str, Any], w_px: float, h_px: float, dpi: int) -> None:
    label_id, name = _classify_red_size_label_id(w_px, h_px, dpi)
    extra["SIZE"] = {
        "name": name,
        "label_id": label_id,
        "confidence": 1.0,
    }


def _classify_plat_sez_label_id(w_px: float, h_px: float, dpi: int) -> tuple[str, str]:
    """根据框宽高（像素）与 DPI 换算为微米后，按 (w+h)/2 判定血小板大小异常。"""
    um_px = um_per_pixel_from_dpi(dpi)
    avg_um = (w_px * um_px + h_px * um_px) / 2.0
    if avg_um < 2:
        label_id = "2"
    elif avg_um < 5:
        label_id = "0"
    elif avg_um <= 8:
        label_id = "1"
    else:
        label_id = "3"
    return label_id, PLAT_SEZ_LABELS[label_id]


def _append_plat_sez_extra(extra: dict[str, Any], w_px: float, h_px: float, dpi: int) -> None:
    label_id, name = _classify_plat_sez_label_id(w_px, h_px, dpi)
    extra["SIZE"] = {
        "name": name,
        "label_id": label_id,
        "confidence": 1.0,
    }


def _build_plat_extra(
    i: int,
    morph: Optional[np.ndarray],
    morph_prob: Optional[np.ndarray],
    dist: Optional[np.ndarray],
    dist_prob: Optional[np.ndarray],
    color: Optional[np.ndarray],
    color_prob: Optional[np.ndarray],
    *,
    w_px: float = 0.0,
    h_px: float = 0.0,
    dpi: int = DPI_714756,
) -> dict[str, Any]:
    """写入 extra：形态/分布/颜色及对应置信度；大小异常由框尺寸与 DPI 计算。"""
    extra: dict[str, Any] = {}
    _append_plat_abnormality_extra(extra, "MORPH", PLAT_MORPH_LABELS, morph, morph_prob, i)
    _append_plat_abnormality_extra(extra, "DIST", PLAT_DIST_LABELS, dist, dist_prob, i)
    _append_plat_abnormality_extra(extra, "COLOR", PLAT_COLOR_LABELS, color, color_prob, i)
    if w_px > 0 and h_px > 0:
        _append_plat_sez_extra(extra, w_px, h_px, dpi)
    return extra


def _build_red_rbc_extra(
    i: int,
    struct: Optional[np.ndarray],
    struct_prob: Optional[np.ndarray],
    color: Optional[np.ndarray],
    color_prob: Optional[np.ndarray],
    morph: Optional[np.ndarray],
    morph_prob: Optional[np.ndarray],
    agg: Optional[np.ndarray],
    agg_prob: Optional[np.ndarray],
    *,
    w_px: float = 0.0,
    h_px: float = 0.0,
    dpi: int = DPI_714756,
) -> dict[str, Any]:
    """写入 extra：结构/颜色/形态/聚集及对应置信度；大小异常由框尺寸与 DPI 计算。"""
    extra: dict[str, Any] = {}
    # 结构异常
    _append_red_abnormality_extra(extra, "STRUCT", RED_STRUCT_INV, RED_STRUCT_LABELS, struct, struct_prob, i)
    # 颜色异常
    _append_red_abnormality_extra(extra, "COLOR", RED_COLOR_INV, RED_COLOR_LABELS, color, color_prob, i)
    # 形态异常
    _append_red_abnormality_extra(extra, "MORPH", RED_MORPH_INV, RED_MORPH_LABELS, morph, morph_prob, i)
    # 聚集异常
    _append_red_abnormality_extra(extra, "AGG", RED_AGG_INV, RED_AGG_LABELS, agg, agg_prob, i)
    # 大小异常（规则计算，置信度固定为 1）
    if w_px > 0 and h_px > 0:
        _append_red_size_extra(extra, w_px, h_px, dpi)
    return extra


def _rbc_head_label_to_cell_type(_key: str, label_id: str) -> Optional[int]:
    """红细胞 label_id → CELL_TYPE_RBC；"1" 为无异常。"""
    try:
        lid = int(label_id)
    except (TypeError, ValueError):
        return None
    cell_type = 500000 if lid == 1 else 500000 + lid
    return cell_type if cell_type in CELL_TYPE_RBC else None


def _plat_head_label_to_cell_type(key: str, label_id: str) -> Optional[int]:
    cell_type = _PLAT_HEAD_LABEL_TO_TYPE.get((key, str(label_id)))
    if cell_type is None or cell_type not in CELL_TYPE_PLT:
        return None
    return cell_type


def _top_item_from_abnormality(
    cell_type: int,
    class_confidence: float,
    bbox_confidence: float,
    smear_type: str,
) -> dict[str, Any]:
    return {
        "cell_type": cell_type,
        "class_confidence": float(class_confidence),
        "bbox_confidence": float(bbox_confidence),
        "count_type": get_counting_cell_type(cell_type, smear_type),
    }


def _tops_from_abnormal_extra(
    extra: dict[str, Any],
    bbox_confidence: float,
    smear_type: str,
    *,
    priority_keys: tuple[str, ...],
    normal_label_ids: frozenset[str],
    label_to_cell_type: Callable[[str, str], Optional[int]],
    normal_cell_type: int,
) -> list[dict[str, Any]]:
    """按优先级收集异常分类，最多 5 项；全无异常则唯一项为正常类型。"""
    tops: list[dict[str, Any]] = []
    for key in priority_keys:
        info = extra.get(key)
        if not isinstance(info, dict):
            continue
        label_id = str(info.get("label_id", "")).strip()
        if not label_id or label_id in normal_label_ids:
            continue
        cell_type = label_to_cell_type(key, label_id)
        if cell_type is None or cell_type == normal_cell_type:
            continue
        conf = info.get("confidence")
        tops.append(_top_item_from_abnormality(
            cell_type,
            float(conf) if conf is not None else 1.0,
            bbox_confidence,
            smear_type,
        ))
        if len(tops) >= 5:
            break
    if not tops:
        tops.append(_top_item_from_abnormality(normal_cell_type, 1.0, bbox_confidence, smear_type))
    return tops


def _finalize_abnormal_top5_cells(
    cells: List[Cell],
    smear_type: str,
    *,
    type_map: dict,
    normal_cell_type: int,
    priority_keys: tuple[str, ...],
    normal_label_ids: frozenset[str],
    label_to_cell_type: Callable[[str, str], Optional[int]],
) -> list[dict[str, Any]]:
    """根据 extra 生成 TOP5，并回写 cell_type / cell_type_name / class_confidence。"""
    out: list[dict[str, Any]] = []
    for c in cells:
        tops = _tops_from_abnormal_extra(
            c.extra or {},
            float(c.bbox_confidence),
            smear_type,
            priority_keys=priority_keys,
            normal_label_ids=normal_label_ids,
            label_to_cell_type=label_to_cell_type,
            normal_cell_type=normal_cell_type,
        )
        top0 = tops[0]
        c.cell_type = int(top0["cell_type"])
        type_info = type_map.get(c.cell_type)
        if type_info and isinstance(type_info, (tuple, list)):
            c.cell_type_name = str(type_info[1])
        c.class_confidence = float(top0["class_confidence"])
        item: dict[str, Any] = {
            "cell_xmin": c.cell_xmin,
            "cell_ymin": c.cell_ymin,
            "cell_xmax": c.cell_xmax,
            "cell_ymax": c.cell_ymax,
            "tops": tops,
        }
        if c.extra:
            item["extra"] = c.extra
        out.append(item)
    return out


def _infer_714756_bm_from_pipeline_json(
    res: dict[str, Any],
    smear_type: str,
    *,
    dpi: int = DPI_714756,
    resolved: Optional[ResolvedModels] = None,
) -> dict[str, Any]:
    # logger.info("714756_bm pipeline 原始返回:\n%s", res)
    if res.get("error"):
        raise RuntimeError(str(res.get("error")))

    wbc_det = _pipeline_task_det(res, "wbc")
    wbc_cls = _pipeline_task_cls(res, "wbc")
    rbc_det = _pipeline_task_det(res, "rbc")
    rbc_cls = _pipeline_task_cls(res, "rbc")
    plat_det = _pipeline_task_det(res, "plat")
    plat_cls = _pipeline_task_cls(res, "plat")

    boxes, scores, wbc_num = _xyxy_boxes_and_scores(wbc_det)
    wbc_class_ids_raw = wbc_cls.get("tops") if wbc_cls else None
    wbc_class_probs_raw = wbc_cls.get("scores") if wbc_cls else None

    rbc_boxes, red_scores_arr, red_num = _xyxy_boxes_and_scores(rbc_det)
    rd: Optional[np.ndarray] = _xyxy_to_xywh_rows(rbc_boxes) if rbc_boxes is not None and red_num > 0 else None
    red_scores = red_scores_arr

    plat_boxes, plat_scores_arr, plat_num = _xyxy_boxes_and_scores(plat_det)
    pd: Optional[np.ndarray] = _xyxy_to_xywh_rows(plat_boxes) if plat_boxes is not None and plat_num > 0 else None
    plat_scores = plat_scores_arr

    red_class_struct = _cls_head_field(rbc_cls, "struct", "tops")
    red_class_struct_prob = _cls_head_field(rbc_cls, "struct", "scores")
    red_class_color = _cls_head_field(rbc_cls, "color", "tops")
    red_class_color_prob = _cls_head_field(rbc_cls, "color", "scores")
    red_class_morph = _cls_head_field(rbc_cls, "morph", "tops")
    red_class_morph_prob = _cls_head_field(rbc_cls, "morph", "scores")
    red_class_agg = _cls_head_field(rbc_cls, "agg", "tops")
    red_class_agg_prob = _cls_head_field(rbc_cls, "agg", "scores")

    plat_class_morph = _cls_head_field(plat_cls, "morph", "tops")
    plat_class_morph_prob = _cls_head_field(plat_cls, "morph", "scores")
    plat_class_dist = _cls_head_field(plat_cls, "dist", "tops")
    plat_class_dist_prob = _cls_head_field(plat_cls, "dist", "scores")
    plat_class_color = _cls_head_field(plat_cls, "color", "tops")
    plat_class_color_prob = _cls_head_field(plat_cls, "color", "scores")

    if wbc_num <= 0:
        boxes_raw = _res_get(res, "boxes", "BOXES")
        scores_raw = _res_get(res, "scores", "SCORES")
        class_ids_raw = _res_get(res, "class_ids", "CLASS_IDS")
        wbc_num = _scalar_int(res, "num_detections", "NUM_DETECTIONS", default=-1)
        boxes = _as_float64_array(boxes_raw)
        if boxes is None or boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float64)
            wbc_num = 0
        else:
            if boxes.ndim == 1 and boxes.size % 4 == 0:
                boxes = boxes.reshape(-1, 4)
            elif boxes.ndim < 2:
                boxes = np.zeros((0, 4), dtype=np.float64)
                wbc_num = 0
            else:
                nbox = int(boxes.shape[0])
                wbc_num = nbox if wbc_num < 0 else min(wbc_num, nbox)
        scores = _as_float64_array(scores_raw)
        wbc_class_ids_raw = class_ids_raw
        wbc_class_probs_raw = _res_get(res, "class_probs", "CLASS_PROBS")
    else:
        class_ids_raw = wbc_class_ids_raw

    if red_num <= 0:
        red_num = _scalar_int(res, "red_num_detections", "RED_NUM_DETECTIONS", "red_num", default=-1)
        rd, red_num = _prepare_xywh_detections(_res_get(res, "red_detections", "RED_DETECTIONS"), red_num)
        red_scores = _flatten_det_scores(_res_get(res, "red_det_scores", "RED_DET_SCORES"))
        if red_class_struct is None:
            red_class_struct = _as_float64_array(_res_get(res, "red_class_struct", "RED_CLASS_STRUCT"))
            red_class_struct_prob = _as_float64_array(_res_get(res, "red_class_struct_prob", "RED_CLASS_STRUCT_PROB"))
            red_class_color = _as_float64_array(_res_get(res, "red_class_color", "RED_CLASS_COLOR"))
            red_class_color_prob = _as_float64_array(_res_get(res, "red_class_color_prob", "RED_CLASS_COLOR_PROB"))
            red_class_morph = _as_float64_array(_res_get(res, "red_class_morph", "RED_CLASS_MORPH"))
            red_class_morph_prob = _as_float64_array(_res_get(res, "red_class_morph_prob", "RED_CLASS_MORPH_PROB"))
            red_class_agg = _as_float64_array(_res_get(res, "red_class_agg", "RED_CLASS_AGG"))
            red_class_agg_prob = _as_float64_array(_res_get(res, "red_class_agg_prob", "RED_CLASS_AGG_PROB"))

    if plat_num <= 0:
        plat_num = _scalar_int(res, "plat_num_detections", "PLAT_NUM_DETECTIONS", "plat_num", default=-1)
        pd, plat_num = _prepare_xywh_detections(_res_get(res, "plat_detections", "PLAT_DETECTIONS"), plat_num)
        plat_scores = _flatten_det_scores(_res_get(res, "plat_det_scores", "PLAT_DET_SCORES"))
        if plat_class_morph is None:
            plat_class_morph = _as_float64_array(_res_get(res, "plat_class_morph", "PLAT_CLASS_MORPH"))
            plat_class_morph_prob = _as_float64_array(_res_get(res, "plat_class_morph_prob", "PLAT_CLASS_MORPH_PROB"))
            plat_class_dist = _as_float64_array(_res_get(res, "plat_class_dist", "PLAT_CLASS_DIST"))
            plat_class_dist_prob = _as_float64_array(_res_get(res, "plat_class_dist_prob", "PLAT_CLASS_DIST_PROB"))
            plat_class_color = _as_float64_array(_res_get(res, "plat_class_color", "PLAT_CLASS_COLOR"))
            plat_class_color_prob = _as_float64_array(_res_get(res, "plat_class_color_prob", "PLAT_CLASS_COLOR_PROB"))

    class_ids = np.asarray(class_ids_raw, dtype=np.int32) if class_ids_raw is not None else None

    cells: List[Cell] = []
    scores_out: List[float] = []
    cell_list: List[Any] = []

    st = normalize_smear_type(smear_type)
    classify_wbc = resolved.has_classifier_for("WBC") if resolved is not None else True
    classify_rbc = resolved.has_classifier_for("RBC") if resolved is not None else True
    classify_plat = resolved.has_classifier_for("PLAT") if resolved is not None else True

    if wbc_num > 0 and boxes is not None:
        b = boxes[:wbc_num]
        s = scores[:wbc_num] if scores is not None and scores.size >= wbc_num else np.ones(wbc_num)
        c = (
            class_ids[:wbc_num]
            if class_ids is not None and class_ids.shape[0] >= wbc_num
            else np.zeros(wbc_num, dtype=np.int32)
        )
        cprobs_arr = (
            np.asarray(wbc_class_probs_raw, dtype=np.float64)[:wbc_num].reshape(wbc_num, -1)
            if wbc_class_probs_raw is not None
            else None
        )
        if classify_wbc and st == "CSF":
            csf_names = [CELL_TYPE_CSF.get(400000 + i, ("?", f"cell_{i}"))[1] for i in range(12)]
            wbc_cells = _boxes_xyxy_to_cells(
                b, s, c, 400000, csf_names, CELL_TYPE_CSF, class_probs=cprobs_arr
            )
            cells.extend(wbc_cells)
            scores_out.extend(np.asarray(s).flatten().tolist())
            cids_arr = np.asarray(c, dtype=np.int32).reshape(wbc_num, -1)
            cprobs_for_top5 = cprobs_arr if cprobs_arr is not None else np.ones((wbc_num, 5))
            cell_list.extend(
                _cells_to_cell_list_top5(
                    wbc_cells, cids_arr, cprobs_for_top5, 400000, CELL_TYPE_CSF, csf_names, smear_type
                )
            )
        elif classify_wbc:
            wbc_names = [CELL_TYPES_X100.get(200000 + i, ("?", f"cell_{i}"))[1] for i in range(35)]
            wbc_cells = _boxes_xyxy_to_cells(
                b, s, c, 200000, wbc_names, CELL_TYPES_X100, class_probs=cprobs_arr
            )
            cells.extend(wbc_cells)
            scores_out.extend(np.asarray(s).flatten().tolist())
            cids_arr = np.asarray(c, dtype=np.int32).reshape(wbc_num, -1)
            cprobs_for_top5 = cprobs_arr if cprobs_arr is not None else np.ones((wbc_num, 5))
            cell_list.extend(
                _cells_to_cell_list_top5(
                    wbc_cells, cids_arr, cprobs_for_top5, 200000, CELL_TYPES_X100, wbc_names, smear_type
                )
            )
        else:
            uncls_type = 100007 if st == "CSF" else 100000
            wbc_cells = _boxes_xyxy_to_cells(
                b, s, np.zeros(wbc_num, dtype=np.int32), uncls_type, ["unclassified"], CELL_TYPES_X40
            )
            cells.extend(wbc_cells)
            scores_out.extend(np.asarray(s).flatten().tolist())
            cell_list.extend(_cells_to_cell_list_single(wbc_cells, smear_type))

    if red_num > 0 and rd is not None:

        def _red_extra(i: int) -> dict[str, Any]:
            row = np.asarray(rd[i]).flatten()
            w_px = float(row[2]) if len(row) >= 4 else 0.0
            h_px = float(row[3]) if len(row) >= 4 else 0.0
            return _build_red_rbc_extra(
                i,
                red_class_struct,
                red_class_struct_prob,
                red_class_color,
                red_class_color_prob,
                red_class_morph,
                red_class_morph_prob,
                red_class_agg,
                red_class_agg_prob,
                w_px=w_px,
                h_px=h_px,
                dpi=dpi,
            )

        if classify_rbc:
            rbc_cells = _cells_from_xywh_detections(
                rd, red_num, red_scores, 100005, "已分类红细胞", extra_builder=_red_extra,
            )
            rbc_list = _finalize_abnormal_top5_cells(
                rbc_cells,
                smear_type,
                type_map=CELL_TYPE_RBC,
                normal_cell_type=500000,
                priority_keys=_RED_TOP5_PRIORITY,
                normal_label_ids=_RED_NORMAL_LABEL_IDS,
                label_to_cell_type=_rbc_head_label_to_cell_type,
            )
            cells.extend(rbc_cells)
            scores_out.extend([c.bbox_confidence for c in rbc_cells])
            cell_list.extend(rbc_list)
        else:
            rbc_cells = _cells_from_xywh_detections(
                rd, red_num, red_scores, 100002, "未分类红细胞",
            )
            cells.extend(rbc_cells)
            scores_out.extend([c.bbox_confidence for c in rbc_cells])
            cell_list.extend(_cells_to_cell_list_single(rbc_cells, smear_type))

    if plat_num > 0 and pd is not None:

        def _plat_extra(i: int) -> dict[str, Any]:
            row = np.asarray(pd[i]).flatten()
            w_px = float(row[2]) if len(row) >= 4 else 0.0
            h_px = float(row[3]) if len(row) >= 4 else 0.0
            return _build_plat_extra(
                i,
                plat_class_morph,
                plat_class_morph_prob,
                plat_class_dist,
                plat_class_dist_prob,
                plat_class_color,
                plat_class_color_prob,
                w_px=w_px,
                h_px=h_px,
                dpi=dpi,
            )

        if classify_plat:
            plat_cells = _cells_from_xywh_detections(
                pd, plat_num, plat_scores, 100006, "血小板", extra_builder=_plat_extra,
            )
            plat_list = _finalize_abnormal_top5_cells(
                plat_cells,
                smear_type,
                type_map=CELL_TYPE_PLT,
                normal_cell_type=600000,
                priority_keys=_PLAT_TOP5_PRIORITY,
                normal_label_ids=_PLAT_NORMAL_LABEL_IDS,
                label_to_cell_type=_plat_head_label_to_cell_type,
            )
            cells.extend(plat_cells)
            scores_out.extend([c.bbox_confidence for c in plat_cells])
            cell_list.extend(plat_list)
        else:
            plat_cells = _cells_from_xywh_detections(
                pd, plat_num, plat_scores, 100004, "未分类血小板",
            )
            cells.extend(plat_cells)
            scores_out.extend([c.bbox_confidence for c in plat_cells])
            cell_list.extend(_cells_to_cell_list_single(plat_cells, smear_type))

    return {"cells": cells, "scores": scores_out, "cell_list": cell_list}


def _boxes_to_cells(boxes: Optional[np.ndarray], cell_type: int, type_name_map: dict = None) -> List[Cell]:
    """将 [x,y,w,h] 或 [x,y,w,h,conf] 转为 Cell 列表（xywh 格式）"""
    if boxes is None or boxes.size == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim < 2:
        return []
    boxes = boxes.reshape(-1, boxes.shape[-1])
    cells = []
    type_name = (type_name_map or CELL_TYPES_X40).get(cell_type)
    if type_name and isinstance(type_name, (tuple, list)):
        type_name = type_name[1]
    else:
        type_name = type_name or "未知"
    for box in boxes:
        if np.all(box[:4] == 0):
            continue
        x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        conf = float(box[4]) if len(box) > 4 else 1.0
        cells.append(Cell(
            cell_xmin=int(x),
            cell_ymin=int(y),
            cell_xmax=int(x + w),
            cell_ymax=int(y + h),
            cell_type=cell_type,
            cell_type_name=str(type_name),
            class_confidence=conf,
            bbox_confidence=1.0,
        ))
    return cells


def _boxes_xyxy_to_cells(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    cell_type_base: int,
    class_names: List[str],
    type_map: Optional[dict] = None,
    class_probs: Optional[np.ndarray] = None,
) -> List[Cell]:
    """
    将 x1,y1,x2,y2 格式的 boxes 转为 Cell 列表。
    - class_ids 支持 (N,) 或 (N,5)：2D 时取 [i,0] 为 top-1
    - class_probs 可选：有则用 [i,0] 作为 class_confidence，scores[i] 作为 bbox_confidence
    """
    cells = []
    scores_flat = np.asarray(scores).flatten()
    cids = np.asarray(class_ids)
    cprobs = np.asarray(class_probs) if class_probs is not None else None
    for i in range(len(boxes)):
        if len(boxes[i]) < 4:
            continue
        x1, y1, x2, y2 = float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            continue
        bbox_score = float(scores_flat[i]) if i < len(scores_flat) else 1.0
        if cids.ndim > 1 and cids.shape[1] > 0:
            cls_id = int(cids[i, 0])
        else:
            cls_id = int(cids.flat[i])
        if cprobs is not None and cprobs.size > 0:
            cprobs_2d = np.asarray(cprobs).reshape(len(boxes), -1)
            if i < len(cprobs_2d):
                if cids.ndim > 1 and cids.shape[1] > 0:
                    class_conf = float(cprobs_2d[i, 0])
                else:
                    cid = int(cids.flat[i])
                    n_cols = cprobs_2d.shape[1]
                    class_conf = float(cprobs_2d[i, cid]) if 0 <= cid < n_cols else float(cprobs_2d[i, 0])
            else:
                class_conf = bbox_score
        else:
            class_conf = bbox_score
        cell_type = cell_type_base + cls_id
        if type_map and cell_type in type_map:
            t = type_map[cell_type]
            type_name = t[1] if isinstance(t, (tuple, list)) else str(t)
        else:
            type_name = class_names[cls_id] if cls_id < len(class_names) else "分类不明/无法分类巨核细胞"
        cells.append(Cell(
            cell_xmin=int(x1),
            cell_ymin=int(y1),
            cell_xmax=int(x2),
            cell_ymax=int(y2),
            cell_type=cell_type,
            cell_type_name=type_name,
            class_confidence=class_conf,
            bbox_confidence=bbox_score,
        ))
    return cells


def _cells_to_cell_list_single(cells: List[Cell], smear_type: str) -> list:
    """无 TOP5 时：每个 cell 的 tops 只放一项；有 extra 异常信息时一并写出"""
    out: list[dict[str, Any]] = []
    for c in cells:
        item: dict[str, Any] = {
            "cell_xmin": c.cell_xmin,
            "cell_ymin": c.cell_ymin,
            "cell_xmax": c.cell_xmax,
            "cell_ymax": c.cell_ymax,
            "tops": [{
                "cell_type": c.cell_type,
                "count_type": get_counting_cell_type(c.cell_type, smear_type),
                "class_confidence": float(c.class_confidence),
                "bbox_confidence": float(c.bbox_confidence),
            }],
        }
        if c.extra:
            item["extra"] = c.extra
        out.append(item)
    return out


def _cells_to_cell_list_top5(
    cells: List[Cell],
    class_ids: np.ndarray,
    class_probs: np.ndarray,
    cell_type_base: int,
    type_map: dict,
    class_names: Optional[List[str]] = None,
    smear_type: str = "BM",
) -> list:
    """有 TOP5 时：每个 cell 的 tops 放最多 5 项。支持 (N,5) 或 (N,) + (N,C) 从 probs 派生 TOP5"""
    cids = np.asarray(class_ids)
    cprobs = np.asarray(class_probs)
    if cids.ndim == 1 and cprobs.ndim == 2 and cprobs.shape[1] > 1:
        n = len(cids)
        top5_ids = np.argsort(-cprobs, axis=1)[:, :5]
        cids = top5_ids
        cprobs = np.take_along_axis(cprobs, top5_ids, axis=1)
    elif cids.ndim == 1:
        cids = cids.reshape(-1, 1)
        cprobs = cprobs.reshape(-1, 1) if cprobs.size else np.ones((len(cids), 1))
    elif cprobs.ndim == 1:
        cprobs = cprobs.reshape(-1, 1)
    n_cells = min(len(cells), len(cids))
    out = []
    for i in range(n_cells):
        c = cells[i]
        ids_row = cids[i] if i < len(cids) else []
        probs_row = cprobs[i] if i < len(cprobs) else np.ones(min(5, len(ids_row)))
        k = min(5, len(ids_row), len(probs_row))
        tops = []
        for j in range(k):
            cls_id = int(ids_row[j]) if j < len(ids_row) else 0
            prob = float(probs_row[j]) if j < len(probs_row) else 1.0
            cell_type = cell_type_base + cls_id
            tops.append({
                "cell_type": cell_type,
                "class_confidence": prob,
                "bbox_confidence": float(c.bbox_confidence),
                "count_type": get_counting_cell_type(c.cell_type, smear_type),
            })
        out.append({
            "cell_xmin": c.cell_xmin,
            "cell_ymin": c.cell_ymin,
            "cell_xmax": c.cell_xmax,
            "cell_ymax": c.cell_ymax,
            "tops": tops,
        })
    return out


def _get_client(gpu_id: int = 0):
    """获取或创建指定 GPU 的 Triton gRPC 客户端。"""
    import tritonclient.grpc as grpcclient

    gid = int(gpu_id)
    with _triton_client_lock:
        client = _triton_clients.get(gid)
        if client is None:
            ep = get_triton_endpoint(gid)
            url = ep.get("url")
            client = grpcclient.InferenceServerClient(url=url)
            _triton_clients[gid] = client
        return client


def _resolve_triton_route(gpu_id: Optional[int] = None) -> tuple[int, dict]:
    """与滤镜路径一致：显式 gpu_id 定点，否则 next_triton_endpoint 轮询单卡。"""
    if gpu_id is not None:
        gid = int(gpu_id)
        return gid, get_triton_endpoint(gid)
    return next_triton_endpoint()


# 供 task_service 等模块在 infer 外做 ensure（与 infer 内路由一致）
resolve_triton_route = _resolve_triton_route


def _post_unified_pipeline_infer(
    image_bytes: bytes,
    filename: str,
    route_dpi: int,
    smear_type: str,
    algorithm_types: str,
    endpoint: dict,
) -> dict[str, Any]:
    """POST /infer：multipart 传 image + dpi(actual_dpi) + slide_type + task。"""
    url = _pipeline_infer_url(endpoint=endpoint)
    slide_type = normalize_smear_type(smear_type)
    task = _pipeline_task_from_algorithm_types(algorithm_types, route_dpi, smear_type)
    logger.debug(
        "unified /infer url=%s dpi=%s slide_type=%s task=%s",
        url,
        route_dpi,
        slide_type,
        task,
    )
    return _post_multipart_pipeline_infer(
        url,
        image_bytes,
        filename,
        PIPELINE_HTTP_TIMEOUT_S,
        extra_form={
            "dpi": str(route_dpi),
            "slide_type": slide_type,
            "task": task,
        },
    )


def infer(
    image_bytes: bytes,
    dpi: int,
    smear_type: str = "BM",
    algorithm_types: str = "",
    filename: str = "tile.jpg",
    gpu_id: Optional[int] = None,
) -> dict:
    """
    细胞检测推理。先 resolve_models 查 MODEL_TABLE，再 POST /infer（dpi=actual_dpi）。
    返回 {"cells", "scores", "cell_list"}。

    平扫 upload_image 依赖 create_task 的 warmup_model（load_models）；
    单张识别见 get_task_result_x100。
    gpu_id 未指定时经 next_triton_endpoint 轮询单卡选 endpoint。
    """
    resolved = resolve_models(dpi, smear_type, algorithm_types)
    warning = resolved.warning
    route_dpi = _infer_route_dpi(resolved)
    gpu_id, endpoint = _resolve_triton_route(gpu_id)

    if route_dpi is None or route_dpi not in _PIPELINE_DPI_VALUES:
        result = {"cells": [], "scores": [], "cell_list": []}
        if warning:
            result["warning"] = warning
        return result

    res_json = _post_unified_pipeline_infer(
        image_bytes,
        filename,
        route_dpi,
        smear_type,
        algorithm_types,
        endpoint,
    )

    if route_dpi == DPI_147246:
        wbc_boxes, wbc_scores, wbc_num, meg_boxes, meg_scores, meg_num, cr, cs, cg, wpc, rpc = (
            _parse_pipeline_json_147246(res_json)
        )
        result = _infer_147246_finalize(
            algorithm_types,
            wbc_boxes,
            wbc_scores,
            wbc_num,
            meg_boxes,
            meg_scores,
            meg_num,
            cr,
            cs,
            cg,
            wpc,
            rpc,
            smear_type,
        )
    elif route_dpi == DPI_357378:
        result = _infer_357378_from_pipeline_json(res_json, smear_type)
    elif route_dpi == DPI_714756:
        result = _infer_714756_bm_from_pipeline_json(
            res_json, smear_type, dpi=dpi, resolved=resolved
        )
    elif route_dpi in (DPI_35000, DPI_71000):
        result = _infer_csf_from_pipeline_json(res_json, smear_type)
    else:
        result = {"cells": [], "scores": [], "cell_list": []}

    if warning:
        result["warning"] = warning
    return result


def _pad_image_bytes_to_cell_analysis_size(image_bytes: bytes) -> bytes:
    """
    读取图片尺寸，居中补边至 2448(宽)x2048(高)，填充色 RGB(230, 230, 230)。
    已是目标尺寸则原样返回；任一边超出则先居中裁剪再补边。
    """
    with Image.open(BytesIO(image_bytes)) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w == _CELL_ANALYSIS_TARGET_W and h == _CELL_ANALYSIS_TARGET_H:
            return image_bytes

        if w > _CELL_ANALYSIS_TARGET_W or h > _CELL_ANALYSIS_TARGET_H:
            left = max(0, (w - _CELL_ANALYSIS_TARGET_W) // 2)
            top = max(0, (h - _CELL_ANALYSIS_TARGET_H) // 2)
            right = left + min(w, _CELL_ANALYSIS_TARGET_W)
            bottom = top + min(h, _CELL_ANALYSIS_TARGET_H)
            im = im.crop((left, top, right, bottom))
            w, h = im.size

        canvas = Image.new(
            "RGB",
            (_CELL_ANALYSIS_TARGET_W, _CELL_ANALYSIS_TARGET_H),
            _CELL_ANALYSIS_PAD_RGB,
        )
        offset_x = (_CELL_ANALYSIS_TARGET_W - w) // 2
        offset_y = (_CELL_ANALYSIS_TARGET_H - h) // 2
        canvas.paste(im, (offset_x, offset_y))

        buf = BytesIO()
        canvas.save(buf, format="JPEG", quality=95)
        return buf.getvalue()


def _parse_global_infer_tasks(raw: str) -> tuple[str | None, frozenset[str] | None]:
    tasks = {
        t.strip().lower()
        for t in str(raw or "").replace(";", ",").split(",")
        if t.strip()
    }
    if not tasks:
        return "task is required", None
    invalid = sorted(tasks - _VALID_GLOBAL_TASKS)
    if invalid:
        return f"invalid task {invalid}, allowed: roi, dir", None
    return None, frozenset(tasks)


def _parse_global_smear_type(raw: str) -> tuple[str | None, str | None]:
    smear_type = (raw or "").strip().upper()
    if smear_type not in _VALID_GLOBAL_SMEAR_TYPES:
        return f"smear_type must be BM or PB, got {raw!r}", None
    return None, smear_type


def format_global_infer_result(pipeline: dict[str, Any]) -> dict[str, Any]:
    """将 multi_pipeline /global/infer 原始响应整理为业务 result 字段。"""
    return {
        key: value
        for key, value in pipeline.items()
        if key not in _GLOBAL_INFER_SKIP_RESULT_KEYS
    }


def infer_global_image(
    image_bytes: bytes,
    smear_type: str,
    task: str,
    filename: str = "image.jpg",
    gpu_id: Optional[int] = None,
    client_seq: Optional[str] = None,
) -> dict[str, Any]:
    """
    全局图分析：multi_pipeline_server POST /global/infer。
    固定模型 GLOBAL-IMAGE-ANALYSIS（roi）与 GLOBAL-IMAGE-HEAD-DIR（dir）。
    """
    if not image_bytes:
        raise ValueError("empty image payload")

    err, parsed_smear = _parse_global_smear_type(smear_type)
    if err:
        raise ValueError(err)
    err, tasks = _parse_global_infer_tasks(task)
    if err:
        raise ValueError(err)
    assert parsed_smear is not None and tasks is not None

    gpu_id, endpoint = _resolve_triton_route(gpu_id)
    model_names = [_GLOBAL_TASK_MODELS[t] for t in sorted(tasks)]
    for model_name in model_names:
        ok, msg = ensure_model_loaded(model_name, gpu_id=gpu_id)
        if not ok:
            raise RuntimeError(msg)

    url = _global_pipeline_infer_url(endpoint=endpoint)
    task_str = ",".join(sorted(tasks))
    headers = {"x-client-seq": client_seq} if client_seq else None
    logger.info(
        "infer_global_image route gpu_id=%s smear_type=%s task=%s models=%s client_seq=%s url=%s",
        gpu_id,
        parsed_smear,
        task_str,
        model_names,
        client_seq or "-",
        url,
    )
    return _post_multipart_pipeline_infer(
        url,
        image_bytes,
        filename,
        PIPELINE_HTTP_TIMEOUT_S,
        extra_form={"slide_type": parsed_smear, "task": task_str},
        headers=headers,
    )


def infer_cellularity(
    image_bytes: bytes,
    filename: str = "tile.jpg",
    gpu_id: Optional[int] = None,
    smear_type: str = "BM",
) -> dict[str, Any]:
    """
    骨髓增生程度：multi_pipeline_server POST /infer（dpi=147246, task=cellularity）。
    只依赖 LOWRES-CELLULARITY，不走定位/评分完整 pipeline。
    传图前居中补边至固定尺寸 2448x2048。
    """
    if not image_bytes:
        return {"ok": False, "error": "empty image payload", "infer_ms": 0.0, "pad_ms": 0.0}
    t_pad0 = time.perf_counter()
    try:
        image_bytes = _pad_image_bytes_to_cell_analysis_size(image_bytes)
    except Exception as e:
        return {
            "ok": False,
            "error": f"pad image failed: {e}",
            "infer_ms": 0.0,
            "pad_ms": (time.perf_counter() - t_pad0) * 1000.0,
        }
    pad_ms = (time.perf_counter() - t_pad0) * 1000.0
    gpu_id, endpoint = _resolve_triton_route(gpu_id)
    url = _pipeline_infer_url(endpoint=endpoint)
    try:
        t_infer0 = time.perf_counter()
        res_json = _post_multipart_pipeline_infer(
            url,
            image_bytes,
            filename,
            PIPELINE_HTTP_TIMEOUT_S,
            extra_form={
                "dpi": str(DPI_147246),
                "slide_type": normalize_smear_type(smear_type),
                "task": "cellularity",
            },
        )
        infer_ms = (time.perf_counter() - t_infer0) * 1000.0
    except Exception as e:
        return {"ok": False, "error": str(e), "infer_ms": 0.0, "pad_ms": pad_ms}
    if res_json.get("error"):
        return {
            "ok": False,
            "error": str(res_json.get("error")),
            "infer_ms": infer_ms,
            "pad_ms": pad_ms,
        }
    wpc, rpc = _nested_cellularity_counts(res_json)
    return {
        "ok": True,
        "wbc_pixel_count": wpc,
        "red_pixel_count": rpc,
        "infer_ms": infer_ms,
        "pad_ms": pad_ms,
    }


def infer_image_enhance(image_bytes: bytes) -> tuple[bytes, str]:
    """
    x40 深度学习滤镜：multi_pipeline_server POST /image_enhance/infer（裸流）。
    推理前先 ensure_model_loaded(Image_enhance_pipeline)，与检测模型共用 LRU。
    """
    gpu_id, endpoint = next_triton_endpoint()
    _ensure_filter_model_loaded(endpoint, gpu_id, "image_enhance")
    url = _filter_pipeline_infer_url("image_enhance", endpoint=endpoint)
    logger.debug(
        "infer_image_enhance route gpu_id=%s name=%s url=%s",
        gpu_id,
        endpoint.get("name"),
        url,
    )
    return _post_filter_pipeline_infer(url, image_bytes, PIPELINE_HTTP_TIMEOUT_S)


def infer_opencv_enhance(image_bytes: bytes) -> tuple[bytes, str]:
    """
    x100 滤镜：multi_pipeline_server POST /opencv_enhance/infer（裸流）。
    输入: 原始图片字节（jpg/png）
    输出: (增强后的图片字节, content_type)
    """
    gpu_id, endpoint = next_triton_endpoint()
    url = _filter_pipeline_infer_url("opencv_enhance", endpoint=endpoint)
    logger.debug(
        "infer_opencv_enhance route gpu_id=%s name=%s url=%s",
        gpu_id,
        endpoint.get("name"),
        url,
    )
    return _post_filter_pipeline_infer(url, image_bytes, PIPELINE_HTTP_TIMEOUT_S)


if __name__ == "__main__":
    # 需在项目根目录执行: python -m backend.tools.triton_client
    print("MODEL_TABLE 定位模型解析:")
    for dpi, smear_type, algorithm_types in [
        (144750, "BM", "WBC,MEG"),
        (144750, "PB", "WBC"),
        (147246, "BM", "WBC,MEG"),
        (357378, "BM", "MEG"),
        (357378, "BM", "WBC"),
        (357378, "BM", "PLAT"),
        (357378, "BM", "WBC,PLAT"),
        (357378, "PB", "PLAT"),
        (357378, "PB", "WBC,RBC,PLAT"),
        (714756, "BM", "WBC,RBC"),
        (714756, "BM", "WBC,RBC,PLAT"),
        (35000, "CF", "WBC"),
        (71000, "CF", "WBC"),
        (40, "BM", "WBC,MEG"),
    ]:
        model = get_model_by_dpi(dpi, smear_type=smear_type, algorithm_types=algorithm_types)
        print(f"  dpi={dpi}, smear_type={smear_type}, types={algorithm_types} -> {model}")
