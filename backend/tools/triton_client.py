# triton_client.py
"""multipart 对齐 project.multi_pipeline_server：POST /{147246|357378|714756}/infer；
滤镜走 multi_pipeline POST /{image_enhance|opencv_enhance}/infer（裸流）。按 DPI 选 target 与结果解析。"""
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
from urllib.parse import quote, urlparse
from typing import Any, Callable, List, Optional

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime

from project.cells import Cell
from backend.tools.MESSAGE_DICT import CELL_TYPES_X40, CELL_TYPES_X100, CELL_TYPES_MEG, get_counting_cell_type
from config import next_triton_endpoint, get_triton_endpoint
from backend.tools.model_control import ensure_model_loaded

logger = logging.getLogger(__name__)

# DPI 基准值（±10% 容差），仅以 DPI 选择模型，不再使用倍率缩写
DPI_144750 = 144750  # 有核细胞/巨核细胞/红细胞/血小板定位
DPI_357378 = 357378  # 巨核细胞定位分类
DPI_714756 = 714756  # 有核细胞/成熟红细胞
DPI_BASES = (DPI_144750, DPI_357378, DPI_714756)
DPI_OUT_OF_RANGE_WARNING = "DPI out of valid range (144750/357378/714756 ±10%)"
TOLERANCE = 0.1

# 模型名称常量（各组预估显存须与 model_control.GROUP_VRAM_GB 一致；357378 为常驻组）
MODEL_144750 = "DPI147246_BM_PB_pipeline"  # 144750: BM/PB 共用  预估显存占用 3.3G（LRU）
MODEL_357378 = "DPI357378_BM_MEG_pipeline"  # 357378: BM 巨核细胞  预估显存占用 0.2G（常驻）
MODEL_714756_BM = "DPI714756_BM_PB_pipeline"  # 714756: BM/PB  预估显存占用 3.1G（LRU）
MODEL_714756_CF = "DPI714756_CF_WBC_pipeline"  # 714756: CF  预估显存占用 7.5G（未启用）
# 图片增强/滤镜 pipeline（x40 超分辨率滤镜深度学习模式）
MODEL_IMAGE_ENHANCE = "Image_enhance_pipeline"  # 预估显存占用 1.6G（LRU）

# 与 multi_pipeline_server 路由一致：POST /{147246|357378|714756}/infer（multipart）。
# 双容器轮询：见 config.next_triton_endpoint（gpu0:9000 / gpu1:9010）。
# PIPELINE_147246_INFER_URL 仅作单点覆盖（会禁用该次请求的端点派生，仍走轮询到的 host 时请勿设置）。
_MULTI_PIPELINE_TARGETS = frozenset({"147246", "357378", "714756"})
_FILTER_PIPELINE_TARGETS = frozenset({"image_enhance", "opencv_enhance"})
_PIPELINE_147246_INFER_URL_RAW = os.environ.get("PIPELINE_147246_INFER_URL", "").strip().rstrip("/")

PIPELINE_HTTP_TIMEOUT_S = float(os.environ.get("PIPELINE_HTTP_TIMEOUT_S", "600"))

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


def _multi_pipeline_infer_url(target: str, endpoint: dict | None = None) -> str:
    """与 multi_pipeline_server 一致的路径：/{target}/infer（multipart）。"""
    if target not in _MULTI_PIPELINE_TARGETS:
        raise ValueError(f"invalid multi pipeline target: {target!r}")

    ov_raw = _PIPELINE_147246_INFER_URL_RAW
    # 轮询场景：优先用本次选中端点的 pipeline_base_url
    if endpoint is not None and not ov_raw:
        bs = _normalize_http_url(endpoint.get("pipeline_base_url") or "")
        if bs:
            return f"{bs.rstrip('/')}/{target}/infer"

    if target == "147246" and ov_raw:
        ov = _normalize_http_url(ov_raw)
        root = _strip_plain_infer_suffix(ov)
        if root is not None:
            return f"{root}/{target}/infer"
        return ov

    if ov_raw:
        ov = _normalize_http_url(ov_raw)
        root = _strip_plain_infer_suffix(ov)
        if root is not None:
            return f"{root}/{target}/infer"
        if "147246" in ov:
            return ov.replace("147246", target, 1)

    ep = endpoint or get_triton_endpoint()
    bs = _normalize_http_url(ep.get("pipeline_base_url") or "")
    return f"{bs.rstrip('/')}/{target}/infer"


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
    推理前经 model_control 加载滤镜模型（参与 LRU，与检测模型共用组数上限）。
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


# X50 14 类 → 200000-200013, CSF 12 类 → 300000+, BM 100x 35 类 → 200000-200034
X50_CLASS_NAMES = [f"类{i}" for i in range(14)]
CSF_CLASS_NAMES = [f"CF_{i+1}" for i in range(12)]

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


def _in_dpi_range(value: int, base: int) -> bool:
    """判断 value 是否在 base 的 ±10% 范围内"""
    low = int(base * (1 - TOLERANCE))
    high = int(base * (1 + TOLERANCE))
    return low <= value <= high


def _get_dpi_bucket(dpi: int) -> tuple[int, str | None]:
    """返回 DPI 所属 bucket；超出范围时返回最近 bucket 和告警。"""
    if dpi in (40, 50, 100):
        return {40: DPI_144750, 50: DPI_357378, 100: DPI_714756}[dpi], None
    for base in DPI_BASES:
        if _in_dpi_range(dpi, base):
            return base, None
    return min(DPI_BASES, key=lambda base: abs(dpi - base)), DPI_OUT_OF_RANGE_WARNING


def get_model_by_dpi(
    dpi: int,
    smear_type: str = "BM",
    algorithm_types: str = "",
    *,
    return_warning: bool = False,
) -> str | tuple[str, str | None]:
    """
    仅根据 DPI 选择 Triton 模型（与 smear_type、target_cell_types 组合见下方有效表）。

    有效组合:
    - 144750 ± 10%: BM(WBC,MEG) / PB(WBC,RBC,PLAT) → MODEL_144750
    - 357378 ± 10%: BM(MEG) → MODEL_357378；BM(WBC) 暂无专用模型，临时走 MODEL_714756_BM（与 714756 BM WBC 同 pipeline）
    - 714756 ± 10%: BM(WBC,RBC) / PB(WBC,RBC) / CF(WBC) → MODEL_714756_CF(CF) / MODEL_714756_BM(BM/PB)
    超出范围时使用绝对差最小的 DPI bucket，并可通过 return_warning 返回告警。
    """
    st = smear_type.strip().upper()
    at = algorithm_types.strip().upper()
    bucket, warning = _get_dpi_bucket(dpi)
    if bucket == DPI_144750:
        model = MODEL_144750
    elif bucket == DPI_357378:
        if st == "BM" and "WBC" in at and "MEG" not in at:
            model = MODEL_714756_BM
        elif st == "PB" and ("WBC" in at or "RBC" in at):
            model = MODEL_714756_BM
        else:
            model = MODEL_357378
    else:
        if st == "CF":
            model = MODEL_714756_CF
        elif "MEG" in at:
            model = MODEL_357378
        else:
            model = MODEL_714756_BM
    if return_warning:
        return model, warning
    return model


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
    params = {name: value for name, value in extra_form.items()} if extra_form else None

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
) -> dict[str, Any]:
    """multipart/form-data：字段名对齐 multi_pipeline_server（image 必选；714756 为 task_mode；147246 为 enable_meg）。"""
    if not url.lower().startswith("http"):
        url = f"http://{url}"

    # requests 自动生成 boundary 并设置 Content-Type；普通字段走 data，文件走 files。
    data = {name: str(value) for name, value in extra_form.items()} if extra_form else None
    files = {"image": (filename, image_bytes, "image/jpeg")}
    # 记录日志，task_id,file_name,发送请求的时间
    logger.info("file_name=%s, 发送请求的时间：%s", filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    try:
        resp = _get_pipeline_session().post(
            url,
            data=data,
            files=files,
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


def _parse_pipeline_json_147246(res: dict[str, Any]) -> tuple[
    Optional[np.ndarray],
    int,
    Optional[np.ndarray],
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
]:
    if res.get("error"):
        raise RuntimeError(str(res.get("error")))

    wbc_raw = res.get("wbc_detections")
    if wbc_raw is None:
        wbc_raw = res.get("WBC_DETECTIONS")
    meg_raw = res.get("meg_detections")
    if meg_raw is None:
        meg_raw = res.get("MEG_DETECTIONS")

    wbc_arr: Optional[np.ndarray]
    if wbc_raw is None:
        wbc_arr = None
    else:
        wbc_arr = np.asarray(wbc_raw, dtype=np.float64)
        if wbc_arr.size == 0:
            wbc_arr = None
    meg_arr: Optional[np.ndarray]
    if meg_raw is None:
        meg_arr = None
    else:
        meg_arr = np.asarray(meg_raw, dtype=np.float64)
        if meg_arr.size == 0:
            meg_arr = None

    wbc_num = _scalar_int(res, "wbc_num", "WBC_NUM_DETECTIONS", default=-1)
    meg_num = _scalar_int(res, "meg_num", "MEG_NUM_DETECTIONS", default=-1)
    if wbc_arr is not None:
        n = len(wbc_arr)
        if wbc_num <= 0:
            wbc_num = n
        wbc_num = min(wbc_num, n)
    else:
        wbc_num = 0
    if meg_arr is not None:
        n = len(meg_arr)
        if meg_num <= 0:
            meg_num = n
        meg_num = min(meg_num, n)
    else:
        meg_num = 0

    cr = (
        res.get("constituency_regions")
        if res.get("constituency_regions") is not None
        else res.get("CONSTITUENCY_REGIONS")
    )
    if cr is None:
        cr = []
    cs = (
        res.get("constituency_scores")
        if res.get("constituency_scores") is not None
        else res.get("CONSTITUENCY_SCORES")
    )
    cg = (
        res.get("constituency_grades")
        if res.get("constituency_grades") is not None
        else res.get("CONSTITUENCY_GRADES")
    )

    regions = np.asarray(cr, dtype=np.float64).reshape(-1, 4) if np.asarray(cr).size > 0 else np.zeros((0, 4), dtype=np.float64)
    scores_a = np.asarray(cs if cs is not None else [], dtype=np.float64).flatten()
    grades_a = np.asarray(cg if cg is not None else [], dtype=np.float64).flatten()

    wpc = _scalar_int(
        res,
        "cell_analysis_wbc_pixel_count",
        "wbc_pixel_count",
        "CELL_ANALYSIS_WBC_PIXEL_COUNT",
    )
    rpc = _scalar_int(
        res,
        "cell_analysis_red_pixel_count",
        "red_pixel_count",
        "CELL_ANALYSIS_RED_PIXEL_COUNT",
    )
    return wbc_arr, wbc_num, meg_arr, meg_num, regions, scores_a, grades_a, wpc, rpc


def _infer_147246_finalize(
    algorithm_types: str,
    wbc: Optional[np.ndarray],
    wbc_num: int,
    meg: Optional[np.ndarray],
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
        if wbc_num > 0 and wbc is not None:
            wc = np.asarray(wbc, dtype=np.float64)
            cells.extend(_boxes_to_cells(wc[:wbc_num], 100000, CELL_TYPES_X40))
    if "MEG" in (algorithm_types or ""):
        if meg_num > 0 and meg is not None:
            mg = np.asarray(meg, dtype=np.float64)
            cells.extend(_boxes_to_cells(mg[:meg_num], 100001, CELL_TYPES_X40))
    cell_list = _cells_to_cell_list_single(cells, smear_type)
    return {
        "cells": cells,
        "scores": constituency_scores_combined,
        "cell_list": cell_list,
        "wbc_pixel_count": wbc_pixel_count,
        "red_pixel_count": red_pixel_count,
    }


def _infer_357378_from_pipeline_json(res: dict[str, Any], smear_type: str) -> dict[str, Any]:
    if res.get("error"):
        raise RuntimeError(str(res.get("error")))
    boxes_raw = res.get("boxes") if res.get("boxes") is not None else res.get("BOXES")
    scores_raw = res.get("scores") if res.get("scores") is not None else res.get("SCORES")
    class_ids_raw = res.get("class_ids") if res.get("class_ids") is not None else res.get("CLASS_IDS")
    class_probs_raw = res.get("class_probs") if res.get("class_probs") is not None else res.get("CLASS_PROBS")

    boxes = np.asarray(boxes_raw, dtype=np.float64) if boxes_raw is not None else None
    if boxes is None or boxes.size == 0:
        return {"cells": [], "scores": [], "cell_list": []}
    if boxes.ndim == 1 and boxes.size % 4 == 0:
        boxes = boxes.reshape(-1, 4)
    elif boxes.ndim < 2:
        return {"cells": [], "scores": [], "cell_list": []}

    navail = int(boxes.shape[0])
    num_det = _scalar_int(res, "num_detections", "NUM_DETECTIONS", default=-1)
    if num_det <= 0:
        num_det = navail
    num_det = min(num_det, navail)
    if num_det <= 0:
        return {"cells": [], "scores": [], "cell_list": []}

    boxes = boxes[:num_det]
    scores = np.asarray(scores_raw, dtype=np.float64)[:num_det] if scores_raw is not None else np.ones(num_det)
    class_ids = (
        np.asarray(class_ids_raw, dtype=np.int32)[:num_det]
        if class_ids_raw is not None
        else np.zeros(num_det, dtype=np.int32)
    )
    class_probs = np.asarray(class_probs_raw, dtype=np.float64) if class_probs_raw is not None else None
    cells = _boxes_xyxy_to_cells(
        boxes, scores, class_ids, 300000, X50_CLASS_NAMES, CELL_TYPES_MEG, class_probs=class_probs
    )
    scores_out = np.asarray(scores).flatten().tolist()
    cids = np.asarray(class_ids).reshape(num_det, -1)
    cprobs = (
        np.asarray(class_probs).reshape(num_det, -1)
        if class_probs is not None
        else np.ones((num_det, 5))
    )
    cell_list = _cells_to_cell_list_top5(cells, cids, cprobs, 300000, CELL_TYPES_MEG, X50_CLASS_NAMES, smear_type)
    return {"cells": cells, "scores": scores_out, "cell_list": cell_list}


def _res_get(res: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in res and res[k] is not None:
            return res[k]
    return None


def _as_float64_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


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
) -> dict[str, Any]:
    """仅在有异常时写入 extra：结构/颜色/形态/聚集及对应置信度。"""
    extra: dict[str, Any] = {}
    # 结构异常
    _append_red_abnormality_extra(extra, "STRUCT", RED_STRUCT_INV, RED_STRUCT_LABELS, struct, struct_prob, i)
    # 颜色异常
    _append_red_abnormality_extra(extra, "COLOR", RED_COLOR_INV, RED_COLOR_LABELS, color, color_prob, i)
    # 形态异常
    _append_red_abnormality_extra(extra, "MORPH", RED_MORPH_INV, RED_MORPH_LABELS, morph, morph_prob, i)
    # 聚集异常
    _append_red_abnormality_extra(extra, "AGG", RED_AGG_INV, RED_AGG_LABELS, agg, agg_prob, i)
    return extra


def _infer_714756_bm_from_pipeline_json(res: dict[str, Any], smear_type: str) -> dict[str, Any]:
    # logger.info("714756_bm pipeline 原始返回:\n%s", res)
    if res.get("error"):
        raise RuntimeError(str(res.get("error")))

    boxes_raw = _res_get(res, "boxes", "BOXES")
    scores_raw = _res_get(res, "scores", "SCORES")
    class_ids_raw = _res_get(res, "class_ids", "CLASS_IDS")

    wbc_num = _scalar_int(res, "num_detections", "NUM_DETECTIONS", default=-1)
    red_num = _scalar_int(res, "red_num_detections", "RED_NUM_DETECTIONS", "red_num", default=-1)
    plat_num = _scalar_int(res, "plat_num_detections", "PLAT_NUM_DETECTIONS", "plat_num", default=-1)

    red_class_struct = _as_float64_array(_res_get(res, "red_class_struct", "RED_CLASS_STRUCT"))
    red_class_struct_prob = _as_float64_array(_res_get(res, "red_class_struct_prob", "RED_CLASS_STRUCT_PROB"))
    red_class_color = _as_float64_array(_res_get(res, "red_class_color", "RED_CLASS_COLOR"))
    red_class_color_prob = _as_float64_array(_res_get(res, "red_class_color_prob", "RED_CLASS_COLOR_PROB"))
    red_class_morph = _as_float64_array(_res_get(res, "red_class_morph", "RED_CLASS_MORPH"))
    red_class_morph_prob = _as_float64_array(_res_get(res, "red_class_morph_prob", "RED_CLASS_MORPH_PROB"))
    red_class_agg = _as_float64_array(_res_get(res, "red_class_agg", "RED_CLASS_AGG"))
    red_class_agg_prob = _as_float64_array(_res_get(res, "red_class_agg_prob", "RED_CLASS_AGG_PROB"))

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
            if wbc_num < 0:
                wbc_num = nbox
            else:
                wbc_num = min(wbc_num, nbox)

    scores = _as_float64_array(scores_raw)
    class_ids = np.asarray(class_ids_raw, dtype=np.int32) if class_ids_raw is not None else None

    cells: List[Cell] = []
    scores_out: List[float] = []
    cell_list: List[Any] = []

    if wbc_num > 0 and boxes is not None:
        wbc_names = [CELL_TYPES_X100.get(200000 + i, ("?", f"cell_{i}"))[1] for i in range(35)]
        b = boxes[:wbc_num]
        s = scores[:wbc_num] if scores is not None and scores.size >= wbc_num else np.ones(wbc_num)
        c = (
            class_ids[:wbc_num]
            if class_ids is not None and class_ids.shape[0] >= wbc_num
            else np.zeros(wbc_num, dtype=np.int32)
        )
        cprobs_raw = (
            res.get("class_probs") if res.get("class_probs") is not None else res.get("CLASS_PROBS")
        )
        cprobs_arr = (
            np.asarray(cprobs_raw, dtype=np.float64)[:wbc_num].reshape(wbc_num, -1)
            if cprobs_raw is not None
            else None
        )
        wbc_cells = _boxes_xyxy_to_cells(b, s, c, 200000, wbc_names, CELL_TYPES_X100, class_probs=cprobs_arr)
        cells.extend(wbc_cells)
        scores_out.extend(np.asarray(s).flatten().tolist())
        cids_arr = np.asarray(c, dtype=np.int32).reshape(wbc_num, -1)
        cprobs_for_top5 = cprobs_arr if cprobs_arr is not None else np.ones((wbc_num, 5))
        cell_list.extend(
            _cells_to_cell_list_top5(wbc_cells, cids_arr, cprobs_for_top5, 200000, CELL_TYPES_X100, wbc_names, smear_type)
        )

    rd, red_num = _prepare_xywh_detections(_res_get(res, "red_detections", "RED_DETECTIONS"), red_num)
    red_scores = _flatten_det_scores(_res_get(res, "red_det_scores", "RED_DET_SCORES"))
    if red_num > 0 and rd is not None:

        def _red_extra(i: int) -> dict[str, Any]:
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
            )

        rbc_cells = _cells_from_xywh_detections(
            rd, red_num, red_scores, 100005, "已分类红细胞", extra_builder=_red_extra,
        )
        cells.extend(rbc_cells)
        scores_out.extend([c.bbox_confidence for c in rbc_cells])
        cell_list.extend(_cells_to_cell_list_single(rbc_cells, smear_type))

    pd, plat_num = _prepare_xywh_detections(_res_get(res, "plat_detections", "PLAT_DETECTIONS"), plat_num)
    plat_scores = _flatten_det_scores(_res_get(res, "plat_det_scores", "PLAT_DET_SCORES"))
    if plat_num > 0 and pd is not None:
        plat_cells = _cells_from_xywh_detections(
            pd, plat_num, plat_scores, 100004, "血小板",
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


def infer(
    image_bytes: bytes,
    dpi: int,
    smear_type: str = "BM",
    algorithm_types: str = "",
    filename: str = "tile.jpg",
    gpu_id: Optional[int] = None,
) -> dict:
    """
    细胞检测推理，仅根据 DPI 选择模型。单图识别与任务模式均使用此接口。

    有效组合见 get_model_by_dpi。返回: {"cells": List[Cell], "scores": List[float] (如有)}

    144750 → target 147246；357378 → 357378；714756(BM) → 714756。
    平扫 upload_image 依赖 create_task 的 warmup_model，不在此做 ensure；百倍见 get_task_result_x100。
    gpu_id 未指定时经 next_triton_endpoint 轮询单卡选 endpoint。
    """
    model, warning = get_model_by_dpi(
        dpi,
        smear_type=smear_type,
        algorithm_types=algorithm_types,
        return_warning=True,
    )
    gpu_id, endpoint = _resolve_triton_route(gpu_id)
    if model == MODEL_144750:
        enable_meg = 1 if "MEG" in (algorithm_types or "") else 0
        url = _multi_pipeline_infer_url("147246", endpoint=endpoint)
        res_json = _post_raw_pipeline_infer(
            url,
            image_bytes,
            filename,
            PIPELINE_HTTP_TIMEOUT_S,
            extra_form={"enable_meg": str(int(enable_meg))},
        )

        wbc, wbc_num, meg, meg_num, cr, cs, cg, wpc, rpc = _parse_pipeline_json_147246(res_json)
        result = _infer_147246_finalize(
            algorithm_types, wbc, wbc_num, meg, meg_num, cr, cs, cg, wpc, rpc, smear_type
        )
        if warning:
            result["warning"] = warning
        return result

    if model == MODEL_357378:
        url = _multi_pipeline_infer_url("357378", endpoint=endpoint)
        res_json = _post_multipart_pipeline_infer(
            url, image_bytes, filename, PIPELINE_HTTP_TIMEOUT_S
        )
        result = _infer_357378_from_pipeline_json(res_json, smear_type)
        if warning:
            result["warning"] = warning
        return result

    if model == MODEL_714756_BM:
        tasks = ",".join(t.strip() for t in algorithm_types.lower().split(","))
        if "rbc" in tasks:
            tasks = tasks.replace("rbc", "red")
        url = _multi_pipeline_infer_url("714756", endpoint=endpoint)
        res_json = _post_multipart_pipeline_infer(
            url,
            image_bytes,
            filename,
            PIPELINE_HTTP_TIMEOUT_S,
            extra_form={"tasks": tasks},
        )
        result = _infer_714756_bm_from_pipeline_json(res_json, smear_type)
        if warning:
            result["warning"] = warning
        return result

    result = {"cells": [], "scores": [], "cell_list": []}
    if warning:
        result["warning"] = warning
    return result


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
    print("get_model_by_dpi 测试:")
    for dpi, smear_type, algorithm_types in [
        (144750, "BM", "WBC,MEG"),
        (144750, "PB", "WBC,RBC"),
        (357378, "BM", "MEG"),
        (357378, "BM", "WBC"),
        (714756, "BM", "WBC,RBC"),
        (714756, "CF", "WBC"),
        (40, "BM", "WBC,MEG"),
    ]:
        model = get_model_by_dpi(dpi, smear_type=smear_type, algorithm_types=algorithm_types)
        print(f"  dpi={dpi}, smear_type={smear_type}, types={algorithm_types} -> {model}")
