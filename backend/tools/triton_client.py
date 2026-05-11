# triton_client.py
"""multipart 对齐 project.multi_pipeline_server：POST /{147246|357378|714756}/infer；
Image_enhance 等仍走 Triton gRPC。按 DPI 选 target 与结果解析。"""
from __future__ import annotations

import os
import sys

# 直接运行本文件时（python -m backend.tools.triton_client）将项目根加入 path
if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)

import json
import threading
import uuid
import urllib.error
import urllib.request
from urllib.parse import urlparse
from typing import Any, List, Optional

import numpy as np

from project.cells import Cell
from backend.tools.model_control import ensure_model_loaded
from backend.tools.MESSAGE_DICT import CELL_TYPES_X40, CELL_TYPES_X100, CELL_TYPES_MEG
from config import TRITON_URL, TRITON_IP

# DPI 基准值（±10% 容差），仅以 DPI 选择模型，不再使用倍率缩写
DPI_144750 = 144750  # 有核细胞/巨核细胞/红细胞/血小板定位
DPI_357378 = 357378  # 巨核细胞定位分类
DPI_714756 = 714756  # 有核细胞/成熟红细胞
TOLERANCE = 0.1

# 模型名称常量：144750 常驻 pipeline 见 config.TRITON_PINNED_PIPELINE_NAME（各组预估显存须与 GROUP_VRAM_GB 一致）
MODEL_144750 = "DPI147246_BM_PB_pipeline"  # 144750: BM/PB 共用  预估显存占用6G
MODEL_357378 = "DPI357378_BM_MEG_pipeline"  # 357378: BM 巨核细胞  预估显存占用3.5G
MODEL_714756_BM = "DPI714756_BM_PB_pipeline" #  预估显存占用3G
MODEL_714756_CF = "DPI714756_CF_WBC_pipeline" #  预估显存占用7.5G
# 图片增强/滤镜 pipeline（x40 超分辨率滤镜深度学习模式）
MODEL_IMAGE_ENHANCE = "Image_enhance_pipeline" # 预估显存占用3G

# 与 multi_pipeline_server 路由一致：POST /{147246|357378|714756}/infer（multipart）。
# - PIPELINE_SERVER_BASE_URL：推荐，例如 http://192.168.1.10:9000（勿带 /infer）
# - MULTI_PIPELINE_PORT：未设 BASE 时，与 TRITON_IP 拼成 http://TRITON_IP:port
# - PIPELINE_147246_INFER_URL：可写完整 147246 地址；若仅为 http://host:port/infer 会自动补上 /147246/infer，
#   并向 357378/714756 派生同主机的 /{target}/infer
_MULTI_PIPELINE_TARGETS = frozenset({"147246", "357378", "714756"})
_DEFAULT_MULTI_PIPELINE_PORT = int(os.environ.get("MULTI_PIPELINE_PORT", "9000"))
_PIPELINE_SERVER_BASE_URL_RAW = os.environ.get("PIPELINE_SERVER_BASE_URL", "").strip().rstrip("/")
_PIPELINE_147246_INFER_URL_RAW = os.environ.get("PIPELINE_147246_INFER_URL", "").strip().rstrip("/")

PIPELINE_HTTP_TIMEOUT_S = float(os.environ.get("PIPELINE_HTTP_TIMEOUT_S", "600"))


def _normalize_http_url(url_or_hostport: str) -> str:
    s = url_or_hostport.strip().rstrip("/")
    if not s:
        return s
    if not s.lower().startswith(("http://", "https://")):
        s = f"http://{s}"
    return s


def _default_multi_pipeline_base() -> str:
    return _normalize_http_url(f"{TRITON_IP}:{_DEFAULT_MULTI_PIPELINE_PORT}")


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


def _multi_pipeline_infer_url(target: str) -> str:
    """与 multi_pipeline_server 一致的路径：/{target}/infer（multipart）。"""
    if target not in _MULTI_PIPELINE_TARGETS:
        raise ValueError(f"invalid multi pipeline target: {target!r}")

    bs_raw = _PIPELINE_SERVER_BASE_URL_RAW
    ov_raw = _PIPELINE_147246_INFER_URL_RAW

    if target == "147246" and ov_raw:
        ov = _normalize_http_url(ov_raw)
        root = _strip_plain_infer_suffix(ov)
        if root is not None:
            return f"{root}/{target}/infer"
        return ov

    if bs_raw:
        bs = _normalize_http_url(bs_raw)
        return f"{bs.rstrip('/')}/{target}/infer"

    if ov_raw:
        ov = _normalize_http_url(ov_raw)
        root = _strip_plain_infer_suffix(ov)
        if root is not None:
            return f"{root}/{target}/infer"
        if "147246" in ov:
            return ov.replace("147246", target, 1)

    db = _default_multi_pipeline_base()
    return f"{db.rstrip('/')}/{target}/infer"

# X50 14 类 → 200000-200013, CSF 12 类 → 300000+, BM 100x 35 类 → 200000-200034
X50_CLASS_NAMES = [f"类{i}" for i in range(14)]
CSF_CLASS_NAMES = [f"CF_{i+1}" for i in range(12)]

_triton_client = None
_triton_client_lock = threading.Lock()


def _in_dpi_range(value: int, base: int) -> bool:
    """判断 value 是否在 base 的 ±10% 范围内"""
    low = int(base * (1 - TOLERANCE))
    high = int(base * (1 + TOLERANCE))
    return low <= value <= high


def get_model_by_dpi(
    dpi: int,
    smear_type: str = "BM",
    algorithm_types: str = "",
) -> str:
    """
    仅根据 DPI 选择 Triton 模型（与 smear_type、target_cell_types 组合见下方有效表）。

    有效组合:
    - 144750 ± 10%: BM(WBC,MEG) / PB(WBC,RBC,PLAT) → MODEL_144750
    - 357378 ± 10%: BM(MEG) → MODEL_357378；BM(WBC) 暂无专用模型，临时走 MODEL_714756_BM（与 714756 BM WBC 同 pipeline）
    - 714756 ± 10%: BM(WBC,RBC) / PB(WBC,RBC) / CF(WBC) → MODEL_714756_CF(CF) / MODEL_714756_BM(BM/PB)
    """
    # 遗留倍率缩写 → 实际 DPI
    if dpi in (40, 50, 100):
        dpi = {40: DPI_144750, 50: DPI_357378, 100: DPI_714756}[dpi]
    if _in_dpi_range(dpi, DPI_144750):
        return MODEL_144750
    if _in_dpi_range(dpi, DPI_357378):
        st = (smear_type or "BM").strip().upper()
        at = (algorithm_types or "").upper()
        # 暂无 DPI357378 的 BM WBC 专用模型：仅 WBC（不含 MEG）时临时使用 714756 BM/PB pipeline
        if st == "BM" and "WBC" in at and "MEG" not in at:
            return MODEL_714756_BM
        if st == "PB" and ("WBC" in at or "RBC" in at):
            return MODEL_714756_BM
        return MODEL_357378
    if _in_dpi_range(dpi, DPI_714756):
        if (smear_type or "").upper() == "CF":
            return MODEL_714756_CF
        if (smear_type or "").upper() == "MEG":
            return MODEL_357378
        return MODEL_714756_BM
    return MODEL_144750


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
    boundary = uuid.uuid4().hex
    bnd = boundary.encode("ascii")
    crlf = b"\r\n"
    chunks: list[bytes] = []

    if extra_form:
        for name, value in extra_form.items():
            chunks.append(b"--" + bnd + crlf)
            chunks.append(
                f'Content-Disposition: form-data; name="{name}"'.encode("utf-8") + crlf + crlf
            )
            chunks.append(str(value).encode("utf-8") + crlf)

    disp = (f'Content-Disposition: form-data; name="image"; filename="{filename}"').encode("utf-8")
    chunks.append(b"--" + bnd + crlf)
    chunks.append(disp + crlf)
    chunks.append(b"Content-Type: image/jpeg" + crlf + crlf)
    chunks.append(image_bytes + crlf)
    chunks.append(b"--" + bnd + b"--" + crlf)
    body = b"".join(chunks)

    req = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = str(e)
        try:
            err_json = json.loads(err_body)
        except json.JSONDecodeError:
            raise RuntimeError(f"pipeline_server HTTP {e.code}: {err_body}") from e
        if isinstance(err_json, dict) and err_json.get("error") is not None:
            typ = err_json.get("type", "")
            suf = f" [{typ}]" if typ else ""
            raise RuntimeError(f"pipeline_server HTTP {e.code}{suf}: {err_json['error']}") from e
        raise RuntimeError(f"pipeline_server HTTP {e.code}: {err_body}") from e

    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"pipeline_server 返回非 JSON: {raw[:500]!r}") from e


def _post_multipart_infer_147246(
    url: str,
    image_bytes: bytes,
    filename: str,
    enable_meg: int,
    timeout_s: float,
) -> dict[str, Any]:
    """enable_meg + image（同 multi_pipeline_server /147246/infer Form）。"""
    return _post_multipart_pipeline_infer(
        url,
        image_bytes,
        filename,
        timeout_s,
        extra_form={"enable_meg": str(int(enable_meg))},
    )


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
    cell_list = _cells_to_cell_list_single(cells)
    return {
        "cells": cells,
        "scores": constituency_scores_combined,
        "cell_list": cell_list,
        "wbc_pixel_count": wbc_pixel_count,
        "red_pixel_count": red_pixel_count,
    }


def _infer_357378_from_pipeline_json(res: dict[str, Any]) -> dict[str, Any]:
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
    cell_list = _cells_to_cell_list_top5(cells, cids, cprobs, 300000, CELL_TYPES_MEG, X50_CLASS_NAMES)
    return {"cells": cells, "scores": scores_out, "cell_list": cell_list}


def _infer_714756_bm_from_pipeline_json(res: dict[str, Any]) -> dict[str, Any]:
    if res.get("error"):
        raise RuntimeError(str(res.get("error")))
    boxes_raw = res.get("boxes") if res.get("boxes") is not None else res.get("BOXES")
    scores_raw = res.get("scores") if res.get("scores") is not None else res.get("SCORES")
    class_ids_raw = res.get("class_ids") if res.get("class_ids") is not None else res.get("CLASS_IDS")
    red_det_raw = res.get("red_detections") if res.get("red_detections") is not None else res.get("RED_DETECTIONS")

    wbc_num = _scalar_int(res, "num_detections", "NUM_DETECTIONS", default=-1)
    boxes = np.asarray(boxes_raw, dtype=np.float64) if boxes_raw is not None else None
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

    red_num = _scalar_int(res, "red_num_detections", "RED_NUM_DETECTIONS", "red_num", default=-1)

    scores = np.asarray(scores_raw, dtype=np.float64) if scores_raw is not None else None
    class_ids = np.asarray(class_ids_raw, dtype=np.int32) if class_ids_raw is not None else None

    cells: List[Cell] = []
    scores_out: List[float] = []
    cell_list: List[Any] = []
    wbc_names = [CELL_TYPES_X100.get(200000 + i, ("?", f"cell_{i}"))[1] for i in range(35)]
    if wbc_num > 0 and boxes is not None:
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
            _cells_to_cell_list_top5(wbc_cells, cids_arr, cprobs_for_top5, 200000, CELL_TYPES_X100, wbc_names)
        )

    red_det = np.asarray(red_det_raw, dtype=np.float64) if red_det_raw is not None else None
    if red_det is not None and red_det.size > 0:
        rd = red_det.reshape(-1, red_det.shape[-1]) if red_det.ndim >= 2 else red_det.reshape(1, -1)
        if red_num <= 0:
            red_num = len(rd)
        red_num = min(red_num, len(rd))
    else:
        rd = None
        red_num = 0

    if red_num > 0 and rd is not None:
        rbc_cells: List[Cell] = []
        for r in rd[:red_num]:
            rl = np.asarray(r).flatten().tolist()
            if len(rl) < 4:
                continue
            x, y, w, h = float(rl[0]), float(rl[1]), float(rl[2]), float(rl[3])
            conf = float(rl[4]) if len(rl) > 4 else 1.0
            type_info = CELL_TYPES_X40.get(100002, ("Unclassified_RBC", "未分类红细胞"))
            type_name = type_info[1] if isinstance(type_info, (tuple, list)) else "未分类红细胞"
            cell = Cell(
                cell_xmin=int(x),
                cell_ymin=int(y),
                cell_xmax=int(x + w),
                cell_ymax=int(y + h),
                cell_type=100002,
                cell_type_name=type_name,
                class_confidence=conf,
                bbox_confidence=1.0,
            )
            cells.append(cell)
            rbc_cells.append(cell)
            scores_out.append(conf)
        cell_list.extend(_cells_to_cell_list_single(rbc_cells))

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


def _cells_to_cell_list_single(cells: List[Cell]) -> list:
    """无 TOP5 时：每个 cell 的 tops 只放一项"""
    return [
        {
            "cell_xmin": c.cell_xmin,
            "cell_ymin": c.cell_ymin,
            "cell_xmax": c.cell_xmax,
            "cell_ymax": c.cell_ymax,
            "tops": [{
                "cell_type": c.cell_type,
                "cell_type_name": c.cell_type_name,
                "class_confidence": float(c.class_confidence),
                "bbox_confidence": float(c.bbox_confidence),
            }],
        }
        for c in cells
    ]


def _cells_to_cell_list_top5(
    cells: List[Cell],
    class_ids: np.ndarray,
    class_probs: np.ndarray,
    cell_type_base: int,
    type_map: dict,
    class_names: Optional[List[str]] = None,
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
            if type_map and cell_type in type_map:
                t = type_map[cell_type]
                type_name = t[1] if isinstance(t, (tuple, list)) else str(t)
            else:
                type_name = (class_names[cls_id] if class_names and cls_id < len(class_names) else '分类不明/无法分类巨核细胞')
            tops.append({
                "cell_type": cell_type,
                "cell_type_name": type_name,
                "class_confidence": prob,
                "bbox_confidence": float(c.bbox_confidence),
            })
        out.append({
            "cell_xmin": c.cell_xmin,
            "cell_ymin": c.cell_ymin,
            "cell_xmax": c.cell_xmax,
            "cell_ymax": c.cell_ymax,
            "tops": tops,
        })
    return out


def _get_client():
    """获取或创建 Triton 客户端"""
    global _triton_client
    import tritonclient.grpc as grpcclient
    with _triton_client_lock:
        if _triton_client is None:
            _triton_client = grpcclient.InferenceServerClient(url=TRITON_URL)
        return _triton_client


def infer(
    image_bytes: bytes,
    dpi: int,
    smear_type: str = "BM",
    algorithm_types: str = "",
) -> dict:
    """
    细胞检测推理，仅根据 DPI 选择模型。单图识别与任务模式均使用此接口。

    有效组合见 get_model_by_dpi。返回: {"cells": List[Cell], "scores": List[float] (如有)}

    144750 → target 147246；357378 → 357378；714756(BM) → 714756。
    URL 与 multi_pipeline_server 一致，见 _multi_pipeline_infer_url；环境变量：
    PIPELINE_SERVER_BASE_URL（推荐）、MULTI_PIPELINE_PORT、PIPELINE_147246_INFER_URL（仅覆盖 147246 或作兄弟路径推导）。
    """
    model = get_model_by_dpi(dpi, smear_type=smear_type, algorithm_types=algorithm_types)

    if model == MODEL_144750:
        enable_meg = 1 if "MEG" in (algorithm_types or "") else 0
        url = _multi_pipeline_infer_url("147246")
        res_json = _post_multipart_infer_147246(
            url, image_bytes, "tile.jpg", enable_meg, PIPELINE_HTTP_TIMEOUT_S
        )
        wbc, wbc_num, meg, meg_num, cr, cs, cg, wpc, rpc = _parse_pipeline_json_147246(res_json)
        return _infer_147246_finalize(
            algorithm_types, wbc, wbc_num, meg, meg_num, cr, cs, cg, wpc, rpc
        )

    if model == MODEL_357378:
        url = _multi_pipeline_infer_url("357378")
        res_json = _post_multipart_pipeline_infer(
            url, image_bytes, "tile.jpg", PIPELINE_HTTP_TIMEOUT_S
        )
        return _infer_357378_from_pipeline_json(res_json)

    if model == MODEL_714756_BM:
        task_mode = 0
        if "RBC" in (algorithm_types or "") or "RED" in (algorithm_types or ""):
            task_mode = 2 if "WBC" in (algorithm_types or "") or "MEG" in (algorithm_types or "") else 1
        url = _multi_pipeline_infer_url("714756")
        res_json = _post_multipart_pipeline_infer(
            url,
            image_bytes,
            "tile.jpg",
            PIPELINE_HTTP_TIMEOUT_S,
            extra_form={"task_mode": str(int(task_mode))},
        )
        return _infer_714756_bm_from_pipeline_json(res_json)

    return {"cells": [], "scores": [], "cell_list": []}


def infer_image_enhance(image_bytes: bytes) -> bytes:
    """
    图片增强/滤镜推理（Triton Image_enhance_pipeline）。
    输入: 原始图片字节（jpg/png）
    输出: 增强后的图片字节（jpg 编码）
    """
    import tritonclient.grpc as grpcclient

    ok, err = ensure_model_loaded(MODEL_IMAGE_ENHANCE)
    if not ok:
        raise RuntimeError(f"Model {MODEL_IMAGE_ENHANCE} load failed: {err}")
    client = _get_client()
    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    inp_raw = grpcclient.InferInput("RAW_IMAGE", [len(raw)], "UINT8")
    inp_raw.set_data_from_numpy(raw)
    outputs = [grpcclient.InferRequestedOutput("ENHANCED_IMAGE")]
    result = client.infer(MODEL_IMAGE_ENHANCE, inputs=[inp_raw], outputs=outputs)
    out = result.as_numpy("ENHANCED_IMAGE")
    if out is None:
        raise RuntimeError("模型未返回输出: ENHANCED_IMAGE")
    return out.tobytes()


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
