# triton_client.py
"""Triton 推理客户端：同步调用，直接返回结果。根据 DPI 自动选择模型。"""
from __future__ import annotations

import os
import sys

# 直接运行本文件时（python project/triton_client.py）将项目根加入 path
if __name__ == "__main__" or "project" not in sys.modules:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

import threading
from typing import List, Optional

import numpy as np

from project.cells import Cell
from project.model_control import ensure_model_loaded
from backend.tools.MESSAGE_DICT import CELL_TYPES_X40, CELL_TYPES_X100
from config import TRITON_PINNED_PIPELINE_NAME, TRITON_URL

# DPI 基准值（±10% 容差），仅以 DPI 选择模型，不再使用倍率缩写
DPI_144750 = 144750  # 有核细胞/巨核细胞/红细胞/血小板定位
DPI_357378 = 357378  # 巨核细胞定位分类
DPI_714756 = 714756  # 有核细胞/成熟红细胞
TOLERANCE = 0.1

# 模型名称常量：144750 常驻 pipeline 见 config.TRITON_PINNED_PIPELINE_NAME（各组预估显存须与 GROUP_VRAM_GB 一致）
MODEL_144750 = TRITON_PINNED_PIPELINE_NAME  # 144750: BM/PB 共用  预估显存占用6G
MODEL_357378 = "DPI357378_BM_MEG_pipeline"  # 357378: BM 巨核细胞  预估显存占用3.5G
MODEL_714756_BM = "DPI714756_BM_PB_pipeline" #  预估显存占用3G
MODEL_714756_CF = "DPI714756_CF_WBC_pipeline" #  预估显存占用7.5G
# 图片增强/滤镜 pipeline（x40 超分辨率滤镜深度学习模式）
MODEL_IMAGE_ENHANCE = "Image_enhance_pipeline" # 预估显存占用3G

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
        return MODEL_357378
    if _in_dpi_range(dpi, DPI_714756):
        if (smear_type or "").upper() == "CF":
            return MODEL_714756_CF
        return MODEL_714756_BM
    return MODEL_144750


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
            type_name = class_names[cls_id] if cls_id < len(class_names) else f"类{cls_id}"
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
                type_name = (class_names[cls_id] if class_names and cls_id < len(class_names) else f"类{cls_id}")
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
    """
    import tritonclient.grpc as grpcclient

    model = get_model_by_dpi(dpi, smear_type=smear_type, algorithm_types=algorithm_types)
    ok, err = ensure_model_loaded(model)
    if not ok:
        raise RuntimeError(f"Model {model} load failed: {err}")
    client = _get_client()
    raw = np.frombuffer(image_bytes, dtype=np.uint8)

    if model == MODEL_144750:
        # 144750: WBC/MEG pipeline, BM(WBC,MEG) / PB(WBC,RBC,PLAT)
        inp = grpcclient.InferInput("RAW_IMAGE", [1, len(raw)], "UINT8")
        inp.set_data_from_numpy(raw.reshape(1, -1))
        enable_meg = "MEG" in (algorithm_types or "")
        inp_meg = grpcclient.InferInput("ENABLE_MEG", [1, 1], "INT32")
        inp_meg.set_data_from_numpy(np.array([[1 if enable_meg else 0]], dtype=np.int32))
        req_outputs = [
            grpcclient.InferRequestedOutput("WBC_DETECTIONS"),
            grpcclient.InferRequestedOutput("WBC_NUM_DETECTIONS"),
            grpcclient.InferRequestedOutput("MEG_DETECTIONS"),
            grpcclient.InferRequestedOutput("MEG_NUM_DETECTIONS"),
            grpcclient.InferRequestedOutput("CONSTITUENCY_REGIONS"),
            grpcclient.InferRequestedOutput("CONSTITUENCY_GRADES"),
            grpcclient.InferRequestedOutput("CONSTITUENCY_SCORES"),
            grpcclient.InferRequestedOutput("CELL_ANALYSIS_WBC_PIXEL_COUNT"),
            grpcclient.InferRequestedOutput("CELL_ANALYSIS_RED_PIXEL_COUNT"),
        ]
        result = client.infer(model, inputs=[inp, inp_meg], outputs=req_outputs)
        wbc = result.as_numpy("WBC_DETECTIONS")
        wbc_num = int(result.as_numpy("WBC_NUM_DETECTIONS")[0])
        meg = result.as_numpy("MEG_DETECTIONS")
        meg_num = int(result.as_numpy("MEG_NUM_DETECTIONS")[0])
        con_regions = result.as_numpy("CONSTITUENCY_REGIONS")
        con_grades = result.as_numpy("CONSTITUENCY_GRADES")
        con_scores = result.as_numpy("CONSTITUENCY_SCORES")
        # 组合成 [[x, y, w, h, score, grade], ...] 格式（ndarray 无 append，用 column_stack）
        regions = np.asarray(con_regions).reshape(-1, 4)
        scores_arr = np.asarray(con_scores).flatten()
        grades_arr = np.asarray(con_grades).flatten()
        constituency_scores_combined = np.column_stack([regions, scores_arr, grades_arr]).tolist()
        cells = []
        if "WBC" in (algorithm_types or ""):
            if wbc_num > 0 and wbc is not None:
                cells.extend(_boxes_to_cells(wbc[:wbc_num], 100000, CELL_TYPES_X40))
        if "MEG" in (algorithm_types or ""):
            if meg_num > 0 and meg is not None:
                cells.extend(_boxes_to_cells(meg[:meg_num], 100001, CELL_TYPES_X40))
        cell_list = _cells_to_cell_list_single(cells)
        wbc_pixel_count = int(result.as_numpy("CELL_ANALYSIS_WBC_PIXEL_COUNT").reshape(-1)[0])
        red_pixel_count = int(result.as_numpy("CELL_ANALYSIS_RED_PIXEL_COUNT").reshape(-1)[0])
        return {"cells": cells, "scores": constituency_scores_combined, "cell_list": cell_list, 'wbc_pixel_count': wbc_pixel_count, 'red_pixel_count': red_pixel_count}

    if model == MODEL_357378:
        # DPI357378: BOXES(x1,y1,x2,y2), 14 类 → cell_type 200000+class_id
        inp = grpcclient.InferInput("RAW_IMAGE", [len(raw)], "UINT8")
        inp.set_data_from_numpy(raw)
        req_outputs = [
            grpcclient.InferRequestedOutput("BOXES"),
            grpcclient.InferRequestedOutput("SCORES"),
            grpcclient.InferRequestedOutput("CLASS_IDS"),
            grpcclient.InferRequestedOutput("CLASS_PROBS"),
            grpcclient.InferRequestedOutput("NUM_DETECTIONS"),
        ]
        result = client.infer(model, inputs=[inp], outputs=req_outputs)
        boxes = result.as_numpy("BOXES")
        scores = result.as_numpy("SCORES")
        class_ids = result.as_numpy("CLASS_IDS")
        num_det = int(result.as_numpy("NUM_DETECTIONS")[0])
        if num_det <= 0:
            return {"cells": [], "scores": [], "cell_list": []}
        boxes = boxes[:num_det]
        scores = scores[:num_det]
        class_ids = class_ids[:num_det]
        class_probs = result.as_numpy("CLASS_PROBS")
        cells = _boxes_xyxy_to_cells(
            boxes, scores, class_ids, 200000, X50_CLASS_NAMES, CELL_TYPES_X100, class_probs=class_probs
        )
        scores_out = np.asarray(scores).flatten().tolist()
        cids = np.asarray(class_ids).reshape(num_det, -1)
        cprobs = np.asarray(class_probs).reshape(num_det, -1) if class_probs is not None else np.ones((num_det, 5))
        cell_list = _cells_to_cell_list_top5(cells, cids, cprobs, 200000, CELL_TYPES_X100, X50_CLASS_NAMES)
        return {"cells": cells, "scores": scores_out, "cell_list": cell_list}

    if model == MODEL_714756_CF:
        # DPI714756_CF: CSF 12 类 → cell_type 300000+class_id 以区分
        inp = grpcclient.InferInput("RAW_IMAGE", [len(raw)], "UINT8")
        inp.set_data_from_numpy(raw)
        req_outputs = [
            grpcclient.InferRequestedOutput("BOXES"),
            grpcclient.InferRequestedOutput("SCORES"),
            grpcclient.InferRequestedOutput("CLASS_IDS"),
            grpcclient.InferRequestedOutput("CLASS_PROBS"),
            grpcclient.InferRequestedOutput("NUM_DETECTIONS"),
        ]
        result = client.infer(model, inputs=[inp], outputs=req_outputs)
        boxes = result.as_numpy("BOXES")
        scores = result.as_numpy("SCORES")
        class_ids = result.as_numpy("CLASS_IDS")
        num_det = int(result.as_numpy("NUM_DETECTIONS")[0])
        if num_det <= 0:
            return {"cells": [], "scores": [], "cell_list": []}
        boxes = boxes[:num_det]
        scores = scores[:num_det]
        class_ids = class_ids[:num_det]
        class_probs = result.as_numpy("CLASS_PROBS")
        cells = _boxes_xyxy_to_cells(
            boxes, scores, class_ids, 300000, CSF_CLASS_NAMES, class_probs=class_probs
        )
        scores_out = np.asarray(scores).flatten().tolist()
        cids = np.asarray(class_ids).reshape(num_det, -1)
        cprobs = np.asarray(class_probs).reshape(num_det, -1) if class_probs is not None else np.ones((num_det, 5))
        cell_list = _cells_to_cell_list_top5(cells, cids, cprobs, 300000, None, CSF_CLASS_NAMES)
        return {"cells": cells, "scores": scores_out, "cell_list": cell_list}

    if model == MODEL_714756_BM:
        # DPI714756_BM: 统一 pipeline，TASK_MODE 0=WBC, 1=RED, 2=ALL
        task_mode = 0
        if "RBC" in (algorithm_types or "") or "RED" in (algorithm_types or ""):
            task_mode = 2 if "WBC" in (algorithm_types or "") or "MEG" in (algorithm_types or "") else 1
        inp_raw = grpcclient.InferInput("RAW_IMAGE", [len(raw)], "UINT8")
        inp_raw.set_data_from_numpy(raw)
        inp_mode = grpcclient.InferInput("TASK_MODE", [1], "INT32")
        inp_mode.set_data_from_numpy(np.array([task_mode], dtype=np.int32))
        req_outputs = [
            grpcclient.InferRequestedOutput("BOXES"),
            grpcclient.InferRequestedOutput("SCORES"),
            grpcclient.InferRequestedOutput("CLASS_IDS"),
            grpcclient.InferRequestedOutput("CLASS_PROBS"),
            grpcclient.InferRequestedOutput("NUM_DETECTIONS"),
            grpcclient.InferRequestedOutput("RED_DETECTIONS"),
            grpcclient.InferRequestedOutput("RED_NUM_DETECTIONS"),
        ]
        result = client.infer(model, inputs=[inp_raw, inp_mode], outputs=req_outputs)
        boxes = result.as_numpy("BOXES")
        scores = result.as_numpy("SCORES")
        class_ids = result.as_numpy("CLASS_IDS")
        wbc_num = int(result.as_numpy("NUM_DETECTIONS")[0])
        red_det = result.as_numpy("RED_DETECTIONS")
        red_num = int(result.as_numpy("RED_NUM_DETECTIONS")[0])

        cells = []
        scores_out = []
        cell_list = []
        wbc_names = [CELL_TYPES_X100.get(200000 + i, ("?", f"cell_{i}"))[1] for i in range(35)]
        if wbc_num > 0 and boxes is not None:
            b = boxes[:wbc_num]
            s = scores[:wbc_num] if scores is not None else np.ones(wbc_num)
            c = class_ids[:wbc_num] if class_ids is not None else np.zeros(wbc_num, dtype=np.int32)
            cprobs_raw = result.as_numpy("CLASS_PROBS")
            cprobs_arr = np.asarray(cprobs_raw[:wbc_num]).reshape(wbc_num, -1) if cprobs_raw is not None else None
            wbc_cells = _boxes_xyxy_to_cells(b, s, c, 200000, wbc_names, CELL_TYPES_X100, class_probs=cprobs_arr)
            cells.extend(wbc_cells)
            scores_out.extend(np.asarray(s).flatten().tolist())
            cids_arr = np.asarray(class_ids[:wbc_num]).reshape(wbc_num, -1)
            cprobs_for_top5 = cprobs_arr if cprobs_arr is not None else np.ones((wbc_num, 5))
            cell_list.extend(_cells_to_cell_list_top5(wbc_cells, cids_arr, cprobs_for_top5, 200000, CELL_TYPES_X100, wbc_names))
        if red_num > 0 and red_det is not None:
            rbc_cells = []
            for r in red_det[:red_num]:
                if len(r) >= 4:
                    x, y, w, h = float(r[0]), float(r[1]), float(r[2]), float(r[3])
                    conf = float(r[4]) if len(r) > 4 else 1.0
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
    # 需在项目根目录执行: python -m project.triton_client
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
