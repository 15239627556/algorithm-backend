# -*- coding: utf-8 -*-
"""单张 x100 识别：DPI 缩放、大图分块推理、坐标映射回原图。"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image

from backend.tools.MESSAGE_DICT import (
    DPI_NOT_SUITABLE,
    get_counting_cell_type,
    model_dpi_ranges,
    model_max_src_by_actual_dpi,
)
from backend.tools.combo_validator import LEGACY_DPI_MAP, _parse_cell_types, normalize_smear_type
from backend.tools.image_tiling import (
    DEFAULT_TILE_OVERLAP,
    tile_ranges_1d,
)
from backend.tools.triton_client import infer, resolve_triton_route
from backend.tools.pipeline_guard import (
    PipelineUnavailable,
    assert_inference_allowed,
    trip_on_timeout,
    wait_while_circuit_open,
)
from backend.tools.filter_edge_incomplete_cells import (
    filter_cell_dicts_edge_elongated_1pct,
    filter_cell_dicts_edge_incomplete,
    filter_cell_dicts_small_wbc_714756,
)
from backend.tools.model_control import (
    load_resolved_models,
    resolve_models,
    resolve_models_by_actual_dpi,
)
from project.cells import Cell
from project.tiles import Tile
from algorithms.SelectArea.dedup_cells_across_tiles import dedup_cells_across_tiles_per_type

logger = logging.getLogger(__name__)

# 小程序猜档：(假定拍摄 DPI, 强制使用的模型 actual_dpi)，按优先级排列。
# 714756 模型试原档、一半、1/4 和 2 倍；40 倍模型试 147246、一半、1/4、2 倍。
# 不走单张接口的 dpi_range 缩放限制。细胞数相同则保留更靠前的一档。
_MINIAPP_MODEL_714756 = 714756
_MINIAPP_MODEL_147246 = 147246
MINIAPP_GUESSES: tuple[tuple[int, int], ...] = (
    (_MINIAPP_MODEL_714756, _MINIAPP_MODEL_714756),
    (_MINIAPP_MODEL_714756 // 2, _MINIAPP_MODEL_714756),
    (_MINIAPP_MODEL_714756 // 4, _MINIAPP_MODEL_714756),
    (_MINIAPP_MODEL_714756 * 2, _MINIAPP_MODEL_714756),
    (_MINIAPP_MODEL_147246, _MINIAPP_MODEL_147246),
    (_MINIAPP_MODEL_147246 // 2, _MINIAPP_MODEL_147246),
    (_MINIAPP_MODEL_147246 // 4, _MINIAPP_MODEL_147246),
    (_MINIAPP_MODEL_147246 * 2, _MINIAPP_MODEL_147246),
)
# 破碎细胞及杂质。小程序若第一名是它，改用第二名。
_MINIAPP_BROKEN_CELL_TYPE = 200029
_MINIAPP_LOWRES_WARNING = "您上传的图像像素不足，准确度可能会下降"
# 假定细胞物理边长 20 微米，用原图框反推 DPI；不超过该值才返回像素不足告警。
_MINIAPP_CELL_UM = 20.0
_MINIAPP_DPI_WARN_MAX = 350000

# 切块最小尺寸（文档未规定 max 以外的 min，沿用历史值）
_DPI_TILE_MIN: dict[int, tuple[int, int]] = {
    147246: (2448, 2048),
    357378: (2448, 2048),
    714756: (2048, 1536),
}


def model_tile_limits(actual_dpi: int) -> tuple[int, int, int, int] | None:
    """返回 (max_w, max_h, min_w, min_h)；None 表示该 DPI 无尺寸限制。"""
    dpi = int(actual_dpi)
    if dpi in (35000, 71000):
        return None
    max_by_dpi = model_max_src_by_actual_dpi()
    max_limits = max_by_dpi.get(dpi)
    if max_limits is None:
        fallback_dpi = 714756 if dpi >= 500000 else 357378
        max_limits = max_by_dpi.get(fallback_dpi)
        min_limits = _DPI_TILE_MIN.get(fallback_dpi)
    else:
        min_limits = _DPI_TILE_MIN.get(dpi)
    if max_limits is None:
        return None
    min_w, min_h = min_limits or (0, 0)
    return max_limits[0], max_limits[1], min_w, min_h


def model_tile_max_limits(actual_dpi: int) -> tuple[int, int] | None:
    limits = model_tile_limits(actual_dpi)
    if limits is None:
        return None
    max_w, max_h, _, _ = limits
    return max_w, max_h


def model_tile_min_limits(actual_dpi: int) -> tuple[int, int] | None:
    limits = model_tile_limits(actual_dpi)
    if limits is None:
        return None
    _, _, min_w, min_h = limits
    return min_w, min_h


def dpi_needs_scale(input_dpi: int, model_dpi: int) -> bool:
    """区间内且与实际模型 DPI 不同时需要缩放；区间外由校验层直接拒绝。"""
    req = LEGACY_DPI_MAP.get(int(input_dpi), int(input_dpi))
    ranges = model_dpi_ranges()
    rng = ranges.get(int(model_dpi))
    if rng is not None and not (rng[0] <= req <= rng[1]):
        return False
    return req != int(model_dpi)


def compute_dpi_scale_ratio(input_dpi: int, model_dpi: int) -> float:
    """DPI 在适用区间内时可缩放到实际模型 DPI。"""
    if not dpi_needs_scale(input_dpi, model_dpi):
        return 1.0
    return model_dpi / input_dpi


def compute_force_dpi_scale_ratio(input_dpi: int, model_dpi: int) -> float:
    """忽略 dpi_range，按模型 DPI / 假定 DPI 强制缩放。"""
    src = int(input_dpi)
    if src <= 0:
        return 1.0
    return float(model_dpi) / float(src)


def encode_bgr_jpeg(bgr: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok or buf is None:
        raise RuntimeError("cv2.imencode failed")
    return bytes(buf)


def decode_image_bgr(image_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def peek_image_size(image_bytes: bytes) -> tuple[int, int] | None:
    """只读图片头拿宽高，避免平扫热路径整图解码。"""
    try:
        with Image.open(BytesIO(image_bytes)) as im:
            w, h = im.size
            return int(w), int(h)
    except Exception:
        return None


def infer_needs_decode(
    orig_w: int,
    orig_h: int,
    scale_ratio: float,
    max_w: int,
    max_h: int,
    min_w: int,
    min_h: int,
) -> bool:
    """需要缩放、填充或切块时才解码；否则原图字节可直送 Triton。"""
    if abs(scale_ratio - 1.0) >= 1e-9:
        return True
    if min_w > 0 and min_h > 0 and (orig_w < min_w or orig_h < min_h):
        return True
    if max_w > 0 and max_h > 0 and (orig_w > max_w or orig_h > max_h):
        return True
    return False


def scale_bgr(bgr: np.ndarray, scale_ratio: float) -> np.ndarray:
    if abs(scale_ratio - 1.0) < 1e-9:
        return bgr
    h, w = bgr.shape[:2]
    new_w = max(1, int(round(w * scale_ratio)))
    new_h = max(1, int(round(h * scale_ratio)))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def pad_bgr_to_min(
    bgr: np.ndarray,
    min_w: int,
    min_h: int,
) -> tuple[np.ndarray, int, int]:
    """
    宽或高小于模型最小支持时，用黑底居中填充至 min_w x min_h。
    返回 (padded_bgr, pad_x, pad_y)，pad 为原图左上角在画布中的偏移。
    """
    h, w = int(bgr.shape[0]), int(bgr.shape[1])
    min_w = int(min_w)
    min_h = int(min_h)
    if w >= min_w and h >= min_h:
        return bgr, 0, 0

    canvas_w = max(w, min_w)
    canvas_h = max(h, min_h)
    pad_x = (canvas_w - w) // 2
    pad_y = (canvas_h - h) // 2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=bgr.dtype)
    canvas[pad_y:pad_y + h, pad_x:pad_x + w] = bgr
    return canvas, pad_x, pad_y


def map_bbox_from_scaled(
    xmin: int | float,
    ymin: int | float,
    xmax: int | float,
    ymax: int | float,
    scale_ratio: float,
) -> tuple[int, int, int, int]:
    inv = 1.0 / scale_ratio
    return (
        max(0, int(round(float(xmin) * inv))),
        max(0, int(round(float(ymin) * inv))),
        max(0, int(round(float(xmax) * inv))),
        max(0, int(round(float(ymax) * inv))),
    )


def map_bbox_from_pad(
    xmin: int | float,
    ymin: int | float,
    xmax: int | float,
    ymax: int | float,
    pad_x: int,
    pad_y: int,
) -> tuple[int, int, int, int]:
    return (
        int(round(float(xmin) - pad_x)),
        int(round(float(ymin) - pad_y)),
        int(round(float(xmax) - pad_x)),
        int(round(float(ymax) - pad_y)),
    )


def map_cell_list_from_pad(
    cell_list: list[dict[str, Any]],
    pad_x: int,
    pad_y: int,
) -> list[dict[str, Any]]:
    if pad_x == 0 and pad_y == 0:
        return cell_list
    mapped: list[dict[str, Any]] = []
    for item in cell_list:
        xmin, ymin, xmax, ymax = map_bbox_from_pad(
            item["cell_xmin"], item["cell_ymin"], item["cell_xmax"], item["cell_ymax"],
            pad_x, pad_y,
        )
        new_item = dict(item)
        new_item["cell_xmin"] = xmin
        new_item["cell_ymin"] = ymin
        new_item["cell_xmax"] = xmax
        new_item["cell_ymax"] = ymax
        mapped.append(new_item)
    return mapped


def map_cells_from_pad(
    cells: list[Cell],
    pad_x: int,
    pad_y: int,
) -> list[Cell]:
    if pad_x == 0 and pad_y == 0:
        return cells
    mapped: list[Cell] = []
    for c in cells:
        xmin, ymin, xmax, ymax = map_bbox_from_pad(
            c.cell_xmin, c.cell_ymin, c.cell_xmax, c.cell_ymax,
            pad_x, pad_y,
        )
        mapped.append(replace(
            c,
            cell_xmin=xmin,
            cell_ymin=ymin,
            cell_xmax=xmax,
            cell_ymax=ymax,
        ))
    return mapped


def map_cell_list_from_scaled(
    cell_list: list[dict[str, Any]],
    scale_ratio: float,
) -> list[dict[str, Any]]:
    if abs(scale_ratio - 1.0) < 1e-9:
        return cell_list
    mapped: list[dict[str, Any]] = []
    for item in cell_list:
        xmin, ymin, xmax, ymax = map_bbox_from_scaled(
            item["cell_xmin"], item["cell_ymin"], item["cell_xmax"], item["cell_ymax"],
            scale_ratio,
        )
        new_item = dict(item)
        new_item["cell_xmin"] = xmin
        new_item["cell_ymin"] = ymin
        new_item["cell_xmax"] = xmax
        new_item["cell_ymax"] = ymax
        mapped.append(new_item)
    return mapped


def map_cells_from_scaled(
    cells: list[Cell],
    scale_ratio: float,
) -> list[Cell]:
    if abs(scale_ratio - 1.0) < 1e-9:
        return cells
    mapped: list[Cell] = []
    for c in cells:
        xmin, ymin, xmax, ymax = map_bbox_from_scaled(
            c.cell_xmin, c.cell_ymin, c.cell_xmax, c.cell_ymax,
            scale_ratio,
        )
        mapped.append(replace(
            c,
            cell_xmin=xmin,
            cell_ymin=ymin,
            cell_xmax=xmax,
            cell_ymax=ymax,
        ))
    return mapped


def _cells_to_cell_list(cells: list[Cell]) -> list[dict[str, Any]]:
    return [{
        "cell_xmin": c.cell_xmin, "cell_ymin": c.cell_ymin,
        "cell_xmax": c.cell_xmax, "cell_ymax": c.cell_ymax,
        "tops": [{
            "cell_type": c.cell_type, "cell_type_name": c.cell_type_name,
            "class_confidence": c.class_confidence,
            "bbox_confidence": c.bbox_confidence,
        }],
    } for c in cells]


def _cell_dict_to_cell(cell_dict: dict[str, Any]) -> Cell:
    """cell_list 单项 → Cell（局部坐标，供 dedup_cells_across_tiles 使用）。"""
    tops = cell_dict.get("tops") or []
    top0 = tops[0] if tops and isinstance(tops[0], dict) else {}
    extra = dict(cell_dict.get("extra") or {})
    if len(tops) > 1:
        extra["_tops"] = tops
    return Cell(
        cell_xmin=int(cell_dict["cell_xmin"]),
        cell_ymin=int(cell_dict["cell_ymin"]),
        cell_xmax=int(cell_dict["cell_xmax"]),
        cell_ymax=int(cell_dict["cell_ymax"]),
        cell_type=int(top0.get("cell_type", cell_dict.get("cell_type", 0))),
        cell_type_name=str(top0.get("cell_type_name", cell_dict.get("cell_type_name", ""))),
        class_confidence=float(
            top0.get("class_confidence", cell_dict.get("class_confidence", 1.0)) or 1.0
        ),
        bbox_confidence=float(
            top0.get("bbox_confidence", cell_dict.get("bbox_confidence", 1.0)) or 1.0
        ),
        extra=extra,
    )


def _cell_to_global_dict(cell: Cell, ox: int, oy: int) -> dict[str, Any]:
    """去重后的 Cell（局部坐标）→ 全图 cell_list 项。"""
    saved_tops = cell.extra.get("_tops") if cell.extra else None
    if saved_tops:
        tops = saved_tops
    else:
        tops = [{
            "cell_type": cell.cell_type,
            "cell_type_name": cell.cell_type_name,
            "class_confidence": float(cell.class_confidence),
            "bbox_confidence": float(cell.bbox_confidence),
        }]
    item: dict[str, Any] = {
        "cell_xmin": int(cell.cell_xmin) + ox,
        "cell_ymin": int(cell.cell_ymin) + oy,
        "cell_xmax": int(cell.cell_xmax) + ox,
        "cell_ymax": int(cell.cell_ymax) + oy,
        "tops": tops,
    }
    extra = {k: v for k, v in (cell.extra or {}).items() if k != "_tops"}
    if extra:
        item["extra"] = extra
    return item


def _build_tiles_for_dedup_grid(
    ys: list[tuple[int, int]],
    xs: list[tuple[int, int]],
    cell_lists: dict[tuple[int, int, int, int], list[dict[str, Any]]],
) -> list[Tile]:
    """按切图网格构建 Tile，row_index/col_index 与 dedup_cells_across_tiles 邻接关系对齐。"""
    tiles: list[Tile] = []
    for row_idx, (y0, y1) in enumerate(ys):
        for col_idx, (x0, x1) in enumerate(xs):
            key = (y0, y1, x0, x1)
            cell_dicts = cell_lists.get(key, [])
            tiles.append(Tile(
                image_uid=f"{row_idx}_{col_idx}",
                w=x1 - x0,
                h=y1 - y0,
                x=x0,
                y=y0,
                meta={"row_index": row_idx, "col_index": col_idx},
                cells=[_cell_dict_to_cell(d) for d in cell_dicts],
            ))
    return tiles


def _tiles_to_global_cell_list(tiles: list[Tile]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for tile in tiles:
        ox = int(tile.x or 0)
        oy = int(tile.y or 0)
        for cell in tile.cells or []:
            merged.append(_cell_to_global_dict(cell, ox, oy))
    return merged


def dedup_tiled_x100_results(
    ys: list[tuple[int, int]],
    xs: list[tuple[int, int]],
    cell_lists: dict[tuple[int, int, int, int], list[dict[str, Any]]],
    *,
    tile_w: int,
    tile_h: int,
    iou_thresh: float = 0.2,
) -> list[dict[str, Any]]:
    """分块 cell_list 经 Tile 适配后做跨块 NMS 去重，返回全图坐标 cell_list。"""
    tiles = _build_tiles_for_dedup_grid(ys, xs, cell_lists)
    before = sum(len(t.cells or []) for t in tiles)
    if before == 0:
        return []
    dedup_cells_across_tiles_per_type(
        tiles,
        tile_w=tile_w,
        tile_h=tile_h,
        iou_thresh=iou_thresh,
        ios_thresh=0.7,
    )
    merged = _tiles_to_global_cell_list(tiles)
    logger.info(
        "x100 tiled dedup: %d tiles, %d -> %d cells (iou_thresh=%.2f)",
        len(tiles), before, len(merged), iou_thresh,
    )
    return merged


def infer_x100_on_bgr(
    bgr: np.ndarray,
    *,
    infer_dpi: int,
    smear_type: str,
    target_cell_types: str,
    filename: str,
    gpu_id: int,
    max_w: int,
    max_h: int,
    test: bool = False,
    resolved: Any = None,
    route_dpi: int | None = None,
    no_cls: bool = False,
) -> dict[str, Any]:
    """对单张 BGR 图推理：必要时按模型最大尺寸分块，返回 cell_list / cells。"""
    h, w = int(bgr.shape[0]), int(bgr.shape[1])

    def _run_infer(tile_bytes: bytes) -> dict[str, Any]:
        # resolved / route_dpi 仅小程序强制档位时传入；平扫和单张保持原来的选模。
        extra: dict[str, Any] = {}
        if resolved is not None:
            extra["resolved"] = resolved
        if route_dpi is not None:
            extra["route_dpi"] = route_dpi
        if no_cls:
            extra["no_cls"] = True
        return infer(
            tile_bytes,
            dpi=infer_dpi,
            smear_type=smear_type,
            algorithm_types=target_cell_types or "",
            filename=filename,
            gpu_id=gpu_id,
            test=test,
            **extra,
        )

    # max_w/max_h <= 0 表示模型无尺寸上限，整图直接推理
    if max_w <= 0 or max_h <= 0:
        return _run_infer(encode_bgr_jpeg(bgr))

    if w <= max_w and h <= max_h:
        return _run_infer(encode_bgr_jpeg(bgr))

    ys = tile_ranges_1d(h, max_h, DEFAULT_TILE_OVERLAP)
    xs = tile_ranges_1d(w, max_w, DEFAULT_TILE_OVERLAP)
    tile_count = len(ys) * len(xs)
    logger.info(
        "x100 tiled infer: %dx%d -> %d tiles (overlap=%d, max=%dx%d)",
        w, h, tile_count, DEFAULT_TILE_OVERLAP, max_w, max_h,
    )

    cell_lists: dict[tuple[int, int, int, int], list[dict[str, Any]]] = {}
    warning: str | None = None
    scores_out: list[Any] = []
    wbc_pixel_count = 0
    red_pixel_count = 0
    for y0, y1 in ys:
        for x0, x1 in xs:
            crop = bgr[y0:y1, x0:x1]
            if crop.size == 0:
                cell_lists[(y0, y1, x0, x1)] = []
                continue
            part = _run_infer(encode_bgr_jpeg(crop))
            warning = warning or part.get("warning")
            raw_list = part.get("cell_list") or []
            cells = part.get("cells") or []
            cl: list[dict[str, Any]] = [x for x in raw_list if isinstance(x, dict)]
            if not cl and cells:
                cl = _cells_to_cell_list(cells)
            cell_lists[(y0, y1, x0, x1)] = cl
            wbc_pixel_count += int(part.get("wbc_pixel_count") or 0)
            red_pixel_count += int(part.get("red_pixel_count") or 0)
            for row in part.get("scores") or []:
                if isinstance(row, (list, tuple)) and len(row) >= 4:
                    mapped = list(row)
                    mapped[0] = float(mapped[0]) + x0
                    mapped[1] = float(mapped[1]) + y0
                    mapped[2] = float(mapped[2]) + x0
                    mapped[3] = float(mapped[3]) + y0
                    scores_out.append(mapped)
                else:
                    scores_out.append(row)

    merged_list = dedup_tiled_x100_results(
        ys, xs, cell_lists, tile_w=max_w, tile_h=max_h,
    )

    result: dict[str, Any] = {
        "cell_list": merged_list,
        "cells": [_cell_dict_to_cell(d) for d in merged_list if isinstance(d, dict)],
        "scores": scores_out,
        "wbc_pixel_count": wbc_pixel_count,
        "red_pixel_count": red_pixel_count,
    }
    if warning:
        result["warning"] = warning
    return result


def prepare_x100_bgr(
    image_bytes: bytes,
    input_dpi: int,
    actual_dpi: int,
    *,
    allow_dpi_scale: bool = True,
    scale_ratio: float | None = None,
) -> tuple[np.ndarray, int, int, float, int, int, int, int, int]:
    """
    解码、按 DPI 比值缩放，并在必要时黑底居中填充至模型最小尺寸。
    返回 (bgr, orig_w, orig_h, scale_ratio, model_dpi, max_w, max_h, pad_x, pad_y)。
    scale_ratio 已给定时直接使用，不再按 dpi_range 重算。
    """
    model_dpi = int(actual_dpi)
    limits = model_tile_limits(model_dpi)
    if limits is None:
        max_w, max_h = 0, 0
    else:
        max_w, max_h, min_w, min_h = limits

    bgr = decode_image_bgr(image_bytes)
    if bgr is None:
        raise ValueError("cannot decode image")

    orig_h, orig_w = int(bgr.shape[0]), int(bgr.shape[1])
    if scale_ratio is None:
        scale_ratio = compute_dpi_scale_ratio(input_dpi, model_dpi) if allow_dpi_scale else 1.0
    else:
        scale_ratio = float(scale_ratio)
    if scale_ratio != 1.0:
        bgr = scale_bgr(bgr, scale_ratio)

    pad_x, pad_y = 0, 0
    if limits is not None:
        _, _, min_w, min_h = limits
        bgr, pad_x, pad_y = pad_bgr_to_min(bgr, min_w, min_h)

    return bgr, orig_w, orig_h, scale_ratio, model_dpi, max_w, max_h, pad_x, pad_y


def _bbox_key(d: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(d["cell_xmin"]),
        int(d["cell_ymin"]),
        int(d["cell_xmax"]),
        int(d["cell_ymax"]),
    )


def _sync_cells_and_cell_list(
    cells: list[Cell],
    cell_list: list[dict[str, Any]],
) -> tuple[list[Cell], list[dict[str, Any]]]:
    if not cell_list and cells:
        cell_list = _cells_to_cell_list(cells)
    if not cells and cell_list:
        cells = [_cell_dict_to_cell(d) for d in cell_list if isinstance(d, dict)]
    return cells, cell_list


def _apply_post_filters(
    cell_list: list[dict[str, Any]],
    *,
    orig_w: int,
    orig_h: int,
    input_dpi: int,
    use_714756_filter: bool,
    edge_cell_filter: bool,
) -> list[dict[str, Any]]:
    if not cell_list:
        return cell_list
    if use_714756_filter:
        try:
            cell_list = filter_cell_dicts_edge_elongated_1pct(cell_list, orig_w, orig_h)
            cell_list = filter_cell_dicts_small_wbc_714756(cell_list, input_dpi)
        except Exception as e:
            logger.warning("714756 cell filter skipped: %s", e)
        return cell_list
    if edge_cell_filter:
        try:
            cell_list = filter_cell_dicts_edge_incomplete(cell_list, orig_w, orig_h)
        except Exception as e:
            logger.warning("edge_cell_filter skipped: %s", e)
    return cell_list


def run_cell_image_infer(
    image_bytes: bytes,
    dpi: int,
    smear_type: str,
    target_cell_types: str,
    filename: str = "image.jpg",
    *,
    edge_cell_filter: bool = True,
    test: bool = False,
    gpu_id: int | None = None,
    ensure_loaded: bool = True,
    allow_dpi_scale: bool = True,
    force_model_dpi: int | None = None,
    include_single_only: bool = False,
    no_cls: bool = False,
    apply_post_filters: bool = True,
) -> dict[str, Any]:
    """
    平扫 upload_image 与单张 get_task_result_x100 共用：
    选模型 → 缩放/填充 → 加载 → 推理（大图切块）→ 坐标回映射 → 过滤。
    成功 ok=True；失败 ok=False 且 error 为原因。

    ensure_loaded: True 时推理前 load_models（已 READY 的跳过）。平扫瓦片也要开，
    否则推理服务重启后 create_task 的预热失效，upload_image 会报模型未加载。
    allow_dpi_scale: 平扫传 False，尺寸合适时原图直送，跳过解码/缩放/重编码。
    force_model_dpi: 忽略 dpi_range，只加载该 actual_dpi 的模型，并按比值强制缩放。
    include_single_only: 单张识别才加载 single_only 分类器（如 LOWRES-WBC-CLS）。
    平扫保持 False，只定位。可选分类器不存在时跳过，结果退回定位。
    no_cls: 为 True 时请求带 no_cls=true，只返回定位框。
    apply_post_filters: 为 False 时不做靠边过滤和 714756 过小有核过滤。小程序传 False。
    """
    try:
        assert_inference_allowed()
    except PipelineUnavailable as e:
        wait_while_circuit_open()
        return {"ok": False, "error": str(e)}
    if test:
        url = None
        try:
            gid, endpoint = resolve_triton_route(gpu_id)
            base = (endpoint.get("pipeline_base_url") or "").rstrip("/")
            url = f"{base}/infer" if base else None
        except PipelineUnavailable as e:
            wait_while_circuit_open()
            return {"ok": False, "error": str(e)}
        trip_on_timeout(gpu_id=gid, url=url)
        return {
            "ok": False,
            "error": "测试熔断：已触发全部推理服务重启，暂不接收图片推理",
        }
    input_dpi = int(dpi)
    smear_type = smear_type or "BM"
    target_cell_types = target_cell_types or ""
    forced_dpi = int(force_model_dpi) if force_model_dpi is not None else None

    if forced_dpi is not None:
        resolved = resolve_models_by_actual_dpi(
            forced_dpi,
            smear_type,
            target_cell_types,
            include_single_only=include_single_only,
        )
    else:
        resolved = resolve_models(
            input_dpi,
            smear_type,
            target_cell_types,
            include_single_only=include_single_only,
        )
        if resolved.dpi_unsuitable:
            return {"ok": False, "error": DPI_NOT_SUITABLE}
    requested = _parse_cell_types(target_cell_types)
    if not requested:
        return {"ok": False, "error": "target_cell_types cannot be empty"}
    covered: set[str] = set()
    for spec in resolved.detection:
        covered |= spec.targets
    for spec in resolved.score:
        covered |= spec.targets
    missing = requested - covered
    if missing:
        st = normalize_smear_type(smear_type)
        dpi_label = forced_dpi if forced_dpi is not None else input_dpi
        return {
            "ok": False,
            "error": (
                f"Invalid combo: DPI={dpi_label} smear_type={st} has no detection model for "
                f"{sorted(missing)}"
            ),
        }
    route_specs = resolved.detection or resolved.score
    if not route_specs:
        return {"ok": False, "error": DPI_NOT_SUITABLE}

    model_names = ",".join(spec.name for spec in resolved.specs)
    model_dpi = forced_dpi if forced_dpi is not None else int(route_specs[0].actual_dpi)
    model_warning = resolved.warning
    limits = model_tile_limits(model_dpi)
    if limits is None:
        max_w, max_h, min_w, min_h = 0, 0, 0, 0
    else:
        max_w, max_h, min_w, min_h = limits

    if forced_dpi is not None:
        scale_ratio = compute_force_dpi_scale_ratio(input_dpi, model_dpi)
    else:
        scale_ratio = compute_dpi_scale_ratio(input_dpi, model_dpi) if allow_dpi_scale else 1.0
    peeked = peek_image_size(image_bytes)
    use_original_bytes = False
    pad_x, pad_y = 0, 0
    orig_w = orig_h = 0
    bgr: np.ndarray | None = None
    if peeked is not None:
        orig_w, orig_h = peeked
        use_original_bytes = not infer_needs_decode(
            orig_w, orig_h, scale_ratio, max_w, max_h, min_w, min_h,
        )

    if not use_original_bytes:
        try:
            bgr, orig_w, orig_h, scale_ratio, model_dpi, max_w, max_h, pad_x, pad_y = prepare_x100_bgr(
                image_bytes,
                input_dpi,
                model_dpi,
                allow_dpi_scale=allow_dpi_scale,
                scale_ratio=scale_ratio,
            )
        except ValueError as e:
            return {"ok": False, "error": str(e)}

    infer_dpi = model_dpi if (forced_dpi is not None or scale_ratio != 1.0) else input_dpi
    # 把已解析（并去掉没加载上的可选分类器）的模型传给 infer，避免内部按 DPI 重选时把单张分类器丢掉或给平扫加上。
    force_kwargs: dict[str, Any] = {
        "resolved": resolved,
        "route_dpi": model_dpi,
    }
    if no_cls:
        force_kwargs["no_cls"] = True
    use_714756_filter = any(spec.actual_dpi == 714756 for spec in resolved.detection)

    if gpu_id is None:
        gpu_id, _ = resolve_triton_route()
    if ensure_loaded:
        ok, load_err, ready_names = load_resolved_models(resolved, gpu_id=gpu_id)
        if not ok:
            return {"ok": False, "error": load_err}
        skipped = {name for name in resolved.names if name not in set(ready_names)}
        if skipped:
            logger.info("optional models not loaded, detection only: %s", sorted(skipped))
            resolved = resolved.without_names(skipped)
            model_names = ",".join(spec.name for spec in resolved.specs)
        force_kwargs["resolved"] = resolved

    try:
        if use_original_bytes:
            result = infer(
                image_bytes,
                dpi=infer_dpi,
                smear_type=smear_type,
                algorithm_types=target_cell_types,
                filename=filename,
                gpu_id=gpu_id,
                test=test,
                **force_kwargs,
            )
        else:
            result = infer_x100_on_bgr(
                bgr,
                infer_dpi=infer_dpi,
                smear_type=smear_type,
                target_cell_types=target_cell_types,
                filename=filename,
                gpu_id=gpu_id,
                max_w=max_w,
                max_h=max_h,
                test=test,
                **force_kwargs,
            )
    except PipelineUnavailable as e:
        wait_while_circuit_open()
        return {"ok": False, "error": str(e)}
    except Exception as e:
        logger.exception("Triton infer failed: %s", e)
        return {"ok": False, "error": str(e)}

    cells = list(result.get("cells") or [])
    cell_list = [x for x in (result.get("cell_list") or []) if isinstance(x, dict)]
    if pad_x or pad_y:
        cells = map_cells_from_pad(cells, pad_x, pad_y)
        cell_list = map_cell_list_from_pad(cell_list, pad_x, pad_y)
    if scale_ratio != 1.0:
        cells = map_cells_from_scaled(cells, scale_ratio)
        cell_list = map_cell_list_from_scaled(cell_list, scale_ratio)
    cells, cell_list = _sync_cells_and_cell_list(cells, cell_list)
    if apply_post_filters:
        cell_list = _apply_post_filters(
            cell_list,
            orig_w=orig_w,
            orig_h=orig_h,
            input_dpi=input_dpi,
            use_714756_filter=use_714756_filter,
            edge_cell_filter=False if use_714756_filter else edge_cell_filter,
        )
        keep = {_bbox_key(d) for d in cell_list}
        cells = [
            c for c in cells
            if (int(c.cell_xmin), int(c.cell_ymin), int(c.cell_xmax), int(c.cell_ymax)) in keep
        ]

    warning = model_warning or result.get("warning")
    return {
        "ok": True,
        "cells": cells,
        "cell_list": cell_list,
        "scores": result.get("scores") or [],
        "wbc_pixel_count": int(result.get("wbc_pixel_count") or 0),
        "red_pixel_count": int(result.get("red_pixel_count") or 0),
        "orig_w": orig_w,
        "orig_h": orig_h,
        "warning": warning,
        "model_name": model_names,
    }


def _miniapp_skip_broken_first_top(
    cell_list: list[dict[str, Any]],
    smear_type: str,
) -> list[dict[str, Any]]:
    """第一名是破碎细胞及杂质且还有第二名时，把第二名提到第一位。"""
    for item in cell_list:
        if not isinstance(item, dict):
            continue
        tops = item.get("tops")
        if not isinstance(tops, list) or len(tops) < 2:
            continue
        first, second = tops[0], tops[1]
        if not isinstance(first, dict) or not isinstance(second, dict):
            continue
        try:
            first_type = int(first.get("cell_type"))
            second_type = int(second.get("cell_type"))
        except (TypeError, ValueError):
            continue
        if first_type != _MINIAPP_BROKEN_CELL_TYPE:
            continue
        promoted = dict(second)
        promoted["count_type"] = get_counting_cell_type(second_type, smear_type)
        item["tops"] = [promoted, first, *tops[2:]]
    return cell_list


def _miniapp_estimated_dpi(cell_list: list[dict[str, Any]]) -> float | None:
    """原图细胞框宽、高各自取中位数，再平均，按 20 微米反推 DPI。"""
    widths: list[float] = []
    heights: list[float] = []
    for item in cell_list:
        if not isinstance(item, dict):
            continue
        try:
            width = float(item["cell_xmax"]) - float(item["cell_xmin"])
            height = float(item["cell_ymax"]) - float(item["cell_ymin"])
        except (KeyError, TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue
        widths.append(width)
        heights.append(height)
    if not widths:
        return None
    side_px = (float(np.median(widths)) + float(np.median(heights))) / 2.0
    if side_px <= 0:
        return None
    # DPI = 25.4 * 1000 * 像素 / 微米
    return 25.4 * 1000.0 * side_px / _MINIAPP_CELL_UM


def run_miniapp_cell_image_infer(
    image_bytes: bytes,
    smear_type: str = "BM",
    target_cell_types: str = "WBC",
    filename: str = "image.jpg",
    *,
    gpu_id: int | None = None,
) -> dict[str, Any]:
    """
    小程序单张识别：不接收 DPI，也不受单张接口的 dpi_range 缩放限制。
    用 714756 模型试原档、一半、1/4 和 2 倍，用 40 倍模型试 147246、一半、1/4 和 2 倍。
    返回细胞更多的一次；数量相同按 714756、714756/2、714756/4、714756*2、147246、147246/2、147246/4、147246*2 优先。
    先并发全部 714756 模型档，这批都返回后再并发全部 147246 模型档。
    低倍分类模型不存在时，该档只做定位。
    若某细胞 tops 第一名是 200029（破碎细胞及杂质）且有第二名，改用第二名。
    用最终结果原图细胞框的宽、高中位数均值，按细胞 20 微米反推 DPI；
    反推 DPI 小于等于 350000 时 warning 提示图像像素不足，否则响应不含该字段。
    猜档请求带 no_cls=true，只比较定位框数量；确定档位后再请求一次完整分类结果。
    小程序不做靠边过滤，也不做 714756 小于 6 微米的有核过滤。
    """
    best: dict[str, Any] | None = None
    best_dpi: int | None = None
    best_model: int | None = None
    best_count = -1
    last_error: dict[str, Any] | None = None
    smear_type = smear_type or "BM"
    target_cell_types = target_cell_types or "WBC"
    if gpu_id is None:
        gpu_id, _ = resolve_triton_route()

    def _guess_one(item: tuple[int, int]) -> tuple[int, int, dict[str, Any]]:
        guess_dpi, model_dpi = item
        result = run_cell_image_infer(
            image_bytes,
            guess_dpi,
            smear_type,
            target_cell_types,
            filename=filename,
            gpu_id=gpu_id,
            include_single_only=True,
            force_model_dpi=model_dpi,
            no_cls=True,
            apply_post_filters=False,
        )
        return int(guess_dpi), int(model_dpi), result

    groups: list[list[tuple[int, int]]] = []
    for guess_dpi, model_dpi in MINIAPP_GUESSES:
        if not groups or groups[-1][0][1] != model_dpi:
            groups.append([])
        groups[-1].append((guess_dpi, model_dpi))

    for group in groups:
        with ThreadPoolExecutor(max_workers=len(group)) as pool:
            outcomes = list(pool.map(_guess_one, group))
        for guess_dpi, model_dpi, result in outcomes:
            if not result.get("ok"):
                last_error = result
                logger.info(
                    "miniapp guess_dpi=%s model=%s failed: %s",
                    guess_dpi,
                    model_dpi,
                    result.get("error"),
                )
                continue
            count = len(result.get("cell_list") or [])
            logger.info(
                "miniapp guess_dpi=%s model=%s cells=%s",
                guess_dpi,
                model_dpi,
                count,
            )
            if best is None or count > best_count:
                best = result
                best_dpi = int(guess_dpi)
                best_model = int(model_dpi)
                best_count = count

    if best is None or best_dpi is None or best_model is None:
        return last_error or {"ok": False, "error": "infer failed"}

    logger.info("miniapp pick dpi=%s model=%s cells=%s", best_dpi, best_model, best_count)
    best = run_cell_image_infer(
        image_bytes,
        best_dpi,
        smear_type,
        target_cell_types,
        filename=filename,
        gpu_id=gpu_id,
        include_single_only=True,
        force_model_dpi=best_model,
        apply_post_filters=False,
    )
    if not best.get("ok"):
        return best
    best["cell_list"] = _miniapp_skip_broken_first_top(
        list(best.get("cell_list") or []),
        smear_type,
    )
    best["guess_dpi"] = best_dpi
    best["model_dpi"] = best_model
    est_dpi = _miniapp_estimated_dpi(list(best.get("cell_list") or []))
    best["est_dpi"] = int(round(est_dpi)) if est_dpi is not None else None
    if est_dpi is not None and est_dpi <= _MINIAPP_DPI_WARN_MAX:
        logger.info("miniapp estimated_dpi=%.0f pixel warning", est_dpi)
        prev = str(best.get("warning") or "").strip()
        if prev and prev != _MINIAPP_LOWRES_WARNING:
            best["warning"] = f"{prev}；{_MINIAPP_LOWRES_WARNING}"
        else:
            best["warning"] = _MINIAPP_LOWRES_WARNING
    else:
        logger.info("miniapp estimated_dpi=%s no pixel warning", est_dpi)
        warning = best.get("warning")
        if not warning or warning == _MINIAPP_LOWRES_WARNING:
            best.pop("warning", None)
    return best
