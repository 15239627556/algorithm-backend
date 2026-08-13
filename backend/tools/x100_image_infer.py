# -*- coding: utf-8 -*-
"""单张 x100 识别：DPI 缩放、大图分块推理、坐标映射回原图。"""
from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Any

import cv2
import numpy as np

from backend.tools.combo_validator import TOLERANCE
from backend.tools.image_tiling import (
    DEFAULT_TILE_OVERLAP,
    tile_ranges_1d,
)
from backend.tools.triton_client import (
    MODEL_144750,
    MODEL_357378,
    MODEL_714756_BM,
    MODEL_714756_CF,
    infer,
)
from project.cells import Cell
from project.tiles import Tile
from algorithms.SelectArea.dedup_cells_across_tiles import dedup_cells_across_tiles_per_type

logger = logging.getLogger(__name__)

_MODEL_TILE_LIMITS: dict[str, tuple[int, int]] = {
    MODEL_144750: (3200, 2200),
    MODEL_357378: (2448, 2048),
    MODEL_714756_BM: (4896, 4096),
    MODEL_714756_CF: (4896, 4096),
}


def extract_model_dpi(model_name: str) -> int:
    """从 model_name 提取模型对应 DPI（147246 映射为 144750）。"""
    m = re.search(r"DPI(\d+)", model_name)
    if not m:
        raise ValueError(f"cannot extract DPI from model name: {model_name}")
    val = int(m.group(1))
    if val == 147246:
        return 144750
    return val


def model_tile_limits(model_name: str) -> tuple[int, int]:
    limits = _MODEL_TILE_LIMITS.get(model_name)
    if limits is None:
        model_dpi = extract_model_dpi(model_name)
        return _MODEL_TILE_LIMITS.get(
            MODEL_714756_BM if model_dpi == 714756 else MODEL_357378,
            (4896, 4096),
        )
    return limits


def dpi_needs_scale(input_dpi: int, model_dpi: int) -> bool:
    low = int(model_dpi * (1 - TOLERANCE))
    high = int(model_dpi * (1 + TOLERANCE))
    return not (low <= input_dpi <= high)


def compute_dpi_scale_ratio(input_dpi: int, model_dpi: int) -> float:
    """DPI 超出 ±10% 时返回 model_dpi/input_dpi，否则返回 1.0。"""
    if not dpi_needs_scale(input_dpi, model_dpi):
        return 1.0
    return model_dpi / input_dpi


def encode_bgr_jpeg(bgr: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok or buf is None:
        raise RuntimeError("cv2.imencode failed")
    return bytes(buf)


def decode_image_bgr(image_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def scale_bgr(bgr: np.ndarray, scale_ratio: float) -> np.ndarray:
    if abs(scale_ratio - 1.0) < 1e-9:
        return bgr
    h, w = bgr.shape[:2]
    new_w = max(1, int(round(w * scale_ratio)))
    new_h = max(1, int(round(h * scale_ratio)))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


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
) -> dict[str, Any]:
    """对单张 BGR 图推理：必要时按模型最大尺寸分块，返回 cell_list / cells。"""
    h, w = int(bgr.shape[0]), int(bgr.shape[1])

    def _run_infer(tile_bytes: bytes) -> dict[str, Any]:
        return infer(
            tile_bytes,
            dpi=infer_dpi,
            smear_type=smear_type,
            algorithm_types=target_cell_types or "",
            filename=filename,
            gpu_id=gpu_id,
        )

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

    merged_list = dedup_tiled_x100_results(
        ys, xs, cell_lists, tile_w=max_w, tile_h=max_h,
    )

    result: dict[str, Any] = {
        "cell_list": merged_list,
        "cells": [],
    }
    if warning:
        result["warning"] = warning
    return result


def prepare_x100_bgr(
    image_bytes: bytes,
    input_dpi: int,
    model_name: str,
) -> tuple[np.ndarray, int, int, float, int, int, int]:
    """
    解码并按 DPI 比值缩放图像。
    返回 (bgr, orig_w, orig_h, scale_ratio, model_dpi, max_w, max_h)。
    """
    model_dpi = extract_model_dpi(model_name)
    max_w, max_h = model_tile_limits(model_name)

    bgr = decode_image_bgr(image_bytes)
    if bgr is None:
        raise ValueError("cannot decode image")

    orig_h, orig_w = int(bgr.shape[0]), int(bgr.shape[1])
    scale_ratio = compute_dpi_scale_ratio(input_dpi, model_dpi)
    if scale_ratio != 1.0:
        bgr = scale_bgr(bgr, scale_ratio)
        logger.info(
            "x100 dpi scale: input=%d model=%d ratio=%.6f %dx%d -> %dx%d",
            input_dpi, model_dpi, scale_ratio,
            orig_w, orig_h, int(bgr.shape[1]), int(bgr.shape[0]),
        )

    return bgr, orig_w, orig_h, scale_ratio, model_dpi, max_w, max_h
