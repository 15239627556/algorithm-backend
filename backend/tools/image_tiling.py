# -*- coding: utf-8 -*-
"""大图按模型最大边长分块：切分步长 = 块边长 - 重叠像素；合并时对各块坐标加偏移。"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_TILE_OVERLAP = 30


def tile_ranges_1d(
    length: int,
    max_tile: int,
    overlap: int,
    min_tile: int = 0,
) -> List[Tuple[int, int]]:
    """沿单轴生成切片 [start, end)，end 为开区间。

    每段长度不超过 max_tile，相邻段重叠 overlap。
    最后一段短于 min_tile 时，把起点前移，使这段长度等于 max_tile，用更大的重叠换合格尺寸。
    """
    if length <= 0:
        return []
    if max_tile <= 0 or length <= max_tile:
        return [(0, length)]
    min_span = max(0, min(int(min_tile), int(max_tile)))
    ov = max(0, min(int(overlap), int(max_tile) - 1))
    ranges: List[Tuple[int, int]] = []
    x0 = 0
    while x0 < length:
        x1 = min(length, x0 + max_tile)
        if x1 >= length and ranges and (x1 - x0) < min_span:
            x0 = max(0, length - max_tile)
            if ranges[-1][0] == x0:
                break
            x1 = length
        ranges.append((x0, x1))
        if x1 >= length:
            break
        nxt = x1 - ov
        if nxt <= x0:
            logger.warning(
                "tile_ranges_1d: overlap too large vs max_tile, advancing x0+1"
            )
            nxt = x0 + 1
        x0 = nxt
    return ranges


def offset_cell_dict(cell: Dict[str, Any], ox: int, oy: int) -> Dict[str, Any]:
    out = copy.deepcopy(cell)
    out["cell_xmin"] = int(out["cell_xmin"]) + ox
    out["cell_ymin"] = int(out["cell_ymin"]) + oy
    out["cell_xmax"] = int(out["cell_xmax"]) + ox
    out["cell_ymax"] = int(out["cell_ymax"]) + oy
    return out


def merge_tiled_results(
    segments: List[Tuple[int, int, int, int, List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    """
    segments: (y0, y1, x0, x1, cell_list) 各块在整图中的半开区间 [y0,y1) x [x0,x1)
    将细胞坐标变换到全图后合并。
    """
    merged: List[Dict[str, Any]] = []
    for y0, _y1, x0, _x1, cells in segments:
        for c in cells:
            if isinstance(c, dict):
                merged.append(offset_cell_dict(c, x0, y0))
    return merged
