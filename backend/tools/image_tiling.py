# -*- coding: utf-8 -*-
"""大图按模型最大边长分块：切分步长 = 块边长 - 重叠像素；合并时对各块坐标加偏移。"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_TILE_OVERLAP = 30


def tile_ranges_1d(length: int, max_tile: int, overlap: int) -> List[Tuple[int, int]]:
    """沿单轴生成切片 [start, end)，end 为开区间；每段长度 <= max_tile，相邻段重叠 overlap。"""
    if length <= 0:
        return []
    if length <= max_tile:
        return [(0, length)]
    ov = max(0, overlap)
    ranges: List[Tuple[int, int]] = []
    x0 = 0
    while True:
        x1 = min(length, x0 + max_tile)
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
