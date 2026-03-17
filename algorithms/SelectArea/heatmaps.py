# heatmaps.py ：热力图算法只调用 adapter，不直接解析 meta
from __future__ import annotations

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2] 
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from project.tiles import Tile

from dataclasses import dataclass
from typing import List, Optional, Tuple

import math
import numpy as np

from .adapters import TileAdapter, DefaultScoresAdapter
from .config import BM40Config
from .data_structure import Rect, ScoreRegion





@dataclass
class HeatmapGrid:
    """
    origin_x, origin_y: 网格左上角对应的全局坐标
    cell_size: 每个网格单元边长
    values: 累计 score*weight
    weights: 累计 weight
    """
    origin_x: float
    origin_y: float
    cell_size: float
    values: np.ndarray
    weights: np.ndarray

    def global_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        col = int(math.floor((x - self.origin_x) / self.cell_size))
        row = int(math.floor((y - self.origin_y) / self.cell_size))
        return col, row  

    def grid_to_global(self, col: int, row: int) -> Tuple[float, float]:
        x = self.origin_x + col * self.cell_size
        y = self.origin_y + row * self.cell_size
        return x, y

    def add_rect_value(self, rect: Rect, score: float) -> None:
        """
        把一个矩形的分值写入网格
        """
        if rect.w <= 0 or rect.h <= 0:
            return

        rows, cols = self.values.shape

        c0 = int(math.floor((rect.x - self.origin_x) / self.cell_size))
        r0 = int(math.floor((rect.y - self.origin_y) / self.cell_size))
        c1 = int(math.ceil((rect.x2 - self.origin_x) / self.cell_size))
        r1 = int(math.ceil((rect.y2 - self.origin_y) / self.cell_size))

        c0 = max(0, min(cols, c0))
        c1 = max(0, min(cols, c1))
        r0 = max(0, min(rows, r0))
        r1 = max(0, min(rows, r1))

        if r0 >= r1 or c0 >= c1:
            return

        cell = float(self.cell_size)
        cell_area = cell * cell

        # 行/列索引
        rr = np.arange(r0, r1, dtype=np.int32)
        cc = np.arange(c0, c1, dtype=np.int32)

        # 每行对应的 cell y0/y1
        y0 = self.origin_y + rr.astype(np.float32) * cell
        y1 = y0 + cell

        # 每列对应的 cell x0/x1
        x0 = self.origin_x + cc.astype(np.float32) * cell
        x1 = x0 + cell

        # y 方向重叠高度：oh.shape = (r1-r0,)
        oh = np.minimum(y1, rect.y2) - np.maximum(y0, rect.y)
        oh = np.clip(oh, 0.0, None)

        # x 方向重叠宽度：ow.shape = (c1-c0,)
        ow = np.minimum(x1, rect.x2) - np.maximum(x0, rect.x)
        ow = np.clip(ow, 0.0, None)

        # overlap_area.shape = (r1-r0, c1-c0)
        overlap_area = oh[:, None] * ow[None, :]
        w = overlap_area / cell_area  # 权重矩阵

        # 直接对切片累加（NumPy 在 C 层循环）
        self.values[r0:r1, c0:c1] += (score * w).astype(self.values.dtype, copy=False)
        self.weights[r0:r1, c0:c1] += w.astype(self.weights.dtype, copy=False)

    def finalize(self, fill_value: float = np.nan) -> np.ndarray:
        out = np.full_like(self.values, fill_value, dtype=np.float32)
        mask = self.weights > 0
        out[mask] = (self.values[mask] / self.weights[mask]).astype(np.float32)
        return out


def compute_global_bounds_from_tiles(tiles: List[Tile]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    xe: List[float] = []
    ye: List[float] = []

    for t in tiles:
        if t.x is None or t.y is None:
            raise ValueError(f"tile missing global position x/y, image_uid={t.image_uid}")
        xs.append(float(t.x))
        ys.append(float(t.y))
        xe.append(float(t.x) + float(t.w))
        ye.append(float(t.y) + float(t.h))

    return min(xs), min(ys), max(xe), max(ye)


def build_score_heatmap(
    tiles: List[Tile],
    *,
    config: BM40Config,
    adapter: Optional[TileAdapter] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> HeatmapGrid:
    """
    算法层不再关心 meta["scores"] 的结构，只通过 adapter 取 ScoreRegion。
    当上游 scores 格式变化：只改 adapter。
    """
    if config.cell_size <= 0:
        raise ValueError("cell_size must be > 0")

    if adapter is None:
        adapter = DefaultScoresAdapter()

    if bounds is None:
        min_x, min_y, max_x, max_y = compute_global_bounds_from_tiles(tiles)
    else:
        min_x, min_y, max_x, max_y = bounds

    width = max_x - min_x
    height = max_y - min_y
    cols = int(math.ceil(width / config.cell_size))
    rows = int(math.ceil(height / config.cell_size))

    values = np.zeros((rows, cols), dtype=np.float32)
    weights = np.zeros((rows, cols), dtype=np.float32)

    grid = HeatmapGrid(
        origin_x=min_x,
        origin_y=min_y,
        cell_size=config.cell_size,
        values=values,
        weights=weights,
    )

    for tile in tiles:
        regions: List[ScoreRegion] = adapter.get_score_regions(tile, config=config)
        # region 局部 -> 全局 rect
        for r in regions:
            rect = Rect(
                x=float(tile.x) + r.lx,
                y=float(tile.y) + r.ly,
                w=r.w,
                h=r.h,
            )
            grid.add_rect_value(rect, r.score)

    return grid




def build_cell_count_grid(
    tiles: List[Tile],
    grid_info: HeatmapGrid,
    *,
    config: BM40Config,
) -> np.ndarray:
    """
    根据已有的 HeatmapGrid 结构，创建一个对齐的细胞数量矩阵。
    
    参数:
    - tiles: 包含细胞检测结果的 Tile 列表
    - grid_info: 已经创建好的 HeatmapGrid 对象（用于获取 origin 和 cell_size）
    
    返回:
    - cell_count_matrix: np.ndarray (shape 与 grid_info.values 一致)
    """
    rows, cols = grid_info.values.shape
    cell_count_matrix = np.zeros((rows, cols), dtype=np.float32)

    for tile in tiles:
        if tile.x is None or tile.y is None:
            continue
            
        for cell in tile.cells:
            # 先按细胞类型过滤：只统计配置中的 WBC 类型
            if getattr(cell, "cell_type", None) != config.WBC_cell_type:
                continue

            # 1. 属性访问 (cell.xxx)
            local_cx = (cell.cell_xmin + cell.cell_xmax) / 2.0
            local_cy = (cell.cell_ymin + cell.cell_ymax) / 2.0
            
            # 2. 转换为全局坐标
            global_cx = tile.x + local_cx
            global_cy = tile.y + local_cy
            
            # 3. 映射索引 (注意 global_to_grid 返回 col, row)
            g_col, g_row = grid_info.global_to_grid(global_cx, global_cy)
            
            # 4. 边界检查并填充 [row, col]
            if 0 <= g_row < rows and 0 <= g_col < cols:
                cell_count_matrix[g_row, g_col] += 1

                
    return cell_count_matrix