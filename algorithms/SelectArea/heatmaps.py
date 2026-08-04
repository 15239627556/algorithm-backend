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
import cv2
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


def compute_search_window_pad_px(config: BM40Config) -> float:
    """最大搜索窗半对角线（像素）+ setcover_pad，用于 user_choice_area 热力图外扩。"""
    one_tile_grid_area = (config.tile_w * config.tile_h) / (config.cell_size ** 2)
    max_scale = max(config.get_search_area_scales())
    max_grid_area = max_scale * one_tile_grid_area
    max_aspect = max(
        max(rw / rh, rh / rw)
        for rw, rh in config.window_aspect_ratios
    )
    side_px = math.sqrt(max_grid_area * max_aspect) * config.cell_size
    return float(side_px) + float(config.setcover_pad)


def bounds_from_user_choice_area(config: BM40Config) -> Optional[Tuple[float, float, float, float]]:
    """由 user_choice_area 推导热力图 bounds；无选区则返回 None（全图）。"""
    area = config.user_choice_area
    if not area:
        return None
    try:
        x_min = float(area["x_min"])
        y_min = float(area["y_min"])
        x_max = float(area["x_max"])
        y_max = float(area["y_max"])
    except (KeyError, TypeError, ValueError):
        return None
    if x_max <= x_min or y_max <= y_min:
        return None

    ua_w = x_max - x_min
    ua_h = y_max - y_min
    max_pad = compute_search_window_pad_px(config)
    tile_diag = math.hypot(config.tile_w, config.tile_h)
    # 外扩不超过选区一半、一个 tile 对角线，避免 padding 把局部选区撑成近全图
    pad = min(max_pad, ua_w * 0.5, ua_h * 0.5, tile_diag)
    pad = max(pad, config.cell_size)
    return x_min - pad, y_min - pad, x_max + pad, y_max + pad


def filter_tiles_by_bounds(
    tiles: List[Tile],
    bounds: Tuple[float, float, float, float],
) -> List[Tile]:
    """保留与 bounds 相交的 tile。"""
    min_x, min_y, max_x, max_y = bounds
    out: List[Tile] = []
    for t in tiles:
        if t.x is None or t.y is None:
            continue
        tx1, ty1 = float(t.x), float(t.y)
        tx2, ty2 = tx1 + float(t.w), ty1 + float(t.h)
        if tx2 < min_x or tx1 > max_x or ty2 < min_y or ty1 > max_y:
            continue
        out.append(t)
    return out


def filter_cells_xyxy_by_bounds(
    cells_xyxy: np.ndarray,
    bounds: Tuple[float, float, float, float],
) -> np.ndarray:
    """保留中心落在 bounds 内的细胞 (N,4)。"""
    if cells_xyxy.size == 0:
        return cells_xyxy
    min_x, min_y, max_x, max_y = bounds
    cx = 0.5 * (cells_xyxy[:, 0] + cells_xyxy[:, 2])
    cy = 0.5 * (cells_xyxy[:, 1] + cells_xyxy[:, 3])
    mask = (cx >= min_x) & (cx <= max_x) & (cy >= min_y) & (cy <= max_y)
    return cells_xyxy[mask]


def centered_box_sum_map(
    arr: np.ndarray,
    w: int,
    h: int,
    border_value: float = 0.0,
) -> np.ndarray:
    """居中 w×h 窗口求和图（与 selection 中 angle=0 的 boxFilter 对齐）。"""
    rows, cols = arr.shape
    py, px = h // 2, w // 2
    padded = cv2.copyMakeBorder(
        arr, py, py, px, px, cv2.BORDER_CONSTANT, value=float(border_value),
    )
    full = cv2.boxFilter(padded, -1, (w, h), normalize=False)
    return full[py: py + rows, px: px + cols]


def crop_heatmap_grid(
    grid: HeatmapGrid,
    bounds: Tuple[float, float, float, float],
) -> HeatmapGrid:
    """从全图热力图裁剪 bounds 对应子网格（用于 user_choice_area 复用预计算热力图）。"""
    min_x, min_y, max_x, max_y = bounds
    rows, cols = grid.values.shape
    c0 = int(math.floor((min_x - grid.origin_x) / grid.cell_size))
    r0 = int(math.floor((min_y - grid.origin_y) / grid.cell_size))
    c1 = int(math.ceil((max_x - grid.origin_x) / grid.cell_size))
    r1 = int(math.ceil((max_y - grid.origin_y) / grid.cell_size))
    c0, r0 = max(0, c0), max(0, r0)
    c1, r1 = min(cols, c1), min(rows, r1)
    if c1 <= c0 or r1 <= r0:
        raise ValueError(f"crop bounds empty: {bounds}")
    return HeatmapGrid(
        origin_x=grid.origin_x + c0 * grid.cell_size,
        origin_y=grid.origin_y + r0 * grid.cell_size,
        cell_size=grid.cell_size,
        values=grid.values[r0:r1, c0:c1].copy(),
        weights=grid.weights[r0:r1, c0:c1].copy(),
    )


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
        adapter = DefaultScoresAdapter(validate=False)

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
            # 先按细胞类型过滤：统一统计配置中的 WBC 类型
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