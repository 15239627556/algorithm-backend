from __future__ import annotations
import numpy as np
from typing import List, Tuple
from .heatmaps import HeatmapGrid
from .data_structure import Rect
from .config import BM40Config


def heatmap_rect_to_abs_rect(grid_rect: Rect, grid: HeatmapGrid) -> Rect:
    """将网格坐标系下的 Rect 转换为全局物理坐标系下的 Rect"""
    abs_x, abs_y = grid.grid_to_global(int(grid_rect.y), int(grid_rect.x))
    abs_w = grid_rect.w * grid.cell_size
    abs_h = grid_rect.h * grid.cell_size
    return Rect(x=abs_x, y=abs_y, w=abs_w, h=abs_h)


def compute_head_crop(grid: HeatmapGrid, heatmap_orientation: int, config: BM40Config) -> Rect:
    """
    计算热力图头部区域的裁剪矩形（返回网格坐标系下的 Rect）。
    """
    rows, cols = grid.values.shape

    # 1. 获取有效数据蒙版
    valid_mask = grid.weights > 0
    cols_pixel_stat = np.sum(valid_mask, axis=0)

    # 2. 判定有效列 (使用配置的比例系数)
    # threshold = rows * 0.1
    threshold = rows * config.head_crop_height_threshold_ratio
    valid_col_indices = np.where(cols_pixel_stat > threshold)[0]

    if valid_col_indices.size == 0:
        return Rect(x=0.0, y=0.0, w=0.0, h=0.0)

    # 3. 计算裁剪宽度
    min_col = valid_col_indices[0]
    max_col = valid_col_indices[-1]
    available_w = (max_col - min_col) + 1
    
    # 使用配置的宽度比例 (例如 available_w * 0.2)
    crop_w_val = int(available_w * config.head_crop_width_ratio)
    crop_w = crop_w_val + min_col

    # 4. 生成 Rect
    if heatmap_orientation == 0:  # 头部在右
        return Rect(x=float(cols - crop_w), y=0.0, w=float(crop_w), h=float(rows))
    elif heatmap_orientation == 1:  # 头部在左
        return Rect(x=0.0, y=0.0, w=float(crop_w), h=float(rows))
    
    return Rect(x=0.0, y=0.0, w=0.0, h=0.0)



def generate_search_window_sizes(config: BM40Config) -> List[Tuple[int, int]]:
    """
    根据配置中的面积跨度和比例，生成网格坐标系下的搜索窗口尺寸（保证奇数以便中心对齐）。
    """
    # 1. 计算一个标准 Tile 在网格中的面积
    # Area = (2448 * 2048) / (cell_size^2)
    one_tile_grid_area = (config.tile_w * config.tile_h) / (config.cell_size ** 2)
    
    rect_sizes = []

    for scale in config.get_search_area_scales():
        # 目标总网格面积
        target_grid_area = scale * one_tile_grid_area
        
        for rw, rh in config.window_aspect_ratios:
            # 根据面积 a = (w * h) 且 w/h = rw/rh 推导：
            # w = sqrt(a * rw / rh), h = sqrt(a * rh / rw)
            w_float = np.sqrt(target_grid_area * rw / rh)
            h_float = np.sqrt(target_grid_area * rh / rw)
            
            w = int(round(w_float))
            h = int(round(h_float))

            # 保证是奇数（用于均值滤波或中心点对齐时的对称性）
            if w % 2 == 0: w += 1
            if h % 2 == 0: h += 1
            
            rect_sizes.append((w, h))

    # 去重并排序
    unique_sizes = sorted(set(rect_sizes), key=lambda x: (x[0] * x[1], x[0]))
    return unique_sizes