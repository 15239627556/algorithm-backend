from __future__ import annotations

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2] 
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from project.tiles import Tile

import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional

from .heatmaps import HeatmapGrid, centered_box_sum_map
from .config import BM40Config
from .data_structure import SelectionResult, Rect


def _search_one_window_angle(
    args: Tuple[
        int,
        int,
        int,
        int,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Rect,
        int,
        float,
        int,
        int,
        float,
        Optional[dict],
        Optional[dict],
    ]
) -> Optional[Tuple[int, bool, SelectionResult]]:
    (
        order_idx,
        w,
        h,
        angle,
        adjusted_score_map,
        cell_matrix,
        user_search_mask,
        head_crop_rect,
        orientation,
        heatmap_penalty_value,
        kernel_margin,
        rows,
        cols,
        box_score_cache,
        box_cell_cache,
    ) = args

    area_pixels = float(w * h)

    if angle == 0:
        cache_key = (w, h)
        if box_score_cache is not None and cache_key in box_score_cache:
            sum_scores = box_score_cache[cache_key]
            sum_cells = box_cell_cache[cache_key]
        else:
            sum_scores = centered_box_sum_map(
                adjusted_score_map, w, h, border_value=heatmap_penalty_value,
            )
            sum_cells = centered_box_sum_map(
                cell_matrix.astype(np.float32, copy=False), w, h, border_value=0.0,
            )
    else:
        kernel_size = int(np.sqrt(w**2 + h**2)) + kernel_margin
        if kernel_size % 2 == 0:
            kernel_size += 1
        pad = kernel_size // 2

        padded_scores = cv2.copyMakeBorder(
            adjusted_score_map,
            pad,
            pad,
            pad,
            pad,
            cv2.BORDER_CONSTANT,
            value=float(heatmap_penalty_value),
        )
        padded_cells = cv2.copyMakeBorder(
            cell_matrix,
            pad,
            pad,
            pad,
            pad,
            cv2.BORDER_CONSTANT,
            value=0.0,
        )

        base_mask = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        pad_w, pad_h = (kernel_size - w) // 2, (kernel_size - h) // 2
        base_mask[pad_h : pad_h + h, pad_w : pad_w + w] = 1.0

        M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), -angle, 1.0)
        rotated_kernel = cv2.warpAffine(base_mask, M, (kernel_size, kernel_size), flags=cv2.INTER_NEAREST)

        sum_scores_full = cv2.filter2D(padded_scores, -1, rotated_kernel)
        sum_cells_full = cv2.filter2D(padded_cells, -1, rotated_kernel)
        sum_scores = sum_scores_full[pad : pad + rows, pad : pad + cols]
        sum_cells = sum_cells_full[pad : pad + rows, pad : pad + cols]

    if user_search_mask is not None:
        sum_scores = sum_scores.copy()
        sum_scores[user_search_mask == 0] = -1e12

    _, max_val, _, max_loc = cv2.minMaxLoc(sum_scores)
    cx, cy = int(max_loc[0]), int(max_loc[1])
    rect_points = cv2.boxPoints(((cx, cy), (w, h), float(angle)))

    if user_search_mask is not None:
        pts_idx = rect_points.astype(np.int32)
        for px, py in pts_idx:
            if not (0 <= px < cols and 0 <= py < rows) or user_search_mask[py, px] == 0:
                return None

    # 头尾判定：对旋转后略出网格边界的顶点做裁剪，避免如 x=-1 导致误判为尾部
    cls_x = np.clip(rect_points[:, 0], 0, cols - 1)
    extreme_x = float(np.max(cls_x) if orientation == 0 else np.min(cls_x))
    in_head = head_crop_rect.x <= extreme_x <= head_crop_rect.x2

    res = SelectionResult(
        area_score=max_val / area_pixels,
        cell_count=int(sum_cells[cy, cx]),
        angle=-angle,
        center_grid=(cx, cy),
        rect_size_grid=(w, h),
        vertices_grid=rect_points,
    )
    return order_idx, in_head, res


def _pick_best_from_sum_maps(
    order_idx: int,
    w: int,
    h: int,
    angle: int,
    sum_scores: np.ndarray,
    sum_cells: np.ndarray,
    user_search_mask: Optional[np.ndarray],
    head_crop_rect: Rect,
    orientation: int,
    rows: int,
    cols: int,
) -> Optional[Tuple[int, bool, SelectionResult]]:
    area_pixels = float(w * h)
    scores = sum_scores
    if user_search_mask is not None:
        scores = sum_scores.copy()
        scores[user_search_mask == 0] = -1e12

    _, max_val, _, max_loc = cv2.minMaxLoc(scores)
    cx, cy = int(max_loc[0]), int(max_loc[1])
    rect_points = cv2.boxPoints(((cx, cy), (w, h), float(angle)))

    if user_search_mask is not None:
        pts_idx = rect_points.astype(np.int32)
        for px, py in pts_idx:
            if not (0 <= px < cols and 0 <= py < rows) or user_search_mask[py, px] == 0:
                return None

    cls_x = np.clip(rect_points[:, 0], 0, cols - 1)
    extreme_x = float(np.max(cls_x) if orientation == 0 else np.min(cls_x))
    in_head = head_crop_rect.x <= extreme_x <= head_crop_rect.x2

    res = SelectionResult(
        area_score=max_val / area_pixels,
        cell_count=int(sum_cells[cy, cx]),
        angle=-angle,
        center_grid=(cx, cy),
        rect_size_grid=(w, h),
        vertices_grid=rect_points,
    )
    return order_idx, in_head, res

def find_candidate_regions(
    grid: HeatmapGrid,
    cell_matrix: np.ndarray,
    search_rects: List[Tuple[int, int]],
    head_crop_rect: Rect,
    config: BM40Config,
    user_search_mask: Optional[np.ndarray] = None
) -> Dict[str, List[SelectionResult]]:
    """
    在热力图网格上搜索候选区域，支持用户选区强制约束。
    """
    orientation = config.heatmap_orientation   # 0: 头部在右，1: 头部在左
    
    # A. 准备基础分值图
    avg_score_map = grid.finalize(fill_value=config.heatmap_penalty_value)
    rows, cols = avg_score_map.shape

    # B. 计算边缘惩罚图
    valid_mask = (grid.weights > 0).astype(np.uint8)
    dist_map = cv2.distanceTransform(valid_mask, cv2.DIST_L2, 5)
    penalty_map = np.zeros_like(avg_score_map)
    radius = config.edge_avoidance_radius
    near_edge_mask = (dist_map < radius) & (valid_mask > 0)
    penalty_map[near_edge_mask] = config.edge_penalty_magnitude * (1.0 - dist_map[near_edge_mask] / radius)
    
    adjusted_score_map = avg_score_map + penalty_map

    # 如果有用户选区，初步压低底图分数
    if user_search_mask is not None:
        adjusted_score_map[user_search_mask == 0] = config.heatmap_penalty_value * 100

    # angle=0 窗口求和图预计算（每种 (w,h) 仅 boxFilter 一次）
    unique_sizes = set(search_rects)
    box_score_cache: dict[tuple[int, int], np.ndarray] = {}
    box_cell_cache: dict[tuple[int, int], np.ndarray] = {}
    cell_f32 = cell_matrix.astype(np.float32, copy=False)
    for w, h in unique_sizes:
        box_score_cache[(w, h)] = centered_box_sum_map(
            adjusted_score_map, w, h, border_value=config.heatmap_penalty_value,
        )
        box_cell_cache[(w, h)] = centered_box_sum_map(cell_f32, w, h, border_value=0.0)

    head_results: List[SelectionResult] = []
    tail_results: List[SelectionResult] = []
    angles = config.get_angles()
    order_idx = 0

    # angle=0：同步 + 复用求和图，避免线程池调度开销
    if 0 in angles:
        for w, h in search_rects:
            item = _pick_best_from_sum_maps(
                order_idx, w, h, 0,
                box_score_cache[(w, h)], box_cell_cache[(w, h)],
                user_search_mask, head_crop_rect, orientation, rows, cols,
            )
            order_idx += 1
            if item is None:
                continue
            _, in_head, res = item
            if in_head:
                head_results.append(res)
            else:
                tail_results.append(res)

    rotated_tasks = [
        (
            order_idx + i,
            w,
            h,
            angle,
            adjusted_score_map,
            cell_matrix,
            user_search_mask,
            head_crop_rect,
            orientation,
            config.heatmap_penalty_value,
            config.kernel_margin,
            rows,
            cols,
            box_score_cache,
            box_cell_cache,
        )
        for i, (w, h, angle) in enumerate(
            (w, h, angle)
            for (w, h) in search_rects
            for angle in angles
            if angle != 0
        )
    ]

    if rotated_tasks:
        cpu_count = os.cpu_count() or 1
        max_workers = min(len(rotated_tasks), max(1, min(8, cpu_count)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            ordered_results = list(executor.map(_search_one_window_angle, rotated_tasks))

        for item in ordered_results:
            if item is None:
                continue
            _, in_head, res = item
            if in_head:
                head_results.append(res)
            else:
                tail_results.append(res)

    return {"head_results": head_results, "tail_results": tail_results}



def filter_candidates(
    results: Dict[str, List[SelectionResult]], 
    config: BM40Config,
    all_cell_count: int
) -> List[SelectionResult]:
    """
    结果过滤
    根据细胞数量目标筛选候选区域。
    """
    # 选区目标：基础目标 * 冗余系数
    target_num = config.target_cell_num_WBC * config.target_ratio
    
    # 优先看尾部结果（体尾交界），没有则看头部
    candidates = results.get("tail_results", []) or results.get("head_results", [])
    if not candidates:
        print("警告：没有可用的候选选区，返回空列表。")
        return []

    select_ratio = config.select_ratio_init
    selected = []
    
    while not selected:
        # 定义当前的搜索上限
        upper_limit = target_num * (1 + select_ratio)
        
        # 寻找细胞数落在 [target_num, upper_limit] 的区域
        selected = [item for item in candidates if target_num <= item.cell_count <= upper_limit]
        
        if selected:
            break
            
        # 如果当前上限已经超过了全图总细胞数，说明再扩大比例也没有意义了
        if upper_limit > all_cell_count:
            print(f"提示：搜索上限({upper_limit:.0f})已超过全图总数({all_cell_count})，返回细胞数最多的选区。")
            # 此时返回细胞数量最接近全图（即最大）的选区
            return [max(candidates, key=lambda x: x.cell_count)]
            
        # 找不到则扩大步进比例
        select_ratio *= 2
            
    return selected


def prepare_uniformity_map(cell_matrix: np.ndarray, config: BM40Config) -> np.ndarray:
    """
    预计算全局均匀性底图（覆盖率图）。
    """
    # 只要格子里有细胞，就认为该空间被占用
    presence_map = (cell_matrix > 0).astype(np.float32)

    # 计算百倍视野对应的网格尺寸
    kw = max(1, int(config.x100_rect_width // config.cell_size))
    kh = max(1, int(config.x100_rect_height // config.cell_size))

    # 均值滤波：计算窗口内的“占用网格占比”
    return cv2.boxFilter(presence_map, -1, (kw, kh), normalize=True)


def select_best_uniform_region(
    selected_results: List[SelectionResult],
    cell_matrix: np.ndarray,
    config: BM40Config
) -> SelectionResult:
    """
    从候选选区中选出最佳。
    """
    if not selected_results: raise ValueError("根据细胞数量目标筛选候选区域为空。")
    if len(selected_results) == 1: return selected_results[0]

    # 1. 预计算覆盖率图
    u_map = prepare_uniformity_map(cell_matrix, config)
    
    # 2. 取得分最高的前 N 名进行评价
    top_candidates = sorted(selected_results, key=lambda x: x.area_score, reverse=True)[:3]
    
    best_res = None
    max_u_score = -1.0

    for res in top_candidates:
        # 3. 计算该选区（多边形）在覆盖率图上的平均表现
        mask = np.zeros(u_map.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [res.vertices_grid.astype(np.int32)], 1)
        
        # 选区覆盖率 = 选区内各点视野覆盖率的均值
        u_score = np.mean(u_map[mask > 0])
        
        if u_score > max_u_score:
            max_u_score = u_score
            best_res = res
            
    return best_res