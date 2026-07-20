from __future__ import annotations

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2] 
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from project.tiles import Tile

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

from .heatmaps import HeatmapGrid
from .config import BM40Config
from .data_structure import SelectionResult, Rect


def _filter_small_forbidden_components(
    forbidden_mask: np.ndarray,
    min_component_size: int,
) -> np.ndarray:
    """
    仅保留 8-连通域格数 >= min_component_size 的禁区。
    用于过滤过小的 label=5 网格块，避免孤立小区域误伤细胞选区。
    """
    if min_component_size <= 1 or not np.any(forbidden_mask):
        return forbidden_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        forbidden_mask.astype(np.uint8),
        connectivity=8,
    )
    filtered = np.zeros_like(forbidden_mask)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_component_size:
            filtered[labels == label_id] = 1
    return filtered


def build_forbidden_mask(
    grid: HeatmapGrid, 
    config: BM40Config,
    tiles: Optional[List[Tile]] = None, 
    particle_rects: Optional[List[Tuple[float, float, float, float]]] = None
) -> np.ndarray:
    """
    构建网格禁区掩码。支持从 Tile 评分中识别 label=5，或直接传入物理坐标矩形。
    label=5 区域映射到网格后，仅保留 8-连通域格数 >= forbidden_label5_min_component_size
    的块（默认 5，即连续超过 4 格才视为禁区）。
    """
    rows, cols = grid.values.shape
    forbidden_mask = np.zeros((rows, cols), dtype=np.uint8)

    # --- 逻辑 A: 从 Tile 的评分列表中解析 (label=5) ---
    if tiles:
        for t in tiles:
            if t.x is None or t.y is None: continue
            scores_list = t.meta.get("scores", [])
            for row in scores_list:
                if len(row) < 6: continue
                # 提取 Label 
                label = int(row[5])
                
                if label == config.bmp_label:
                    lx, ly = float(row[0]) * config.scale, float(row[1]) * config.scale
                    lw, lh = float(row[2]) * config.scale, float(row[3]) * config.scale
                    
                    gx1, gy1 = grid.global_to_grid(t.x + lx, t.y + ly)
                    gx2, gy2 = grid.global_to_grid(t.x + lx + lw, t.y + ly + lh)
                    
                    # 确保至少占据一个网格
                    if gx1 == gx2: gx2 = gx1 + 1
                    if gy1 == gy2: gy2 = gy1 + 1
                    
                    gx1, gx2 = max(0, min(cols, gx1)), max(0, min(cols, gx2))
                    gy1, gy2 = max(0, min(rows, gy1)), max(0, min(rows, gy2))
                    if gx1 < gx2 and gy1 < gy2:
                        forbidden_mask[gy1:gy2, gx1:gx2] = 1

    # --- 逻辑 B: 直接处理物理定位矩形 [(x, y, w, h), ...] ---
    if particle_rects:
        for (abs_x, abs_y, abs_w, abs_h) in particle_rects:
            gx1, gy1 = grid.global_to_grid(abs_x, abs_y)
            gx2, gy2 = grid.global_to_grid(abs_x + abs_w, abs_y + abs_h)
            
            if gx1 == gx2: gx2 = gx1 + 1
            if gy1 == gy2: gy2 = gy1 + 1
            
            gx1, gx2 = max(0, min(cols, gx1)), max(0, min(cols, gx2))
            gy1, gy2 = max(0, min(rows, gy1)), max(0, min(rows, gy2))
            if gx1 < gx2 and gy1 < gy2:
                forbidden_mask[gy1:gy2, gx1:gx2] = 1

    min_component_size = int(getattr(config, "forbidden_label5_min_component_size", 1) or 1)
    if min_component_size > 1:
        forbidden_mask = _filter_small_forbidden_components(
            forbidden_mask, min_component_size
        )

    return forbidden_mask



def _find_initial_task_bm(
    grid: HeatmapGrid,
    cell_matrix: np.ndarray,
    valid_search_mask: np.ndarray,
    config: BM40Config,
    target_cells: int
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """骨髓初始框：阈值二分 + 最大高分连通域外接矩形。"""
    score_map = grid.finalize(fill_value=config.heatmap_penalty_value)
    valid_scores = score_map[valid_search_mask > 0]
    
    if valid_scores.size == 0:
        return None, None

    # 初始化搜索区间：从实际数据的最小分到最大分
    min_thresh = np.nanmin(valid_scores)
    max_thresh = np.nanmax(valid_scores)
    
    # 设定接受范围：[target * 0.7, target * 1.3] (以 ratio=0.3 为例)
    lower_bound = target_cells * (1 - config.init_task_select_ratio)
    upper_bound = target_cells * (1 + config.init_task_select_ratio)
    
    best_rect = None
    best_th = None
    fallback_rect = None
    fallback_th = None
    max_cells_found = -1

    # 二分查找
    for _ in range(12):
        thresh = (max_thresh + min_thresh) / 2
        _, th = cv2.threshold(score_map, thresh, 255, cv2.THRESH_BINARY)
        th = cv2.bitwise_and(th.astype(np.uint8), (valid_search_mask * 255))
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th)
        if num_labels <= 1:
            # 阈值过高导致没有连通域，必须降低阈值（向左半区搜索）
            max_thresh = thresh
            continue

        # 选面积最大的高分连通域
        max_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x, y, w, h, _ = stats[max_idx]
        
        # 计算该连通域矩形范围内的细胞数
        rect_roi_cells = cell_matrix[y:y+h, x:x+w]
        rect_roi_valid = valid_search_mask[y:y+h, x:x+w]
        count = int(np.sum(rect_roi_cells[rect_roi_valid > 0]))
        
        current_th = th.copy()

        # 兜底方案：保存能找到最多细胞的情况，防止最终二分失败
        if count > max_cells_found:
            max_cells_found = count
            fallback_rect = (x, y, w, h)
            fallback_th = current_th

        # --- 核心更新逻辑 ---
        if count >= lower_bound:
            # 1. 满足下限，记录为最佳候选
            best_rect = (x, y, w, h)
            best_th = current_th
            
            # 2. 如果已经在理想区间 [0.7T, 1.3T] 内，可直接退出
            if count <= upper_bound:
                break
            
            # 3. 细胞数仍超过 1.3T，说明区域太大，需要提高阈值来缩小区域（向右半区搜索）
            min_thresh = thresh
        else:
            # 细胞数不足 0.7T，说明区域太小，需要降低阈值来扩大区域（向左半区搜索）
            max_thresh = thresh

    res_rect = best_rect if best_rect else fallback_rect
    res_th = best_th if best_th is not None else fallback_th
    return res_rect, res_th


def _find_initial_task_pb(
    grid: HeatmapGrid,
    cell_matrix: np.ndarray,
    valid_search_mask: np.ndarray,
    config: BM40Config,
    target_cells: int
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """血片初始框：局部滑窗搜索目标细胞数窗口，避免退化为整片连通域外接矩形。"""
    score_map = grid.finalize(fill_value=config.heatmap_penalty_value).astype(np.float32, copy=False)
    valid_mask = (valid_search_mask > 0).astype(np.float32, copy=False)
    valid_scores = score_map[valid_search_mask > 0]
    if valid_scores.size == 0:
        return None, None

    masked_cells = cell_matrix.astype(np.float32, copy=False) * valid_mask
    masked_scores = score_map * valid_mask
    rows, cols = cell_matrix.shape

    valid_area = float(np.count_nonzero(valid_search_mask))
    total_cells = float(np.sum(masked_cells))
    density = total_cells / max(valid_area, 1.0)
    density = max(density, 1e-6)

    target_area = max(9.0, target_cells / density)
    area_scales = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    lower_bound = target_cells * (1 - config.init_task_select_ratio)
    upper_bound = target_cells * (1 + config.init_task_select_ratio)

    candidate_sizes = set()
    for area_scale in area_scales:
        area = target_area * area_scale
        for rw, rh in config.window_aspect_ratios:
            w = max(3, int(round(np.sqrt(area * rw / rh))))
            h = max(3, int(round(np.sqrt(area * rh / rw))))
            w = min(cols, w)
            h = min(rows, h)
            if w > 0 and h > 0:
                candidate_sizes.add((w, h))

    if not candidate_sizes:
        return None, None

    best_in_range = None
    best_fallback = None
    best_fallback_key = None
    min_valid_ratio = 0.6

    for w, h in sorted(candidate_sizes, key=lambda s: (s[0] * s[1], s[0])):
        area = float(w * h)
        valid_count = cv2.boxFilter(valid_mask, ddepth=-1, ksize=(w, h), normalize=False, borderType=cv2.BORDER_CONSTANT)
        cell_sum = cv2.boxFilter(masked_cells, ddepth=-1, ksize=(w, h), normalize=False, borderType=cv2.BORDER_CONSTANT)
        score_sum = cv2.boxFilter(masked_scores, ddepth=-1, ksize=(w, h), normalize=False, borderType=cv2.BORDER_CONSTANT)

        valid_centers = np.where(valid_count >= max(9.0, area * min_valid_ratio))
        if valid_centers[0].size == 0:
            continue

        counts = cell_sum[valid_centers]
        score_avgs = score_sum[valid_centers] / np.maximum(valid_count[valid_centers], 1.0)

        in_range_mask = (counts >= lower_bound) & (counts <= upper_bound)
        if np.any(in_range_mask):
            in_scores = score_avgs[in_range_mask]
            in_rows = valid_centers[0][in_range_mask]
            in_cols = valid_centers[1][in_range_mask]
            best_idx = int(np.argmax(in_scores))
            candidate = (
                float(in_scores[best_idx]),
                int(round(float(counts[in_range_mask][best_idx]))),
                int(in_cols[best_idx] - w // 2),
                int(in_rows[best_idx] - h // 2),
                w,
                h,
            )
            if best_in_range is None or candidate[0] > best_in_range[0]:
                best_in_range = candidate

        count_delta = np.abs(counts - target_cells)
        best_idx = int(np.lexsort((-score_avgs, count_delta))[0])
        fallback_key = (float(count_delta[best_idx]), float(-score_avgs[best_idx]))
        fallback = (
            float(score_avgs[best_idx]),
            int(round(float(counts[best_idx]))),
            int(valid_centers[1][best_idx] - w // 2),
            int(valid_centers[0][best_idx] - h // 2),
            w,
            h,
        )
        if best_fallback is None or fallback_key < best_fallback_key:
            best_fallback = fallback
            best_fallback_key = fallback_key

    chosen = best_in_range or best_fallback
    if chosen is None:
        return None, None

    _, _, x, y, w, h = chosen
    x = max(0, min(cols - w, x))
    y = max(0, min(rows - h, y))

    score_threshold = float(np.nanmedian(valid_scores))
    pb_priority = ((score_map >= score_threshold) & (valid_search_mask > 0)).astype(np.uint8) * 255
    return (x, y, w, h), pb_priority


def find_initial_task(
    grid: HeatmapGrid,
    cell_matrix: np.ndarray,
    valid_search_mask: np.ndarray,
    config: BM40Config,
    target_cells: int
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """
    BM/PB 统一使用局部滑窗搜索目标细胞数窗口，
    避免 BM 阈值连通域外接矩形把 Initial 撑满整块选区。
    （旧 BM 阈值逻辑保留在 _find_initial_task_bm，暂不调用。）
    """
    return _find_initial_task_pb(grid, cell_matrix, valid_search_mask, config, target_cells)


def generate_initial_and_extra_tasks(
    best_selection: SelectionResult,
    grid: HeatmapGrid,
    cell_matrix: np.ndarray,
    tiles: List[Tile],
    config: BM40Config
) -> List[Tuple[int, int, int, int]]:
    """
    生成一个初始拍摄框，随后通过行/列扩张覆盖整个大选区。
    """
    # 1. 含骨髓小粒的网格
    rows, cols = grid.values.shape
    forbidden_mask = build_forbidden_mask(grid, config, tiles=tiles)
    
    # 选区掩码（大框范围）
    selection_mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.fillPoly(selection_mask, [best_selection.vertices_grid.astype(np.int32)], 1)
    
    # 有效搜索区 = 选区 - 禁区 
    valid_search_mask = cv2.bitwise_and(selection_mask, cv2.bitwise_not(forbidden_mask))
    
    # 2. 寻找初始拍摄框 (小框目标为 target_cell_num_WBC，不乘 ratio)
    initial_rect, final_th = find_initial_task(
        grid, cell_matrix, valid_search_mask, config, config.target_cell_num_WBC
    )

    # 选区内细胞不足 / 滑窗无法落入选区时：用有效选区外接矩形兜底，
    # 后续仍对选区内全部有效细胞生成百倍视野，不中断流程。
    if initial_rect is None:
        fallback_mask = valid_search_mask if np.count_nonzero(valid_search_mask) else selection_mask
        ys, xs = np.where(fallback_mask > 0)
        if xs.size == 0:
            print("[WARNING] 选区内无有效网格，跳过初始/补拍框生成，返回空 task_rects。")
            return []
        x0, y0 = int(xs.min()), int(ys.min())
        w0, h0 = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        initial_rect = (x0, y0, w0, h0)
        score_map = grid.finalize(fill_value=config.heatmap_penalty_value)
        valid_scores = score_map[fallback_mask > 0]
        score_threshold = float(np.nanmedian(valid_scores)) if valid_scores.size else 0.0
        final_th = ((score_map >= score_threshold) & (fallback_mask > 0)).astype(np.uint8) * 255
        print(
            f"[WARNING] 未能按目标细胞数找到初始拍摄框，"
            f"改用有效选区外接矩形 Initial={initial_rect}，将覆盖选区内全部细胞。"
        )

    task_rects = [initial_rect]
    
    # 3. 补拍生长逻辑：直到覆盖完选区内的所有有效网格
    # 已覆盖掩码
    covered_mask = np.zeros((rows, cols), dtype=np.uint8)
    rx, ry, rw, rh = initial_rect
    covered_mask[ry:ry+rh, rx:rx+rw] = 1
    
    # 当前整体包围盒
    cur_x, cur_y, cur_w, cur_h = initial_rect

    max_iter = 1000 
    while max_iter > 0:
        max_iter -= 1
        
        # 检查是否还有 valid_search_mask 里的点没被 covered_mask 覆盖
        remaining = cv2.bitwise_and(valid_search_mask, cv2.bitwise_not(covered_mask))
        if np.count_nonzero(remaining) == 0:
            break # 全部覆盖完成，退出

        # 探测四个方向的 1-像素宽 条带
        directions = [
            (cur_x, cur_y - 1, cur_w, 1),      # 上 (Top)
            (cur_x, cur_y + cur_h, cur_w, 1), # 下 (Bottom)
            (cur_x - 1, cur_y, 1, cur_h),     # 左 (Left)
            (cur_x + cur_w, cur_y, 1, cur_h)  # 右 (Right)
        ]
        
        best_dir_rect = None
        max_score_hits = -1   # 优先级1: 高分网格数
        max_valid_hits = -1   # 优先级2: 有效网格总数
        
        for dx, dy, dw, dh in directions:
            if dx < 0 or dy < 0 or dx + dw > cols or dy + dh > rows:
                continue
            
            # 这里的 strip 指的是这一排还没被拍过的区域
            strip_valid = valid_search_mask[dy:dy+dh, dx:dx+dw]
            strip_covered = covered_mask[dy:dy+dh, dx:dx+dw]
            new_area_mask = (strip_valid > 0) & (strip_covered == 0)
            
            # 计算该条带内的高分网格数 (来自 find_initial_task 确定的 final_th)
            strip_th = final_th[dy:dy+dh, dx:dx+dw]
            score_hits = np.count_nonzero(strip_th[new_area_mask])
            
            # 计算该条带内的有效网格总数
            valid_hits = np.count_nonzero(new_area_mask)
            
            # 策略判定：优先比 score_hits，如果都是 0，再比 valid_hits
            if score_hits > max_score_hits:
                max_score_hits = score_hits
                max_valid_hits = valid_hits
                best_dir_rect = (dx, dy, dw, dh)
            elif score_hits == max_score_hits and score_hits == 0:
                if valid_hits > max_valid_hits:
                    max_valid_hits = valid_hits
                    best_dir_rect = (dx, dy, dw, dh)
            elif score_hits == max_score_hits and score_hits > 0:
                # 如果高分网格数一样，选包含更多有效区域的
                if valid_hits > max_valid_hits:
                    max_valid_hits = valid_hits
                    best_dir_rect = (dx, dy, dw, dh)

        if best_dir_rect is None:
            break
            
        # 记录补拍任务
        task_rects.append(best_dir_rect)
        
        # 更新覆盖状态和整体包围盒
        bx, by, bw, bh = best_dir_rect
        covered_mask[by:by+bh, bx:bx+bw] = 1
        
        x1, y1 = min(cur_x, bx), min(cur_y, by)
        x2, y2 = max(cur_x + cur_w, bx + bw), max(cur_y + cur_h, by + bh)
        cur_x, cur_y, cur_w, cur_h = x1, y1, x2 - x1, y2 - y1

    return task_rects