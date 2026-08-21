from __future__ import annotations
import numpy as np
import cv2
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
from .heatmaps import HeatmapGrid
from .config import BM40Config
from .data_structure import SelectionResult, TaskOutput, CellOutput
from .setcover import solve, SetCoverSolverParameter, expand_rect_to_nominal_bounds


def _task_view_center(task: TaskOutput) -> tuple[float, float]:
    return (
        (task.view_xmin + task.view_xmax) * 0.5,
        (task.view_ymin + task.view_ymax) * 0.5,
    )


def _open_path_length(centers: np.ndarray, order: List[int]) -> float:
    if len(order) <= 1:
        return 0.0
    idx = np.asarray(order, dtype=np.int32)
    seg = centers[idx[1:]] - centers[idx[:-1]]
    return float(np.sqrt((seg * seg).sum(axis=1)).sum())


def _two_opt_open_path(order: List[int], centers: np.ndarray) -> List[int]:
    """开放路径 2-opt：不闭合回路，仅反转中间段以缩短总路程。"""
    n = len(order)
    if n <= 3:
        return order

    best = order[:]
    best_len = _open_path_length(centers, best)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for k in range(i + 2, n):
                candidate = best[: i + 1] + best[i + 1 : k + 1][::-1] + best[k + 1 :]
                cand_len = _open_path_length(centers, candidate)
                if cand_len + 1e-6 < best_len:
                    best = candidate
                    best_len = cand_len
                    improved = True
                    break
            if improved:
                break
    return best


def _order_tasks_tsp(tasks: List[TaskOutput]) -> List[TaskOutput]:
    """组内开放 TSP：从 xmin/ymin 最小的视野出发，最近邻构造路径，再 2-opt 优化。"""
    if len(tasks) <= 2:
        return sorted(tasks, key=lambda t: (t.view_xmin, t.view_ymin))

    centers = np.asarray([_task_view_center(t) for t in tasks], dtype=np.float64)
    n = len(tasks)
    start = int(np.lexsort((centers[:, 1], centers[:, 0]))[0])

    order: List[int] = [start]
    unvisited = set(range(n))
    unvisited.remove(start)
    current = start

    while unvisited:
        cx, cy = centers[current]
        nxt = min(
            unvisited,
            key=lambda j: (centers[j, 0] - cx) ** 2 + (centers[j, 1] - cy) ** 2,
        )
        order.append(nxt)
        unvisited.remove(nxt)
        current = nxt

    order = _two_opt_open_path(order, centers)
    return [tasks[i] for i in order]


def _order_tasks_scanline(tasks: List[TaskOutput]) -> List[TaskOutput]:
    return sorted(tasks, key=lambda t: (t.view_xmin, t.view_ymin))


def _task_center_dist_sq(a: TaskOutput, b: TaskOutput) -> float:
    ax, ay = _task_view_center(a)
    bx, by = _task_view_center(b)
    dx, dy = ax - bx, ay - by
    return dx * dx + dy * dy


def _order_tasks_band_snake(tasks: List[TaskOutput], band_height: int) -> List[TaskOutput]:
    """
    按 Y 分带、带内按 (xmin, ymin) 排序；每条带选正/反序，使与上一带末点衔接更短。
    单 band 时退化为 scanline，BM 等紧凑样本无额外开销。
    """
    if len(tasks) <= 1:
        return tasks[:]

    band_height = max(int(band_height), 1)
    buckets: Dict[int, List[TaskOutput]] = defaultdict(list)
    for task in tasks:
        band_key = int(task.view_ymin) // band_height
        buckets[band_key].append(task)

    if len(buckets) <= 1:
        return _order_tasks_scanline(tasks)

    ordered: List[TaskOutput] = []
    prev_end: Optional[TaskOutput] = None
    for band_key in sorted(buckets.keys()):
        row = sorted(buckets[band_key], key=lambda t: (t.view_xmin, t.view_ymin))
        if prev_end is None:
            chosen = row
        else:
            # 比较从上一带末点进入本带左端 vs 右端的距离，选更短方向
            chosen = (
                row[::-1]
                if _task_center_dist_sq(prev_end, row[-1]) < _task_center_dist_sq(prev_end, row[0])
                else row
            )
        ordered.extend(chosen)
        prev_end = chosen[-1]
    return ordered


def order_tasks_in_group(tasks: List[TaskOutput], config: BM40Config) -> List[TaskOutput]:
    if not tasks:
        return tasks
    mode = config.view_path_order_mode.lower()
    if mode == "scanline":
        return _order_tasks_scanline(tasks)
    if mode == "band_snake":
        return _order_tasks_band_snake(tasks, config.view_path_band_height)
    if mode == "tsp":
        return _order_tasks_tsp(tasks)
    return _order_tasks_band_snake(tasks, config.view_path_band_height)



def collect_valid_cells_vectorized(
    all_cells_array: np.ndarray,  # 预先提取好的 (N, 4) 全局细胞矩阵
    best_selection: SelectionResult,
    grid: HeatmapGrid,
    forbidden_mask: np.ndarray 
) -> np.ndarray:
    """
    提取有效细胞点 (全局坐标)
    使用向量化方式快速过滤有效细胞。
    all_cells_array: 形状为 (N, 4), 列为 [xmin, ymin, xmax, ymax]
    """
    if all_cells_array.size == 0:
        return np.array([])

    # 1. 计算所有细胞的中心点 (向量化操作)
    # centers shape: (N, 2)
    centers = 0.5 * (all_cells_array[:, 0:2] + all_cells_array[:, 2:4])
    
    # 2. 将物理中心点一次性映射到网格索引 (向量化)
    # 利用 HeatmapGrid 的属性计算：gx = (x - origin_x) // cell_size
    gxs = ((centers[:, 0] - grid.origin_x) // grid.cell_size).astype(np.int32)
    gys = ((centers[:, 1] - grid.origin_y) // grid.cell_size).astype(np.int32)

    # 3. 边界安全过滤
    rows, cols = forbidden_mask.shape
    in_bounds = (gxs >= 0) & (gxs < cols) & (gys >= 0) & (gys < rows)
    
    # 只处理在网格内的点
    valid_idx = np.where(in_bounds)[0]
    gxs_valid = gxs[valid_idx]
    gys_valid = gys[valid_idx]

    # 4. 构建选区掩码 (用于快速判定多边形内部)
    # 比对每个点调用 pointPolygonTest 快得多
    selection_mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.fillPoly(selection_mask, [best_selection.vertices_grid.astype(np.int32)], 1)

    # 5. 核心判定：同时满足 (不在禁区) 且 (在选区内)
    # 利用 NumPy 高级索引，一次性取出所有点对应的掩码值
    is_forbidden = forbidden_mask[gys_valid, gxs_valid] > 0
    is_in_selection = selection_mask[gys_valid, gxs_valid] > 0
    
    # 最终保留的索引
    keep_sub_idx = np.where((~is_forbidden) & is_in_selection)[0]
    final_keep_idx = valid_idx[keep_sub_idx]

    return all_cells_array[final_keep_idx]




def generate_wbc_view_tasks(
    cell_bounds: np.ndarray,      # 过滤后的有效细胞矩阵 (N, 4)
    task_rects: List[Tuple[int, int, int, int]], # 拍摄区域网格坐标任务矩形列表
    grid: HeatmapGrid,
    config: BM40Config,
    params: Optional[SetCoverSolverParameter] = None,
) -> List[TaskOutput]:
    """Smear_type
    生成百倍视野任务，标记所属区域名称，并返回平铺的任务列表。
    """
    if cell_bounds.size == 0:
        print("警告：无有效细胞，无法生成 WBC 任务。")
        return []

    # 1. 准备 Set-Cover 输入
    centers = 0.5 * (cell_bounds[:, 0:2] + cell_bounds[:, 2:4])
    pad = config.setcover_pad
    x_min_all, y_min_all = cell_bounds[:, 0:2].min(axis=0) - pad
    x_max_all, y_max_all = cell_bounds[:, 2:4].max(axis=0) + pad
    bounding_rect = np.array([x_min_all, y_min_all, x_max_all - x_min_all + 1, y_max_all - y_min_all + 1], dtype=np.int32)
    
    params = params or SetCoverSolverParameter(
        rect_width=config.x100_rect_width, 
        rect_height=config.x100_rect_height,
        rect_size_scale=config.x100_rect_size_scale,
    )
    
    # 2. 求解百倍视野分布
    rects_x100 = solve(centers, bounding_rect, params)

    # 3. 结果构建与区域分配
    used = np.zeros(len(cell_bounds), dtype=bool)
    # 内部仍先分组，以便维持各区域内的扫描路径排序
    temp_grouped: List[List[TaskOutput]] = [[] for _ in range(len(task_rects))]

    for rx, ry, rw, rh in rects_x100:
        view_xmin, view_ymin, view_xmax, view_ymax = expand_rect_to_nominal_bounds(
            rx, ry, rw, rh,
            config.x100_rect_width,
            config.x100_rect_height,
        )
        # a. 视野内细胞匹配（落盘标称尺寸，与实际拍摄视野一致）
        in_x = (centers[:, 0] >= view_xmin) & (centers[:, 0] < view_xmax)
        in_y = (centers[:, 1] >= view_ymin) & (centers[:, 1] < view_ymax)
        mask = in_x & in_y & (~used)
        
        current_cell_outputs = []
        if np.any(mask):
            matched_idx = np.where(mask)[0]
            for idx in matched_idx:
                current_cell_outputs.append(CellOutput(
                    cell_xmin=int(round(cell_bounds[idx, 0])),
                    cell_ymin=int(round(cell_bounds[idx, 1])),
                    cell_xmax=int(round(cell_bounds[idx, 2])),
                    cell_ymax=int(round(cell_bounds[idx, 3]))
                ))
            used[mask] = True

        # b. 区域归属判定与名称映射
        view_cx = (view_xmin + view_xmax) * 0.5
        view_cy = (view_ymin + view_ymax) * 0.5
        vgx, vgy = grid.global_to_grid(view_cx, view_cy)
        
        assigned_idx = 0 
        for i, (tx, ty, tw, th) in enumerate(task_rects):
            if tx <= vgx < tx + tw and ty <= vgy < ty + th:
                assigned_idx = i
                break
        
        # c. 实例化 TaskOutput（标称 x100 尺寸）
        task_obj = TaskOutput(
            task_index=0,  # 占位
            view_type=config.View_type,
            smear_type=config.Smear_type,
            view_xmin=view_xmin,
            view_ymin=view_ymin,
            view_xmax=view_xmax,
            view_ymax=view_ymax,
            region_name="",
            cell_list=current_cell_outputs
        )
        temp_grouped[assigned_idx].append(task_obj)

    # 4. 排序、平铺并分配全局索引
    final_flattened_list: List[TaskOutput] = []
    global_counter = 1
    extra_idx_remap = {}
    next_extra_idx = 1

    for original_idx in range(1, len(temp_grouped)):
        if temp_grouped[original_idx]:
            extra_idx_remap[original_idx] = next_extra_idx
            next_extra_idx += 1

    for group_idx, group in enumerate(temp_grouped):
        ordered_group = order_tasks_in_group(group, config)
        for task in ordered_group:
            if group_idx == 0:
                task.region_name = config.Initial_name
            else:
                task.region_name = f"{config.Extra_name}_{extra_idx_remap[group_idx]}"
            task.task_index = global_counter
            final_flattened_list.append(task)
            global_counter += 1

    return final_flattened_list