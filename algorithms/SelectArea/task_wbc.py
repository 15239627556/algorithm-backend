from __future__ import annotations
import numpy as np
import cv2
from typing import List, Dict, Tuple, Any, Optional
from .heatmaps import HeatmapGrid
from .config import BM40Config
from .data_structure import SelectionResult, TaskOutput, CellOutput
from .setcover import solve, SetCoverSolverParameter




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
    params: Optional[SetCoverSolverParameter] = None
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
        rect_height=config.x100_rect_height
    )
    
    # 2. 求解百倍视野分布
    rects_x100 = solve(centers, bounding_rect, params)

    # 3. 结果构建与区域分配
    used = np.zeros(len(cell_bounds), dtype=bool)
    # 内部仍先分组，以便维持各区域内的扫描路径排序
    temp_grouped: List[List[TaskOutput]] = [[] for _ in range(len(task_rects))]

    for rx, ry, rw, rh in rects_x100:
        # a. 视野内细胞匹配
        in_x = (centers[:, 0] >= rx) & (centers[:, 0] < rx + rw)
        in_y = (centers[:, 1] >= ry) & (centers[:, 1] < ry + rh)
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
        view_cx, view_cy = rx + rw/2, ry + rh/2
        vgx, vgy = grid.global_to_grid(view_cx, view_cy)
        
        assigned_idx = 0 
        for i, (tx, ty, tw, th) in enumerate(task_rects):
            if tx <= vgx < tx + tw and ty <= vgy < ty + th:
                assigned_idx = i
                break
        
        # 映射区域名称：0 -> 初始拍摄框, 1+ -> 补拍1, 补拍2...
        region_label = config.Initial_name if assigned_idx == 0 else config.Extra_name + "_" + str(assigned_idx)

        # c. 实例化 TaskOutput
        task_obj = TaskOutput(
            task_index=0,  # 占位
            view_type=config.View_type,
            smear_type=config.Smear_type,
            view_xmin=int(round(rx)),
            view_ymin=int(round(ry)),
            view_xmax=int(round(rx + rw)),
            view_ymax=int(round(ry + rh)),
            region_name=region_label,
            cell_list=current_cell_outputs
        )
        temp_grouped[assigned_idx].append(task_obj)

    # 4. 排序、平铺并分配全局索引
    final_flattened_list: List[TaskOutput] = []
    global_counter = 1

    for group in temp_grouped:
        # 区域内按坐标排序以优化移动路径
        group.sort(key=lambda t: (t.view_xmin, t.view_ymin))
        for task in group:
            task.task_index = global_counter
            final_flattened_list.append(task)
            global_counter += 1

    return final_flattened_list