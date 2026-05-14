# pipeline.py
import os
import numpy as np
from typing import List, Optional

# 导入自定义模块与数据结构
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2] 
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from project.smear_project import SmearProject
from .config import BM40Config
from .data_structure import SelectionResult, TaskOutput
from .heatmaps import build_score_heatmap
from .geometry import compute_head_crop, generate_search_window_sizes
from .selection import (
    find_candidate_regions, 
    filter_candidates, 
    select_best_uniform_region
)
from .task_region_extraction import (
    build_forbidden_mask, 
    generate_initial_and_extra_tasks
)
from .task_wbc import (
    collect_valid_cells_vectorized, 
    generate_wbc_view_tasks
)

def _collect_cells_by_type(tiles, target_type: int) -> np.ndarray:
    """收集指定 cell_type 的全局坐标，预分配 ndarray 避免构造百万级 Python list。"""
    capacity = sum(len(t.cells) for t in tiles if t.x is not None and t.y is not None)
    if capacity == 0:
        return np.empty((0, 4), dtype=np.float32)

    all_cells_array = np.empty((capacity, 4), dtype=np.float32)
    count = 0
    for t in tiles:
        tx, ty = t.x, t.y
        if tx is None or ty is None:
            continue
        for c in t.cells:
            if c.cell_type != target_type:
                continue
            all_cells_array[count, 0] = c.cell_xmin + tx
            all_cells_array[count, 1] = c.cell_ymin + ty
            all_cells_array[count, 2] = c.cell_xmax + tx
            all_cells_array[count, 3] = c.cell_ymax + ty
            count += 1

    if count == 0:
        return np.empty((0, 4), dtype=np.float32)
    return all_cells_array[:count]


def _build_cell_count_grid_from_bounds(all_cells_array: np.ndarray, grid) -> np.ndarray:
    """基于已收集的细胞坐标向量化生成细胞密度矩阵，避免再次扫描 tiles。"""
    rows, cols = grid.values.shape
    cell_count_matrix = np.zeros((rows, cols), dtype=np.float32)
    if all_cells_array.size == 0:
        return cell_count_matrix

    centers = 0.5 * (all_cells_array[:, 0:2] + all_cells_array[:, 2:4])
    g_cols = ((centers[:, 0] - grid.origin_x) // grid.cell_size).astype(np.int32)
    g_rows = ((centers[:, 1] - grid.origin_y) // grid.cell_size).astype(np.int32)
    in_bounds = (g_rows >= 0) & (g_rows < rows) & (g_cols >= 0) & (g_cols < cols)
    flat_idx = g_rows[in_bounds] * cols + g_cols[in_bounds]
    return np.bincount(flat_idx, minlength=rows * cols).reshape(rows, cols).astype(np.float32, copy=False)

class WBCSamplingPipeline:
    def __init__(self, config: BM40Config):
        self.cfg = config
        # --- 中间结果存储占位（用于调试/可视化） ---
        self.grid = None           # 存储 HeatmapGrid 对象 
        self.cell_matrix = None    # 存储细胞密度矩阵 
        self.best_res = None       # 存储最终选区结果 
        self.task_rects = None     # 存储网格坐标任务区域 
        self.forbidden_mask = None # 存储禁区掩码 
        self.user_search_mask = None # 存储用户选区掩码


    def run(self, project: SmearProject) -> List[TaskOutput]:
        """
        执行有核细胞采样任务。
        输入改为 SmearProject 对象。
        """
        # 1. 从项目中提取 40x 扫描层
        # layer_40x_id = 0
        # layer_40x = project.layers[layer_40x_id]
        dpi = self.cfg.dpi
        layer_40x = project.get_layer(dpi)
        if not layer_40x:
            print("[ERROR] 项目中缺少 40x 扫描层数据")
            return []

        # 获取该层所有的 tiles
        # layer.tiles 是一个 Dict[str, Tile] 或提供迭代器
        tiles = list(layer_40x.tiles.values()) 
        if not tiles:
            print("[ERROR] 40x 层中没有找到有效的 Tile 数据")
            return []

        if self.cfg.target_cell_num_WBC <= 0:
            print(f"target_cell_num_WBC({self.cfg.target_cell_num_WBC}) <= 0，将返回空的 wbc_tasks。")
            return []

        # 1. 基础数据准备
        self.grid = build_score_heatmap(tiles, config=self.cfg)
        all_cells_array = _collect_cells_by_type(tiles, self.cfg.WBC_cell_type)
        if all_cells_array.size == 0:
            print("[INFO] 未在 40x tiles 中找到任何有核细胞。")
            return []
        self.cell_matrix = _build_cell_count_grid_from_bounds(all_cells_array, self.grid)

        rows, cols = self.cell_matrix.shape

        # --- 构建用户选区约束掩码 ---
        self.user_search_mask = None
        if self.cfg.user_choice_area:
            self.user_search_mask = np.zeros((rows, cols), dtype=np.uint8)
            # 物理坐标转网格坐标
            c0, r0 = self.grid.global_to_grid(self.cfg.user_choice_area['x_min'], self.cfg.user_choice_area['y_min'])
            c1, r1 = self.grid.global_to_grid(self.cfg.user_choice_area['x_max'], self.cfg.user_choice_area['y_max'])
            # 边界安全裁剪
            c0, c1 = max(0, min(cols, c0)), max(0, min(cols, c1))
            r0, r1 = max(0, min(rows, r0)), max(0, min(rows, r1))
            self.user_search_mask[r0:r1, c0:c1] = 1
            print(f"[INFO] 已应用用户约束选区: Grid({c0},{r0}) to Grid({c1},{r1})")

        # 2. 计算全局统计量
        if self.user_search_mask is not None:
        # 核心修正：只计算用户框选区域内的细胞总数
            all_cell_count = int(np.sum(self.cell_matrix[self.user_search_mask > 0]))
            print(f"用户选区内有核细胞数量：{all_cell_count}")
        else:
            all_cell_count = int(np.sum(self.cell_matrix))
            print(f"项目全部有核细胞数量：{all_cell_count}")


        # 3. 业务参数准备
        target_num = self.cfg.target_cell_num_WBC * self.cfg.target_ratio
        head_rect = compute_head_crop(self.grid, self.cfg.heatmap_orientation, self.cfg)
        search_rects = generate_search_window_sizes(self.cfg)

        # 4. 特殊情况处理：全图细胞不足
        if all_cell_count < target_num:
            rows, cols = self.cell_matrix.shape
            
            # --- 优先遵循用户选区 ---
            if self.cfg.user_choice_area:
                # 将用户物理坐标转换为网格坐标
                c0, r0 = self.grid.global_to_grid(self.cfg.user_choice_area['x_min'], self.cfg.user_choice_area['y_min'])
                c1, r1 = self.grid.global_to_grid(self.cfg.user_choice_area['x_max'], self.cfg.user_choice_area['y_max'])
                
                # 边界安全裁剪
                c0, c1 = max(0, min(cols, c0)), max(0, min(cols, c1))
                r0, r1 = max(0, min(rows, r0)), max(0, min(rows, r1))
                
                # 计算该特定区域内的分值和细胞数
                sub_score_map = self.grid.finalize()[r0:r1, c0:c1]
                sub_cell_matrix = self.cell_matrix[r0:r1, c0:c1]
                
                self.best_res = SelectionResult(
                    area_score=float(np.nanmean(sub_score_map)) if sub_score_map.size > 0 else 0.0,
                    cell_count=int(np.sum(sub_cell_matrix)),
                    angle=0,
                    center_grid=((c0 + c1) // 2, (r0 + r1) // 2),
                    rect_size_grid=(c1 - c0, r1 - r0),
                    vertices_grid=np.array([[c0, r0], [c1, r0], [c1, r1], [c0, r1]])
                )
                print(f"[INFO] 细胞不足，但已将选区锁定在用户指定区域。")
            else:
                # 无用户选区时，维持原有的全图降级逻辑
                self.best_res = SelectionResult(
                    area_score=float(np.nanmean(self.grid.finalize())),
                    cell_count=all_cell_count,
                    angle=0,
                    center_grid=(cols // 2, rows // 2),
                    rect_size_grid=(cols, rows),
                    vertices_grid=np.array([[0, 0], [cols, 0], [cols, rows], [0, rows]])
                )
        else:
            # 5. 正常选区流程：寻找候选区
            results = find_candidate_regions(
                grid=self.grid,
                cell_matrix=self.cell_matrix,
                search_rects=search_rects,
                head_crop_rect=head_rect,
                config=self.cfg,
                user_search_mask=self.user_search_mask
            )
            print(f"[INFO][WBC] 过滤前的 head 候选区域数量: {len(results['head_results'])}")
            # print(f"[INFO][WBC] 过滤前的 head 候选区域: {results['head_results']}")

            print(f"[INFO][WBC] 过滤前的 tail 候选区域数量: {len(results['tail_results'])}")
            # print(f"[INFO][WBC] 过滤前的 tail 候选区域: {results['tail_results']}")

            # 6. 过滤候选区
            selected_list = filter_candidates(
                results=results,
                config=self.cfg,
                all_cell_count=all_cell_count
            )
            print(f"[INFO][RBC] 过滤后的候选区域数量: {len(selected_list)}")
            if os.getenv("SELECT_AREA_DEBUG_CANDIDATES") == "1":
                print(f"[INFO][RBC] 候选区域: {selected_list}")

            # 7. 均匀性评估：选出最佳选区
            self.best_res = select_best_uniform_region(
                selected_results=selected_list,
                cell_matrix=self.cell_matrix,
                config=self.cfg
            )

        # 8. 生成拍摄区域（初始拍摄框 + 补拍区域）
        self.task_rects = generate_initial_and_extra_tasks(
            best_selection=self.best_res,
            grid=self.grid,
            cell_matrix=self.cell_matrix,
            tiles=tiles,
            config=self.cfg
        )
        # 10. 构建禁区掩码并提取有效细胞
        self.forbidden_mask = build_forbidden_mask(self.grid, self.cfg, tiles=tiles) 
        valid_cells = collect_valid_cells_vectorized(
            all_cells_array=all_cells_array, 
            best_selection=self.best_res, 
            grid=self.grid, 
            forbidden_mask=self.forbidden_mask
        )

        # 11. 生成最终采样任务
        final_tasks = generate_wbc_view_tasks(
            cell_bounds=valid_cells,
            task_rects=self.task_rects,
            grid=self.grid,
            config=self.cfg
        )

        return final_tasks