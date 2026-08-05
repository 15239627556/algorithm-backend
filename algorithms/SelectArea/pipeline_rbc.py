import os
import numpy as np
from typing import List, Optional, TYPE_CHECKING

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
    select_best_uniform_region,
)
from .task_region_extraction import (
    build_forbidden_mask,
    generate_initial_and_extra_tasks,
)
from .task_wbc import (
    collect_valid_cells_vectorized,
    generate_wbc_view_tasks,
)

if TYPE_CHECKING:
    from project.roi_store import RoiDataset


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


class RBCSamplingPipeline:
    def __init__(self, config: BM40Config):
        self.cfg = config
        self.grid = None
        self.cell_matrix = None
        self.best_res = None
        self.task_rects = None
        self.forbidden_mask = None
        self.user_search_mask = None

    def run(
        self,
        project: SmearProject | None = None,
        *,
        roi: Optional["RoiDataset"] = None,
    ) -> List[TaskOutput]:
        if roi is not None:
            tiles = roi.tiles
        else:
            if project is None:
                raise ValueError("run() 需要 project 或 roi 之一")
            layer_40x = project.get_layer(self.cfg.dpi)
            if not layer_40x:
                print("[ERROR][RBC] 项目中缺少 40x 扫描层数据")
                return []
            tiles = list(layer_40x.tiles.values())

        if not tiles:
            print("[ERROR][RBC] 40x 层中没有找到有效的 Tile 数据")
            return []

        if self.cfg.target_cell_num_WBC <= 0:
            print(f"[WARNING][RBC] target_cell_num_WBC({self.cfg.target_cell_num_WBC}) <= 0，将返回空任务。")
            return []

        self.grid = (
            roi.build_heatmap_grid(self.cfg)
            if roi is not None
            else build_score_heatmap(tiles, config=self.cfg)
        )
        all_cells_array = (
            roi.cells_xyxy_by_type(self.cfg.WBC_cell_type)
            if roi is not None
            else _collect_cells_by_type(tiles, self.cfg.WBC_cell_type)
        )
        if all_cells_array.size == 0:
            print("[INFO][RBC] 未在 40x tiles 中找到任何有核细胞。")
            return []
        self.cell_matrix = (
            roi.build_cell_matrix(self.cfg, all_cells_array=all_cells_array)
            if roi is not None
            else _build_cell_count_grid_from_bounds(all_cells_array, self.grid)
        )
        rows, cols = self.cell_matrix.shape

        self.user_search_mask = None
        if self.cfg.user_choice_area:
            self.user_search_mask = np.zeros((rows, cols), dtype=np.uint8)
            c0, r0 = self.grid.global_to_grid(self.cfg.user_choice_area["x_min"], self.cfg.user_choice_area["y_min"])
            c1, r1 = self.grid.global_to_grid(self.cfg.user_choice_area["x_max"], self.cfg.user_choice_area["y_max"])
            c0, c1 = max(0, min(cols, c0)), max(0, min(cols, c1))
            r0, r1 = max(0, min(rows, r0)), max(0, min(rows, r1))
            self.user_search_mask[r0:r1, c0:c1] = 1
            print(f"[INFO][RBC] 已应用用户约束选区: Grid({c0},{r0}) to Grid({c1},{r1})")

        if self.user_search_mask is not None:
            all_cell_count = int(np.sum(self.cell_matrix[self.user_search_mask > 0]))
            print(f"[INFO][RBC] 用户选区内细胞数量：{all_cell_count}")
        else:
            all_cell_count = int(np.sum(self.cell_matrix))
            print(f"[INFO][RBC] 全部细胞数量：{all_cell_count}")

        # 血片选区仍按有核细胞数量进行选择
        target_num = self.cfg.target_cell_num_WBC * self.cfg.target_ratio
        head_rect = compute_head_crop(self.grid, self.cfg.heatmap_orientation, self.cfg)
        search_rects = generate_search_window_sizes(self.cfg)

        if all_cell_count < target_num:
            if self.cfg.user_choice_area:
                c0, r0 = self.grid.global_to_grid(self.cfg.user_choice_area["x_min"], self.cfg.user_choice_area["y_min"])
                c1, r1 = self.grid.global_to_grid(self.cfg.user_choice_area["x_max"], self.cfg.user_choice_area["y_max"])
                c0, c1 = max(0, min(cols, c0)), max(0, min(cols, c1))
                r0, r1 = max(0, min(rows, r0)), max(0, min(rows, r1))

                sub_score_map = self.grid.finalize()[r0:r1, c0:c1]
                sub_cell_matrix = self.cell_matrix[r0:r1, c0:c1]
                self.best_res = SelectionResult(
                    area_score=float(np.nanmean(sub_score_map)) if sub_score_map.size > 0 else 0.0,
                    cell_count=int(np.sum(sub_cell_matrix)),
                    angle=0,
                    center_grid=((c0 + c1) // 2, (r0 + r1) // 2),
                    rect_size_grid=(c1 - c0, r1 - r0),
                    vertices_grid=np.array([[c0, r0], [c1, r0], [c1, r1], [c0, r1]]),
                )
            else:
                self.best_res = SelectionResult(
                    area_score=float(np.nanmean(self.grid.finalize())),
                    cell_count=all_cell_count,
                    angle=0,
                    center_grid=(cols // 2, rows // 2),
                    rect_size_grid=(cols, rows),
                    vertices_grid=np.array([[0, 0], [cols, 0], [cols, rows], [0, rows]]),
                )
        else:
            results = find_candidate_regions(
                grid=self.grid,
                cell_matrix=self.cell_matrix,
                search_rects=search_rects,
                head_crop_rect=head_rect,
                config=self.cfg,
                user_search_mask=self.user_search_mask,
            )
            print(f"[INFO][RBC] 过滤前的 head 候选区域数量: {len(results['head_results'])}")
            print(f"[INFO][RBC] 过滤前的 tail 候选区域数量: {len(results['tail_results'])}")

            selected_list = filter_candidates(
                results=results,
                config=self.cfg,
                all_cell_count=all_cell_count,
            )
            print(f"[INFO][RBC] 过滤后的候选区域数量: {len(selected_list)}")
            if os.getenv("SELECT_AREA_DEBUG_CANDIDATES") == "1":
                print(f"[INFO][RBC] 候选区域: {selected_list}")

            self.best_res = select_best_uniform_region(
                selected_results=selected_list,
                cell_matrix=self.cell_matrix,
                config=self.cfg,
            )

        self.task_rects = generate_initial_and_extra_tasks(
            best_selection=self.best_res,
            grid=self.grid,
            cell_matrix=self.cell_matrix,
            tiles=tiles,
            config=self.cfg,
        )

        self.forbidden_mask = build_forbidden_mask(self.grid, self.cfg, tiles=tiles)
        valid_cells = collect_valid_cells_vectorized(
            all_cells_array=all_cells_array,
            best_selection=self.best_res,
            grid=self.grid,
            forbidden_mask=self.forbidden_mask,
        )

        final_tasks = generate_wbc_view_tasks(
            cell_bounds=valid_cells,
            task_rects=self.task_rects,
            grid=self.grid,
            config=self.cfg,
        )
        return final_tasks

