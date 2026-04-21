import numpy as np
from typing import List

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from project.smear_project import SmearProject
from .config import BM40Config
from .data_structure import SelectionResult, TaskOutput
from .heatmaps import build_score_heatmap, build_cell_count_grid
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


def _collect_cells_by_type(tiles, target_type: int):
    """收集指定 cell_type 的全局坐标，避免重复遍历 tiles。"""
    all_cells_list = []
    append = all_cells_list.append

    for t in tiles:
        tx, ty = t.x, t.y
        if tx is None or ty is None:
            continue
        for c in t.cells:
            if getattr(c, "cell_type", None) != target_type:
                continue
            append(
                [c.cell_xmin + tx, c.cell_ymin + ty, c.cell_xmax + tx, c.cell_ymax + ty]
            )
    return all_cells_list


class RBCSamplingPipeline:
    def __init__(self, config: BM40Config):
        self.cfg = config
        self.grid = None
        self.cell_matrix = None
        self.best_res = None
        self.task_rects = None
        self.forbidden_mask = None
        self.user_search_mask = None

    def run(self, project: SmearProject) -> List[TaskOutput]:
        dpi = self.cfg.dpi
        layer_40x = project.get_layer(dpi)
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

        all_cells_list = _collect_cells_by_type(tiles, self.cfg.WBC_cell_type)
        if not all_cells_list:
            print("[INFO][RBC] 未在 40x tiles 中找到任何有核细胞。")
            return []
        all_cells_array = np.array(all_cells_list, dtype=np.float32)

        self.grid = build_score_heatmap(tiles, config=self.cfg)
        self.cell_matrix = build_cell_count_grid(tiles, self.grid, config=self.cfg)
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

            selected_list = filter_candidates(
                results=results,
                config=self.cfg,
                all_cell_count=all_cell_count,
            )

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

