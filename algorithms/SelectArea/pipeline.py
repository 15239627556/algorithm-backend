# pipeline.py
import numpy as np
from typing import List, Optional

# 导入自定义模块与数据结构
from config import BM40Config
from data_structure import Tile, SelectionResult, TaskOutput
from heatmaps import build_score_heatmap, build_cell_count_grid
from geometry import compute_head_crop, generate_search_window_sizes
from selection import (
    find_candidate_regions, 
    filter_candidates, 
    select_best_uniform_region
)
from task_region_extraction import (
    build_forbidden_mask, 
    generate_initial_and_extra_tasks
)
from task_wbc import (
    collect_valid_cells_vectorized, 
    generate_wbc_view_tasks
)

class WBCSamplingPipeline:
    def __init__(self, config: BM40Config):
        self.cfg = config
        # --- 中间结果存储占位（用于调试/可视化） ---
        self.grid = None           # 存储 HeatmapGrid 对象 
        self.cell_matrix = None    # 存储细胞密度矩阵 
        self.best_res = None       # 存储最终选区结果 
        self.task_rects = None     # 存储网格坐标任务区域 
        self.forbidden_mask = None # 存储禁区掩码 

    def run(self, tiles: List[Tile]) -> List[TaskOutput]:
        """
        执行有核细胞（WBC）采样任务生成全流程
        """
        # 1. 基础数据准备
        self.grid = build_score_heatmap(tiles, config=self.cfg)
        self.cell_matrix = build_cell_count_grid(tiles, self.grid) 

        # 2. 计算全局统计量
        all_cell_count = int(np.sum(self.cell_matrix))
        print(f"项目全部有核细胞数量：{all_cell_count}")

        # 3. 业务参数准备
        target_num = self.cfg.target_cell_num * self.cfg.target_ratio
        head_rect = compute_head_crop(self.grid, self.cfg.heatmap_orientation, self.cfg)
        search_rects = generate_search_window_sizes(self.cfg)

        # 4. 特殊情况处理：全图细胞不足
        if all_cell_count < target_num:
            rows, cols = self.cell_matrix.shape
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
                config=self.cfg
            )

            # 6. 过滤候选区
            selected_list = filter_candidates(
                results=results,
                config=self.cfg,
                all_cell_count=all_cell_count
            )

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

        # 9. 准备全局细胞矩阵
        all_cells_list = []
        for t in tiles:
            for c in t.cells:
                all_cells_list.append([c.cell_xmin + t.x, 
                                       c.cell_ymin + t.y, 
                                       c.cell_xmax + t.x, 
                                       c.cell_ymax + t.y
                                    ])
        
        if not all_cells_list:
            return []
            
        all_cells_array = np.array(all_cells_list)

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