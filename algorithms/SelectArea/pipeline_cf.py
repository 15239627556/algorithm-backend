# pipeline_cf.py
"""脑脊液（CF）FOCUS_POINT 选区流水线。

与 WBC 不同：
1. 由细胞密度矩阵提取主连通团，再按质量心累积覆盖取外接矩形 → task_rect；
2. 在该矩形内按 focus_point（通常为 n^2）平铺视野，尽量全覆盖，
   并在各子区内滑动使细胞内细胞数最多 → task_list。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from project.smear_project import SmearProject
from .config import BM40Config
from .data_structure import TaskOutput
from .heatmaps import build_score_heatmap
from .pipeline_wbc import (
    _build_cell_count_grid_from_bounds,
    _collect_cells_by_type,
)
from .task_cf import (
    assign_cells_to_views,
    find_densest_cell_rect,
    place_focus_point_views,
)

if TYPE_CHECKING:
    from project.roi_store import RoiDataset


def _normalize_smear_type(smear_type: str | None) -> str:
    st = (smear_type or "CF").upper()
    return "CF" if st in {"CF", "CFS", "CSF"} else st


class CFSamplingPipeline:
    """脑脊液密集区 FOCUS_POINT 采样流水线。"""

    def __init__(self, config: BM40Config):
        self.cfg = config
        self.grid = None
        self.cell_matrix = None
        self.task_rect: Optional[List[int]] = None  # [xmin, ymin, xmax, ymax]
        self.all_cells_array = None

    def run(
        self,
        project: SmearProject | None = None,
        *,
        roi: Optional["RoiDataset"] = None,
        focus_point: int | None = None,
    ) -> Tuple[List[int], List[TaskOutput]]:
        """
        Returns
        -------
        task_rect : [xmin, ymin, xmax, ymax]
        task_list : List[TaskOutput]
        """
        if roi is not None:
            tiles = roi.tiles
        else:
            if project is None:
                raise ValueError("run() 需要 project 或 roi 之一")
            layer = project.get_layer(self.cfg.dpi)
            if not layer:
                print("[ERROR][CF] 项目中缺少扫描层数据")
                return [0, 0, 0, 0], []
            tiles = list(layer.tiles.values())

        if not tiles:
            print("[ERROR][CF] 无有效 Tile")
            return [0, 0, 0, 0], []

        n_focus = int(focus_point if focus_point is not None else self.cfg.focus_point)
        if n_focus <= 0:
            print(f"[WARNING][CF] focus_point({n_focus}) <= 0，返回空任务")
            return [0, 0, 0, 0], []

        # 1. 热力图网格（仅作坐标系；细胞矩阵不走 roi 预计算的 WBC 矩阵）
        self.grid = (
            roi.build_heatmap_grid(self.cfg)
            if roi is not None
            else build_score_heatmap(tiles, config=self.cfg)
        )

        cell_type = int(getattr(self.cfg, "CF_cell_type", 100007))
        if roi is not None:
            all_cells = roi.cells_xyxy_by_type(cell_type)
            if all_cells.size == 0:
                all_cells = roi.cells_xyxy_by_type(self.cfg.WBC_cell_type)
        else:
            all_cells = _collect_cells_by_type(tiles, cell_type)
            if all_cells.size == 0:
                all_cells = _collect_cells_by_type(tiles, self.cfg.WBC_cell_type)

        if all_cells.size == 0:
            print("[INFO][CF] 未找到脑脊液细胞")
            return [0, 0, 0, 0], []

        self.all_cells_array = all_cells
        # roi.build_cell_matrix 会优先返回预计算 WBC 矩阵（脑脊液常为空），故自行构建
        self.cell_matrix = _build_cell_count_grid_from_bounds(all_cells, self.grid)
        print(
            f"[INFO][CF] 细胞数={len(all_cells)}，密度格非零="
            f"{int(np.count_nonzero(self.cell_matrix))}，focus_point={n_focus}"
        )

        # 2. 最致密区域 → task_rect
        self.task_rect = list(
            find_densest_cell_rect(self.cell_matrix, self.grid, self.cfg, all_cells)
        )
        print(f"[INFO][CF] task_rect={self.task_rect}")

        # 3. 密集区内平铺 focus_point 个视野（积分图加速）
        views = place_focus_point_views(
            task_rect=tuple(self.task_rect),  # type: ignore[arg-type]
            all_cells_array=all_cells,
            config=self.cfg,
            focus_point=n_focus,
        )
        cell_groups = assign_cells_to_views(all_cells, views)

        smear_type = _normalize_smear_type(self.cfg.Smear_type)
        view_type = (self.cfg.View_type or "FOCUS_POINT").upper()
        if view_type in {"CF", "CFS", "CSF", "WBC"}:
            view_type = "FOCUS_POINT"

        tasks: List[TaskOutput] = []
        for i, ((x0, y0, x1, y1), cells) in enumerate(zip(views, cell_groups), start=1):
            tasks.append(
                TaskOutput(
                    task_index=i,
                    view_type=view_type,
                    smear_type=smear_type,
                    view_xmin=int(x0),
                    view_ymin=int(y0),
                    view_xmax=int(x1),
                    view_ymax=int(y1),
                    region_name=self.cfg.Initial_name,
                    cell_list=cells,
                )
            )

        print(
            f"[INFO][CF] 生成视野={len(tasks)}，"
            f"细胞内细胞合计={sum(len(t.cell_list) for t in tasks)}"
        )
        return self.task_rect, tasks
