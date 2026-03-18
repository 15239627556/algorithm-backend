# pipeline_meg.py
import numpy as np
from typing import List
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from project.smear_project import SmearProject
from .config import BM40Config
from .data_structure import TaskOutput
from .heatmaps import build_score_heatmap
from .task_region_extraction import build_forbidden_mask
from .task_meg import generate_meg_view_tasks


class MegSamplingPipeline:
    """
    巨核细胞采样流水线：
    - 从 SmearProject 中取 40x 层 tiles；
    - 构建 HeatmapGrid；
    - 构建 forbidden_mask；
    - 按 MEG_cell_type 提取巨核细胞；
    - 利用外部传入的 wbc_rects 计算 WBC 中心并排序视野；
    - 按 target_cell_num_MEG 截断并生成 TaskOutput 列表。
    """

    def __init__(self, config: BM40Config):
        self.cfg = config
        self.grid = None
        self.forbidden_mask = None

    def run_meg(
        self,
        project: SmearProject,
        wbc_rects: List[List[int]],
    ) -> List[TaskOutput]:
        # 1. 从项目中提取 40x 扫描层
        layer_40x_id = 0
        layer_40x = project.layers[layer_40x_id]
        if not layer_40x:
            print("[ERROR][MEG] 项目中缺少 40x 扫描层数据")
            return []

        # 2. 获取该层所有 tiles
        tiles = list(layer_40x.tiles.values())
        if not tiles:
            print("[ERROR][MEG] 40x 层中没有找到有效的 Tile 数据")
            return []

        # 3. 构建 HeatmapGrid（仅依赖 scores，不涉及细胞类型）
        self.grid = build_score_heatmap(tiles, config=self.cfg)

        # 4. 构建禁区掩码（与 WBC 流程保持一致）
        self.forbidden_mask = build_forbidden_mask(self.grid, self.cfg, tiles=tiles)

        # 5. 提取巨核细胞全局坐标（根据 config.MEG_cell_type）
        all_meg_cells_list = []
        meg_type = getattr(self.cfg, "MEG_cell_type", None)

        if meg_type is None:
            print("[WARN][MEG] 配置中未设置 MEG_cell_type，将不会找到任何巨核细胞。")
        else:
            for t in tiles:
                if t.x is None or t.y is None:
                    continue
                for c in t.cells:
                    if getattr(c, "cell_type", None) != meg_type:
                        continue
                    all_meg_cells_list.append(
                        [
                            c.cell_xmin + t.x,
                            c.cell_ymin + t.y,
                            c.cell_xmax + t.x,
                            c.cell_ymax + t.y,
                        ]
                    )

        if not all_meg_cells_list:
            print("[INFO][MEG] 未在 40x tiles 中找到任何巨核细胞。")
            return []

        all_meg_cells_array = np.array(all_meg_cells_list, dtype=np.float32)

        # 6. 使用 forbidden_mask 过滤巨核细胞（不在禁区内的才保留）
        centers = 0.5 * (all_meg_cells_array[:, 0:2] + all_meg_cells_array[:, 2:4])
        gxs = ((centers[:, 0] - self.grid.origin_x) // self.grid.cell_size).astype(np.int32)
        gys = ((centers[:, 1] - self.grid.origin_y) // self.grid.cell_size).astype(np.int32)

        rows, cols = self.forbidden_mask.shape
        in_bounds = (gxs >= 0) & (gxs < cols) & (gys >= 0) & (gys < rows)
        keep_idx = np.where(
            in_bounds & (self.forbidden_mask[gys, gxs] == 0)
        )[0]  # 0 表示非禁区

        if keep_idx.size == 0:
            print("[INFO][MEG] 所有巨核细胞均落在禁区或网格之外。")
            return []

        valid_meg_cells = all_meg_cells_array[keep_idx]

        # 7. 调用巨核任务生成函数（内部完成排序和截断）
        meg_tasks = generate_meg_view_tasks(
            meg_cell_bounds=valid_meg_cells,
            config=self.cfg,
            wbc_rects=wbc_rects,
        )

        return meg_tasks


# def run_meg(
#     project: SmearProject,
#     config: BM40Config,
#     wbc_rects: List[List[int]],
# ) -> List[TaskOutput]:
#     """
#     对外的一键 MEG 采样入口：
#     - project: SmearProject 实例
#     - config: BM40Config（包含 MEG_cell_type / target_cell_num_MEG 等）
#     - wbc_rects: 外部传入的 WBC 视野列表 [[X, Y, W, H], ...]
#     """
#     pipeline = MegSamplingPipeline(config=config)
#     return pipeline.run_meg(project=project, wbc_rects=wbc_rects)