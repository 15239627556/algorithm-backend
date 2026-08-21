# pipeline_meg.py
import numpy as np
from typing import List, Optional, TYPE_CHECKING
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from project.smear_project import SmearProject
from .config import BM40Config
from .data_structure import TaskOutput
from .heatmaps import HeatmapGrid, build_score_heatmap
from .task_region_extraction import (
    build_bubble_forbidden_mask,
    build_forbidden_mask,
)
from .task_meg import generate_meg_view_tasks

if TYPE_CHECKING:
    from project.roi_store import RoiDataset


def _collect_meg_cells_fast(tiles, meg_type: int):
    """
    快速提取 MEG 细胞全局坐标，减少循环内重复属性查找开销。
    返回 [[xmin, ymin, xmax, ymax], ...]。
    """
    all_meg_cells_list = []
    append = all_meg_cells_list.append

    for t in tiles:
        tx, ty = t.x, t.y
        if tx is None or ty is None:
            continue

        for c in t.cells:
            if c.cell_type != meg_type:
                continue
            append(
                [
                    c.cell_xmin + tx,
                    c.cell_ymin + ty,
                    c.cell_xmax + tx,
                    c.cell_ymax + ty,
                ]
            )

    return all_meg_cells_list


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
        self.bubble_forbidden_mask = None

    def run_meg(
        self,
        project: SmearProject | None = None,
        wbc_rects: List[List[int]] | None = None,
        *,
        roi: Optional["RoiDataset"] = None,
        heatmap_grid: Optional[HeatmapGrid] = None,
    ) -> List[TaskOutput]:
        if roi is not None:
            tiles = roi.tiles
        else:
            if project is None:
                raise ValueError("run_meg() 需要 project 或 roi 之一")
            layer_40x = project.get_layer(self.cfg.dpi)
            if not layer_40x:
                print("[ERROR][MEG] 项目中缺少 40x 扫描层数据")
                return []
            tiles = list(layer_40x.tiles.values())

        if not tiles:
            print("[ERROR][MEG] 40x 层中没有找到有效的 Tile 数据")
            return []

        if not wbc_rects:
            print(
                "[ERROR][MEG] 未传入任何有核细胞，"
                "无法计算 MEG 排序参考。"
            )
            return []
        wbc_rects_array = np.asarray(wbc_rects, dtype=np.float32)

        # 3. 先提取巨核细胞全局坐标（根据 config.MEG_cell_type）。
        # 这样在“无巨核细胞”时可直接返回，避免无意义地构建热力图和禁区掩码。
        meg_type = getattr(self.cfg, "MEG_cell_type", None)

        if meg_type is None:
            print("[WARN][MEG] 配置中未设置 MEG_cell_type，将不会找到任何巨核细胞。")
            all_meg_cells_array = np.empty((0, 4), dtype=np.float32)
        elif roi is not None:
            all_meg_cells_array = roi.cells_xyxy_by_type(meg_type)
        else:
            all_meg_cells_list = _collect_meg_cells_fast(tiles, meg_type)
            all_meg_cells_array = np.asarray(all_meg_cells_list, dtype=np.float32)

        if all_meg_cells_array.size == 0:
            print("[INFO][MEG] 未在 40x tiles 中找到任何巨核细胞。")
            return []

        # 4. 热力图：可复用有核已算好的 grid，否则与 WBC 同样构建
        if heatmap_grid is not None:
            self.grid = heatmap_grid
        else:
            self.grid = (
                roi.build_heatmap_grid(self.cfg)
                if roi is not None
                else build_score_heatmap(tiles, config=self.cfg)
            )

        # 5. 禁区：与有核相同，label=5 与空泡分开构建后再合并过滤细胞
        self.forbidden_mask = build_forbidden_mask(self.grid, self.cfg, tiles=tiles)
        self.bubble_forbidden_mask = build_bubble_forbidden_mask(self.grid, self.cfg)
        region_forbidden = self.forbidden_mask
        if self.bubble_forbidden_mask is not None:
            region_forbidden = np.where((self.forbidden_mask > 0) | (self.bubble_forbidden_mask > 0),1,0,).astype(np.uint8)

        # 6. 使用合并禁区过滤巨核细胞（不在禁区内的才保留）
        centers = 0.5 * (all_meg_cells_array[:, 0:2] + all_meg_cells_array[:, 2:4])
        gxs = ((centers[:, 0] - self.grid.origin_x) // self.grid.cell_size).astype(np.int32)
        gys = ((centers[:, 1] - self.grid.origin_y) // self.grid.cell_size).astype(np.int32)

        rows, cols = region_forbidden.shape
        in_bounds = (gxs >= 0) & (gxs < cols) & (gys >= 0) & (gys < rows)
        if not np.any(in_bounds):
            print("[INFO][MEG] 所有巨核细胞均落在网格之外。")
            return []

        valid_pos_idx = np.where(in_bounds)[0]
        valid_gxs = gxs[in_bounds]
        valid_gys = gys[in_bounds]
        non_forbidden = region_forbidden[valid_gys, valid_gxs] == 0
        keep_idx = valid_pos_idx[non_forbidden]  # 0 表示非禁区

        if keep_idx.size == 0:
            print("[INFO][MEG] 所有巨核细胞均落在禁区或网格之外。")
            return []

        valid_meg_cells = all_meg_cells_array[keep_idx]

        # 7. 调用巨核任务生成函数（内部完成排序和截断）
        meg_tasks = generate_meg_view_tasks(
            meg_cell_bounds=valid_meg_cells,
            config=self.cfg,
            wbc_rects=wbc_rects_array,
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
#     - wbc_rects: 外部传入的 WBC 视野列表 [[x, y, w, h], ...]
#     """
#     pipeline = MegSamplingPipeline(config=config)
#     return pipeline.run_meg(project=project, wbc_rects=wbc_rects)