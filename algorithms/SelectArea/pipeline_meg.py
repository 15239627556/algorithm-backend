# pipeline_meg.py
import numpy as np
from typing import List, Optional, TYPE_CHECKING
import math
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from project.smear_project import SmearProject
from .config import BM40Config
from .data_structure import TaskOutput
from .heatmaps import HeatmapGrid, compute_global_bounds_from_tiles
from .task_region_extraction import build_forbidden_mask
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


def _build_lightweight_grid_for_mask(tiles, cell_size: float) -> HeatmapGrid:
    """
    仅为禁区过滤构建轻量网格：
    - 只计算边界与网格尺寸；
    - 不写入 score/weight（避免 build_score_heatmap 的重计算）。
    """
    min_x, min_y, max_x, max_y = compute_global_bounds_from_tiles(tiles)
    width = max_x - min_x
    height = max_y - min_y
    cols = int(math.ceil(width / cell_size))
    rows = int(math.ceil(height / cell_size))
    values = np.zeros((rows, cols), dtype=np.float32)
    weights = np.zeros((rows, cols), dtype=np.float32)
    return HeatmapGrid(
        origin_x=min_x,
        origin_y=min_y,
        cell_size=cell_size,
        values=values,
        weights=weights,
    )


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
        project: SmearProject | None = None,
        wbc_rects: List[List[int]] | None = None,
        *,
        roi: Optional["RoiDataset"] = None,
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

        # 4. 构建 HeatmapGrid（仅依赖 scores，不涉及细胞类型）
        self.grid = _build_lightweight_grid_for_mask(tiles, self.cfg.cell_size)

        # 5. 构建禁区掩码（与 WBC 流程保持一致）
        self.forbidden_mask = build_forbidden_mask(self.grid, self.cfg, tiles=tiles)

        # 6. 使用 forbidden_mask 过滤巨核细胞（不在禁区内的才保留）
        centers = 0.5 * (all_meg_cells_array[:, 0:2] + all_meg_cells_array[:, 2:4])
        gxs = ((centers[:, 0] - self.grid.origin_x) // self.grid.cell_size).astype(np.int32)
        gys = ((centers[:, 1] - self.grid.origin_y) // self.grid.cell_size).astype(np.int32)

        rows, cols = self.forbidden_mask.shape
        in_bounds = (gxs >= 0) & (gxs < cols) & (gys >= 0) & (gys < rows)
        if not np.any(in_bounds):
            print("[INFO][MEG] 所有巨核细胞均落在网格之外。")
            return []

        valid_pos_idx = np.where(in_bounds)[0]
        valid_gxs = gxs[in_bounds]
        valid_gys = gys[in_bounds]
        non_forbidden = self.forbidden_mask[valid_gys, valid_gxs] == 0
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