"""
过滤瓦片边缘附近不完整、易被误检的框（相对于单张 tile 的局部坐标）。

规则（与产品约定一致即可按需改常量）：
1. 任一边与 tile 边界重合或超出（到边最小距离 <= 0）→ 删除（不完整/贴边细胞）。
2. 任一边到 tile 边界距离 <= edge_margin_px 且框为细长矩形（长边/短边 >= min_aspect_ratio）→ 删除
   （靠边细长框多为截断目标）。

坐标约定与 dedup_cells_across_tiles 一致：cell 为 tile 内局部 xyxy，tile 范围 [0, w) × [0, h)。
"""
from __future__ import annotations

from typing import List

from project.cells import Cell
from project.tiles import Tile

# 默认：距边 ≤5px 且长宽比判定为长方形时过滤
DEFAULT_EDGE_MARGIN_PX = 5
# 长边/短边 ≥ 此值视为细长矩形（“长方形”）
DEFAULT_MIN_ASPECT_RATIO = 2.0


def cell_should_drop_edge_incomplete(
    cell: Cell,
    tile_w: int,
    tile_h: int,
    *,
    edge_margin_px: int = DEFAULT_EDGE_MARGIN_PX,
    min_aspect_ratio: float = DEFAULT_MIN_ASPECT_RATIO,
) -> bool:
    """
    若该细胞在「贴边 / 近边细长框」规则下应丢弃则返回 True。
    """
    xmin = int(cell.cell_xmin)
    ymin = int(cell.cell_ymin)
    xmax = int(cell.cell_xmax)
    ymax = int(cell.cell_ymax)

    w = xmax - xmin
    h = ymax - ymin
    if w <= 0 or h <= 0:
        return True

    tw = int(tile_w)
    th = int(tile_h)
    if tw <= 0 or th <= 0:
        return False

    dist_left = float(xmin)
    dist_top = float(ymin)
    dist_right = float(tw - xmax)
    dist_bottom = float(th - ymax)
    min_d = min(dist_left, dist_top, dist_right, dist_bottom)

    # 完全贴边或越界（不完整）
    if min_d <= 0:
        return True

    short = float(min(w, h))
    long_side = float(max(w, h))
    aspect = long_side / max(short, 1e-6)
    is_elongated = aspect >= float(min_aspect_ratio)

    # 靠边的细长框（距边 ≤ margin）
    if min_d <= float(edge_margin_px) and is_elongated:
        return True

    return False


def filter_edge_incomplete_cells(
    tiles: List[Tile],
    *,
    task_tile_w: int,
    task_tile_h: int,
    edge_margin_px: int = DEFAULT_EDGE_MARGIN_PX,
    min_aspect_ratio: float = DEFAULT_MIN_ASPECT_RATIO,
) -> List[Tile]:
    """
    就地过滤每个 tile.cells，返回同一列表引用以便链式调用。

    :param tiles: Tile 列表（cells 为瓦片局部坐标）
    :param task_tile_w: 任务声明的瓦片宽度（与 create/upload 时的 tile_width 一致）
    :param task_tile_h: 任务声明的瓦片高度（与 create/upload 时的 tile_height 一致）
    :note: 单块若 tile.w/tile.h 有效则优先用单块尺寸，否则退回任务尺寸（兼容异常数据）
    """
    tw_task = int(task_tile_w)
    th_task = int(task_tile_h)
    for tile in tiles:
        tw = int(tile.w) if int(tile.w or 0) > 0 else tw_task
        th = int(tile.h) if int(tile.h or 0) > 0 else th_task
        cells = tile.cells or []
        if not cells:
            continue
        tile.cells = [
            c
            for c in cells
            if not cell_should_drop_edge_incomplete(
                c,
                tw,
                th,
                edge_margin_px=edge_margin_px,
                min_aspect_ratio=min_aspect_ratio,
            )
        ]
    return tiles
