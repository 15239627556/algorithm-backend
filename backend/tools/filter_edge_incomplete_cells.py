"""
过滤瓦片边缘附近不完整、易被误检的框（相对于单张 tile 的局部坐标）。

规则（与产品约定一致即可按需改常量）：
1. 任一边与 tile 边界重合、超出，或到边界距离 <= touch_tolerance_px → 删除（贴边/明显裁切）。
   - 说明：模型多为 xywh→cell_xmax=int(x+w)。后处理裁边或 xmax 表示「最右列下标」时，
     会出现 tw-xmax==1 但视觉上仍贴右/下边缘的情况，仅靠 <=0 删不干净。
2. 任一边到 tile 边界距离 <= edge_margin_px 且框为细长矩形（长边/短边 >= min_aspect_ratio）→ 删除
   （靠边细长框多为截断目标）。

坐标约定与 dedup_cells_across_tiles 一致：cell 为 tile 内局部 xyxy；图像宽 tw、高 th，对应列 0..tw-1、行 0..th-1。
"""
from __future__ import annotations

from typing import Any, Dict, List

from project.cells import Cell
from project.tiles import Tile

# 「完全贴边」判定的额外容差（像素）：解决 xmax/ymax 取整与裁边导致的 tw-xmax==1 仍贴边
DEFAULT_EDGE_TOUCH_TOLERANCE_PX = 1
# 默认：距边 ≤5px 且长宽比判定为长方形时过滤
DEFAULT_EDGE_MARGIN_PX = 5
# 长边/短边 ≥ 此值视为细长矩形（“长方形”）
DEFAULT_MIN_ASPECT_RATIO = 2.0


def cell_should_drop_edge_incomplete(
    cell: Cell,
    tile_w: int,
    tile_h: int,
    *,
    touch_tolerance_px: int = DEFAULT_EDGE_TOUCH_TOLERANCE_PX,
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

    # 贴边 / 越界 / 到边 ≤ touch_tolerance（兼容 xmax 闭区间或裁边导致的 1px 偏差）
    if min_d <= float(touch_tolerance_px):
        return True

    short = float(min(w, h))
    long_side = float(max(w, h))
    aspect = long_side / max(short, 1e-6)
    is_elongated = aspect >= float(min_aspect_ratio)

    # 靠边的细长框（距边 ≤ margin）
    if min_d <= float(edge_margin_px) and is_elongated:
        return True

    return False


def filter_cell_dicts_edge_incomplete(
    cell_list: List[Dict[str, Any]],
    tile_w: int,
    tile_h: int,
    *,
    touch_tolerance_px: int = DEFAULT_EDGE_TOUCH_TOLERANCE_PX,
    edge_margin_px: int = DEFAULT_EDGE_MARGIN_PX,
    min_aspect_ratio: float = DEFAULT_MIN_ASPECT_RATIO,
) -> List[Dict[str, Any]]:
    """
    对 analyze_cell_image 等接口返回的 cell_list（每项含 cell_xmin/ymin/xmax/ymax）做与瓦片一致的边缘过滤。
    无法解析 bbox 的项原样保留。
    """
    out: List[Dict[str, Any]] = []
    tw = int(tile_w)
    th = int(tile_h)
    for d in cell_list:
        if not isinstance(d, dict):
            out.append(d)
            continue
        try:
            c = Cell(
                cell_xmin=int(d["cell_xmin"]),
                cell_ymin=int(d["cell_ymin"]),
                cell_xmax=int(d["cell_xmax"]),
                cell_ymax=int(d["cell_ymax"]),
                cell_type=0,
                cell_type_name="",
            )
        except (KeyError, TypeError, ValueError):
            out.append(d)
            continue
        if not cell_should_drop_edge_incomplete(
            c,
            tw,
            th,
            touch_tolerance_px=touch_tolerance_px,
            edge_margin_px=edge_margin_px,
            min_aspect_ratio=min_aspect_ratio,
        ):
            out.append(d)
    return out


def filter_edge_incomplete_cells(
    tiles: List[Tile],
    *,
    task_tile_w: int,
    task_tile_h: int,
    touch_tolerance_px: int = DEFAULT_EDGE_TOUCH_TOLERANCE_PX,
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
                touch_tolerance_px=touch_tolerance_px,
                edge_margin_px=edge_margin_px,
                min_aspect_ratio=min_aspect_ratio,
            )
        ]
    return tiles
