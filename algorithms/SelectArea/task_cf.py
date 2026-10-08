# task_cf.py
"""脑脊液（CF）FOCUS_POINT：密集区提取与视野平铺。"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .config import BM40Config
from .data_structure import CellOutput


def factor_grid(n: int) -> Tuple[int, int]:
    """将 focus_point 拆成 (rows, cols)，优先接近正方形。"""
    n = max(1, int(n))
    best = (1, n)
    best_score = n
    for rows in range(1, n + 1):
        if n % rows != 0:
            continue
        cols = n // rows
        score = abs(rows - cols)
        if score < best_score:
            best = (rows, cols)
            best_score = score
    return best


def find_densest_cell_rect(
    cell_matrix: np.ndarray,
    grid,
    config: BM40Config,
    all_cells_array: np.ndarray,
) -> Tuple[int, int, int, int]:
    """
    从细胞密度矩阵提取主细胞团的全局轴对齐矩形 [xmin, ymin, xmax, ymax]。

    1) 非零格闭运算后取最大连通域（主团）；
    2) 在主团内以密度加权质心为中心，由近及远累积细胞质量，
       至覆盖 cf_dense_mass_frac（默认 95%）后取外接框。
       可抑制边角稀疏细胞把框撑满整幅 ROI，又避免相对峰值抠核过狠。
    """
    if cell_matrix.size == 0 or float(np.sum(cell_matrix)) <= 0:
        raise ValueError("细胞密度矩阵为空，无法提取密集区")

    k = max(1, int(getattr(config, "cf_dense_close_ksize", 5)))
    kernel = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) if k > 1 else None
    )

    binary = (cell_matrix > 0).astype(np.uint8)
    if kernel is not None:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    rows, cols = cell_matrix.shape
    if num <= 1:
        main_mask = cell_matrix > 0
    else:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        main_mask = labels == largest

    if not np.any(main_mask):
        raise ValueError("主团为空，无法提取密集区")

    # --- 质量心累积覆盖 ---
    ys, xs = np.where(main_mask & (cell_matrix > 0))
    if ys.size == 0:
        ys, xs = np.where(main_mask)
        weights = np.ones(ys.size, dtype=np.float64)
    else:
        weights = cell_matrix[ys, xs].astype(np.float64)

    mass_frac = float(getattr(config, "cf_dense_mass_frac", 0.90))
    mass_frac = min(max(mass_frac, 0.5), 1.0)

    if ys.size == 1 or mass_frac >= 1.0 - 1e-9:
        sel_ys, sel_xs = ys, xs
    else:
        cy = float(np.average(ys, weights=weights))
        cx = float(np.average(xs, weights=weights))
        d2 = (ys.astype(np.float64) - cy) ** 2 + (xs.astype(np.float64) - cx) ** 2
        order = np.argsort(d2, kind="mergesort")
        cum = np.cumsum(weights[order])
        total = float(cum[-1])
        k_keep = int(np.searchsorted(cum, mass_frac * total, side="left")) + 1
        k_keep = max(1, min(k_keep, order.size))
        keep = order[:k_keep]
        sel_ys, sel_xs = ys[keep], xs[keep]

    core_mask = np.zeros_like(main_mask, dtype=bool)
    core_mask[sel_ys, sel_xs] = True
    gx0, gx1 = int(sel_xs.min()), int(sel_xs.max()) + 1
    gy0, gy1 = int(sel_ys.min()), int(sel_ys.max()) + 1

    cs = float(grid.cell_size)
    ox, oy = float(grid.origin_x), float(grid.origin_y)
    centers = 0.5 * (all_cells_array[:, 0:2] + all_cells_array[:, 2:4])
    gxs = ((centers[:, 0] - ox) // cs).astype(np.int32)
    gys = ((centers[:, 1] - oy) // cs).astype(np.int32)
    in_blob = (
        (gxs >= 0)
        & (gxs < cols)
        & (gys >= 0)
        & (gys < rows)
        & core_mask[np.clip(gys, 0, rows - 1), np.clip(gxs, 0, cols - 1)]
    )

    if np.any(in_blob):
        sub = all_cells_array[in_blob]
        xmin = int(math.floor(float(sub[:, 0].min())))
        ymin = int(math.floor(float(sub[:, 1].min())))
        xmax = int(math.ceil(float(sub[:, 2].max())))
        ymax = int(math.ceil(float(sub[:, 3].max())))
    else:
        xmin = int(round(ox + gx0 * cs))
        ymin = int(round(oy + gy0 * cs))
        xmax = int(round(ox + gx1 * cs))
        ymax = int(round(oy + gy1 * cs))

    if xmax <= xmin:
        xmax = xmin + 1
    if ymax <= ymin:
        ymax = ymin + 1
    return xmin, ymin, xmax, ymax


def _build_cell_center_integral(
    centers: np.ndarray,
    origin_x: float,
    origin_y: float,
    width: int,
    height: int,
    bin_size: int = 16,
) -> Tuple[np.ndarray, int]:
    """
    把细胞中心投到分箱计数图再做积分。
    返回 (integral, bin_size)；积分图像素对应物理边长 bin_size。
    """
    bin_size = max(1, int(bin_size))
    cols = max(1, int(math.ceil(width / bin_size)))
    rows = max(1, int(math.ceil(height / bin_size)))
    img = np.zeros((rows, cols), dtype=np.float32)
    if centers.size == 0:
        return cv2.integral(img), bin_size
    xs = np.floor((centers[:, 0] - origin_x) / bin_size).astype(np.int32)
    ys = np.floor((centers[:, 1] - origin_y) / bin_size).astype(np.int32)
    valid = (xs >= 0) & (xs < cols) & (ys >= 0) & (ys < rows)
    if np.any(valid):
        np.add.at(img, (ys[valid], xs[valid]), 1.0)
    return cv2.integral(img), bin_size


def _integral_rect_sum(
    integral: np.ndarray, x0: int, y0: int, x1: int, y1: int
) -> float:
    """半开区间 [x0,x1)×[y0,y1) 求和（分箱坐标）；自动裁剪。"""
    h = integral.shape[0] - 1
    w = integral.shape[1] - 1
    x0 = max(0, min(w, int(x0)))
    x1 = max(0, min(w, int(x1)))
    y0 = max(0, min(h, int(y0)))
    y1 = max(0, min(h, int(y1)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(
        integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    )


def _snap_up_to_bin(value: float, origin: float, bin_size: int) -> float:
    """不小于 value、且相对 origin 为 bin_size 整数倍的坐标。"""
    k = int(math.ceil((value - origin) / bin_size - 1e-9))
    return origin + k * bin_size


def _snap_down_to_bin(value: float, origin: float, bin_size: int) -> float:
    """不大于 value、且相对 origin 为 bin_size 整数倍的坐标。"""
    k = int(math.floor((value - origin) / bin_size + 1e-9))
    return origin + k * bin_size


def place_focus_point_views(
    task_rect: Tuple[int, int, int, int],
    all_cells_array: np.ndarray,
    config: BM40Config,
    focus_point: int,
) -> List[Tuple[int, int, int, int]]:
    """
    将 task_rect 均分为 focus_point 个子区，每个子区内放一个视野：
    - 先保证平铺覆盖整个密集区；
    - 再在子区允许范围内滑动，使该视野内细胞数最多。

    滑动位置与步长均对齐到 bin_size 格线，与积分图离散格子一致。
    返回 [(xmin, ymin, xmax, ymax), ...]。
    """
    rx0, ry0, rx1, ry1 = [int(v) for v in task_rect]
    view_w = max(1, int(config.x100_rect_width))
    view_h = max(1, int(config.x100_rect_height))
    n_rows, n_cols = factor_grid(focus_point)

    rect_w = max(1.0, float(rx1 - rx0))
    rect_h = max(1.0, float(ry1 - ry0))
    tile_w = rect_w / n_cols
    tile_h = rect_h / n_rows

    centers = (
        0.5 * (all_cells_array[:, 0:2] + all_cells_array[:, 2:4])
        if all_cells_array.size
        else np.empty((0, 2), dtype=np.float64)
    )

    pad = max(view_w, view_h)
    origin_x = float(rx0 - pad)
    origin_y = float(ry0 - pad)
    integ_w = int(math.ceil((rx1 + pad) - origin_x)) + 1
    integ_h = int(math.ceil((ry1 + pad) - origin_y)) + 1
    # 分箱边长：约视野宽的 1/16，且不超过 32，兼顾速度与精度
    bin_size = int(
        max(8, min(32, view_w // max(int(getattr(config, "cf_fov_refine_steps", 5)), 1)))
    )
    integral, bin_size = _build_cell_center_integral(
        centers, origin_x, origin_y, integ_w, integ_h, bin_size=bin_size
    )

    refine = max(1, int(getattr(config, "cf_fov_refine_steps", 5)))
    # 步长取 bin_size 的整数倍，且不少于 view/refine
    raw_step_x = max(float(bin_size), view_w / float(refine))
    raw_step_y = max(float(bin_size), view_h / float(refine))
    step_x = float(bin_size * max(1, int(math.ceil(raw_step_x / bin_size))))
    step_y = float(bin_size * max(1, int(math.ceil(raw_step_y / bin_size))))
    # 积分查询窗口：视野换算成整格数（向上取整，不漏边）
    view_bins_w = max(1, int(math.ceil(view_w / bin_size)))
    view_bins_h = max(1, int(math.ceil(view_h / bin_size)))
    views: List[Tuple[int, int, int, int]] = []

    for iy in range(n_rows):
        for ix in range(n_cols):
            tx0 = rx0 + ix * tile_w
            ty0 = ry0 + iy * tile_h
            tx1 = tx0 + tile_w
            ty1 = ty0 + tile_h

            if tile_w >= view_w:
                x_lo, x_hi = tx0, tx1 - view_w
            else:
                cx = 0.5 * (tx0 + tx1) - 0.5 * view_w
                x_lo = x_hi = cx
            if tile_h >= view_h:
                y_lo, y_hi = ty0, ty1 - view_h
            else:
                cy = 0.5 * (ty0 + ty1) - 0.5 * view_h
                y_lo = y_hi = cy

            # 候选左上角对齐到积分格线（相对 origin 为 bin_size 整数倍）
            if x_hi > x_lo:
                x_lo_a = _snap_up_to_bin(x_lo, origin_x, bin_size)
                x_hi_a = _snap_down_to_bin(x_hi, origin_x, bin_size)
                if x_lo_a > x_hi_a:
                    x_lo_a = x_hi_a = _snap_down_to_bin(
                        0.5 * (x_lo + x_hi), origin_x, bin_size
                    )
                xs = np.arange(x_lo_a, x_hi_a + 0.5 * step_x, step_x, dtype=np.float64)
                if xs.size == 0:
                    xs = np.array([x_lo_a], dtype=np.float64)
                elif xs[-1] < x_hi_a - 1e-6:
                    xs = np.append(xs, x_hi_a)
            else:
                xs = np.array(
                    [_snap_down_to_bin(x_lo, origin_x, bin_size)], dtype=np.float64
                )

            if y_hi > y_lo:
                y_lo_a = _snap_up_to_bin(y_lo, origin_y, bin_size)
                y_hi_a = _snap_down_to_bin(y_hi, origin_y, bin_size)
                if y_lo_a > y_hi_a:
                    y_lo_a = y_hi_a = _snap_down_to_bin(
                        0.5 * (y_lo + y_hi), origin_y, bin_size
                    )
                ys = np.arange(y_lo_a, y_hi_a + 0.5 * step_y, step_y, dtype=np.float64)
                if ys.size == 0:
                    ys = np.array([y_lo_a], dtype=np.float64)
                elif ys[-1] < y_hi_a - 1e-6:
                    ys = np.append(ys, y_hi_a)
            else:
                ys = np.array(
                    [_snap_down_to_bin(y_lo, origin_y, bin_size)], dtype=np.float64
                )

            best = (float(xs[len(xs) // 2]), float(ys[len(ys) // 2]))
            best_cnt = -1.0
            for y in ys:
                for x in xs:
                    ix0 = int(round((x - origin_x) / bin_size))
                    iy0 = int(round((y - origin_y) / bin_size))
                    ix1 = ix0 + view_bins_w
                    iy1 = iy0 + view_bins_h
                    cnt = _integral_rect_sum(integral, ix0, iy0, ix1, iy1)
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best = (float(x), float(y))

            bx, by = best
            views.append(
                (
                    int(round(bx)),
                    int(round(by)),
                    int(round(bx + view_w)),
                    int(round(by + view_h)),
                )
            )
    return views


def assign_cells_to_views(
    all_cells_array: np.ndarray,
    views: Sequence[Tuple[int, int, int, int]],
) -> List[List[CellOutput]]:
    """贪心：细胞中心落入视野则分配；已被占用的细胞不再分给后续视野。"""
    assigned: List[List[CellOutput]] = [[] for _ in views]
    if all_cells_array.size == 0:
        return assigned

    centers = 0.5 * (all_cells_array[:, 0:2] + all_cells_array[:, 2:4])
    used = np.zeros(len(all_cells_array), dtype=bool)
    for i, (x0, y0, x1, y1) in enumerate(views):
        mask = (
            (centers[:, 0] >= x0)
            & (centers[:, 0] < x1)
            & (centers[:, 1] >= y0)
            & (centers[:, 1] < y1)
            & (~used)
        )
        idxs = np.where(mask)[0]
        for idx in idxs:
            assigned[i].append(
                CellOutput(
                    cell_xmin=int(round(all_cells_array[idx, 0])),
                    cell_ymin=int(round(all_cells_array[idx, 1])),
                    cell_xmax=int(round(all_cells_array[idx, 2])),
                    cell_ymax=int(round(all_cells_array[idx, 3])),
                )
            )
        used[mask] = True
    return assigned
