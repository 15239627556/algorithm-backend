from __future__ import annotations

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2] 
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from project.tiles import Tile

import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional

from .heatmaps import HeatmapGrid
from .config import BM40Config
from .data_structure import SelectionResult, Rect


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """填充二值主体内部孔洞，图像边界连通的背景保持为背景。"""
    if mask.size == 0:
        return mask

    # 补一圈确定的背景，使所有“外部背景”在 padding 中连通，
    # 随后只需从左上角执行一次 floodFill。
    padded = cv2.copyMakeBorder(
        mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
    )
    inv = cv2.bitwise_not(padded)
    flood = inv.copy()
    flood_mask = np.zeros(
        (padded.shape[0] + 2, padded.shape[1] + 2), dtype=np.uint8
    )
    cv2.floodFill(flood, flood_mask, (0, 0), 128)

    holes = np.where(flood == 255, 255, 0).astype(np.uint8)
    filled = cv2.bitwise_or(padded, holes)
    return filled[1:-1, 1:-1]


def build_valid_score_mask(
    grid: HeatmapGrid,
    config: BM40Config,
    raw_score_map: np.ndarray,
) -> np.ndarray:
    """以有评分区域的重复最低分为背景，构建并修整前景主体掩码。"""
    score_map = np.asarray(raw_score_map, dtype=np.float32)
    # weights == 0 的格子只是没有计算到评分，不能参与最低背景分统计。
    covered = np.isfinite(score_map) & (grid.weights > 0)
    if not np.any(covered):
        return np.zeros(score_map.shape, dtype=np.uint8)

    # BM/PB 均采用同一规则：有评分区域中的重复最低分代表空白背景，
    # 严格高于最低分的区域是原始前景。isclose 用于吸收加权平均的浮点误差。
    background_min = float(np.min(score_map[covered]))
    is_background = np.isclose(
        score_map, background_min, rtol=1e-5, atol=1e-5
    )
    score_valid = covered & ~is_background

    valid = np.where(score_valid, 255, 0).astype(np.uint8)
    if not np.any(valid):
        return valid

    if config.Smear_type.upper() == "PB":
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, config.coast_open_ksize_pb
        )
        valid = cv2.morphologyEx(
            valid,
            cv2.MORPH_OPEN,
            kernel,
            iterations=max(1, int(config.coast_open_iters_pb)),
        )
    else:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, config.coast_open_ksize_bm
        )
        valid = cv2.morphologyEx(
            valid,
            cv2.MORPH_OPEN,
            kernel,
            iterations=max(1, int(config.coast_open_iters_bm)),
        )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        valid, connectivity=8, ltype=cv2.CV_32S
    )
    if count > 1:
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        valid = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    return _fill_mask_holes(valid)


def get_valid_score_range(grid: HeatmapGrid, config: BM40Config) -> float:
    """返回海岸惩罚前有效评分区的 max-min，供接近分候选判定。

    仅统计 weights>0 的真实评分格；形态学 valid 掩码扩到无数据格时，
    finalize 的 heatmap_penalty_value 填充分不参与范围计算。
    """
    raw = grid.finalize(fill_value=config.heatmap_penalty_value)
    valid = build_valid_score_mask(grid, config, raw) > 0
    scored = valid & (grid.weights > 0)
    if not np.any(scored):
        return 0.0
    valid_scores = raw[scored]
    return float(np.max(valid_scores) - np.min(valid_scores))


def _normalize_heatmap_for_debug(
    score_map: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """仅按有效评分区归一化为 uint8，背景保持黑色。"""
    valid = (valid_mask > 0) & np.isfinite(score_map)
    image = np.zeros(score_map.shape, dtype=np.uint8)
    if not np.any(valid):
        return image

    values = score_map[valid]
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    if value_max - value_min <= 1e-12:
        image[valid] = 255
        return image

    normalized = (score_map[valid] - value_min) * (255.0 / (value_max - value_min))
    image[valid] = np.clip(normalized, 0, 255).astype(np.uint8)
    return image


def _save_coast_debug_images(
    raw_score_map: np.ndarray,
    penalized_score_map: np.ndarray,
    valid_mask: np.ndarray,
    inland_mask: np.ndarray,
    coast_band: np.ndarray,
    coast_penalty: np.ndarray,
    top_bottom_penalty: np.ndarray,
    effective_penalty: np.ndarray,
) -> None:
    """保存海岸带、惩罚区域及惩罚前后热力图。"""
    if os.getenv("SELECT_AREA_DEBUG_COAST") != "1":
        return

    debug_dir = Path(
        os.getenv(
            "SELECT_AREA_DEBUG_COAST_DIR",
            str(Path(__file__).resolve().parent / "output" / "coast_debug"),
        )
    )
    debug_dir.mkdir(parents=True, exist_ok=True)

    raw_vis = _normalize_heatmap_for_debug(raw_score_map, valid_mask)
    penalized_vis = _normalize_heatmap_for_debug(
        penalized_score_map, valid_mask
    )

    # 与红细胞 C++ 调试图一致：灰度底图，绿=内陆，红=腐蚀得到的海岸带。
    coast_vis = cv2.cvtColor(raw_vis, cv2.COLOR_GRAY2BGR)
    coast_vis[inland_mask > 0] = (0, 180, 0)
    coast_vis[coast_band > 0] = (0, 0, 255)

    max_penalty = float(np.max(effective_penalty))
    if max_penalty > 0:
        penalty_gray = np.clip(
            effective_penalty * (255.0 / max_penalty), 0, 255
        ).astype(np.uint8)
    else:
        penalty_gray = np.zeros(effective_penalty.shape, dtype=np.uint8)
    penalty_gray[valid_mask == 0] = 0
    penalty_color = cv2.applyColorMap(penalty_gray, cv2.COLORMAP_JET)
    penalty_color[valid_mask == 0] = 0

    debug_images = {
        "valid_mask.png": valid_mask,
        "inland_mask.png": inland_mask,
        "coast_band.png": coast_vis,
        "heatmap_raw.png": raw_vis,
        "heatmap_penalized.png": penalized_vis,
        "coast_penalty.png": np.clip(
            coast_penalty * (255.0 / max(float(np.max(coast_penalty)), 1e-6)),
            0,
            255,
        ).astype(np.uint8),
        "top_bottom_penalty.png": np.clip(
            top_bottom_penalty
            * (255.0 / max(float(np.max(top_bottom_penalty)), 1e-6)),
            0,
            255,
        ).astype(np.uint8),
        "effective_penalty.png": penalty_color,
    }
    for filename, image in debug_images.items():
        cv2.imwrite(str(debug_dir / filename), image)
    print(f"[DEBUG][COAST] 海岸线与惩罚热力图已保存至: {debug_dir}")


def apply_coast_penalty(
    raw_score_map: np.ndarray,
    grid: HeatmapGrid,
    config: BM40Config,
) -> Tuple[np.ndarray, float]:
    """施加海岸线及上下固定边缘惩罚，返回热力图和背景填充值。"""
    score_map = np.asarray(raw_score_map, dtype=np.float32).copy()
    valid_mask = build_valid_score_mask(grid, config, score_map)
    valid = valid_mask > 0
    if not np.any(valid):
        return score_map, float(config.heatmap_penalty_value)

    if config.Smear_type.upper() == "PB":
        coast_penalty_drop = config.coast_penalty_drop_pb
        top_bottom_margin = config.top_bottom_margin_pb
        top_bottom_penalty_drop = config.top_bottom_penalty_drop_pb
        coast_erode_ksize = config.coast_erode_ksize_pb
        coast_erode_iters = config.coast_erode_iters_pb
    else:
        coast_penalty_drop = config.coast_penalty_drop_bm
        top_bottom_margin = config.top_bottom_margin_bm
        top_bottom_penalty_drop = config.top_bottom_penalty_drop_bm
        coast_erode_ksize = config.coast_erode_ksize_bm
        coast_erode_iters = config.coast_erode_iters_bm

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, coast_erode_ksize
    )
    inland = cv2.erode(
        valid_mask,
        kernel,
        iterations=max(1, int(coast_erode_iters)),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    coast_band = cv2.bitwise_and(valid_mask, cv2.bitwise_not(inland))

    distance_to_inland = cv2.distanceTransform(
        cv2.bitwise_not(inland), cv2.DIST_L2, 3
    )
    distance_scale = max(float(config.coast_penalty_distance), 1e-6)
    coast_penalty = (
        float(coast_penalty_drop)
        * np.minimum(distance_to_inland / distance_scale, 1.0)
    ).astype(np.float32, copy=False)
    coast_penalty[~valid] = 0.0

    rows = score_map.shape[0]
    inward = np.minimum(np.arange(rows), np.arange(rows)[::-1]).astype(np.float32)
    margin = max(int(top_bottom_margin), 1)
    row_penalty = float(top_bottom_penalty_drop) * (
        1.0 - np.minimum(inward / float(margin), 1.0)
    )
    top_bottom_penalty = np.broadcast_to(row_penalty[:, None], score_map.shape)

    effective_penalty = np.maximum(coast_penalty, top_bottom_penalty)
    score_map[valid] -= effective_penalty[valid]

    background_floor = float(np.min(score_map[valid]) - 1e-6)
    score_map[~valid] = background_floor

    _save_coast_debug_images(
        raw_score_map=np.asarray(raw_score_map, dtype=np.float32),
        penalized_score_map=score_map,
        valid_mask=valid_mask,
        inland_mask=inland,
        coast_band=coast_band,
        coast_penalty=coast_penalty,
        top_bottom_penalty=top_bottom_penalty,
        effective_penalty=effective_penalty,
    )
    return score_map, background_floor


def _search_one_window_angle(
    args: Tuple[
        int,
        int,
        int,
        int,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Rect,
        int,
        float,
        int,
        int,
        float,
    ]
) -> Optional[Tuple[int, bool, SelectionResult]]:
    (
        order_idx,
        w,
        h,
        angle,
        adjusted_score_map,
        cell_matrix,
        user_search_mask,
        head_crop_rect,
        orientation,
        background_floor,
        kernel_margin,
        rows,
        cols,
    ) = args

    area_pixels = float(w * h)

    if angle == 0:
        # 轴对齐矩形等价于 boxFilter，OpenCV 对该路径有积分图级别优化。
        rect_pad_x, rect_pad_y = w // 2, h // 2
        rect_padded_scores = cv2.copyMakeBorder(
            adjusted_score_map,
            rect_pad_y,
            rect_pad_y,
            rect_pad_x,
            rect_pad_x,
            cv2.BORDER_CONSTANT,
            value=float(background_floor),
        )
        rect_padded_cells = cv2.copyMakeBorder(
            cell_matrix,
            rect_pad_y,
            rect_pad_y,
            rect_pad_x,
            rect_pad_x,
            cv2.BORDER_CONSTANT,
            value=0.0,
        )
        sum_scores_full = cv2.boxFilter(rect_padded_scores, -1, (w, h), normalize=False)
        sum_cells_full = cv2.boxFilter(rect_padded_cells, -1, (w, h), normalize=False)
        sum_scores = sum_scores_full[rect_pad_y : rect_pad_y + rows, rect_pad_x : rect_pad_x + cols]
        sum_cells = sum_cells_full[rect_pad_y : rect_pad_y + rows, rect_pad_x : rect_pad_x + cols]
    else:
        kernel_size = int(np.sqrt(w**2 + h**2)) + kernel_margin
        if kernel_size % 2 == 0:
            kernel_size += 1
        pad = kernel_size // 2

        padded_scores = cv2.copyMakeBorder(
            adjusted_score_map,
            pad,
            pad,
            pad,
            pad,
            cv2.BORDER_CONSTANT,
            value=float(background_floor),
        )
        padded_cells = cv2.copyMakeBorder(
            cell_matrix,
            pad,
            pad,
            pad,
            pad,
            cv2.BORDER_CONSTANT,
            value=0.0,
        )

        base_mask = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        pad_w, pad_h = (kernel_size - w) // 2, (kernel_size - h) // 2
        base_mask[pad_h : pad_h + h, pad_w : pad_w + w] = 1.0

        M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), -angle, 1.0)
        rotated_kernel = cv2.warpAffine(base_mask, M, (kernel_size, kernel_size), flags=cv2.INTER_NEAREST)

        sum_scores_full = cv2.filter2D(padded_scores, -1, rotated_kernel)
        sum_cells_full = cv2.filter2D(padded_cells, -1, rotated_kernel)
        sum_scores = sum_scores_full[pad : pad + rows, pad : pad + cols]
        sum_cells = sum_cells_full[pad : pad + rows, pad : pad + cols]

    if user_search_mask is not None:
        sum_scores = sum_scores.copy()
        sum_scores[user_search_mask == 0] = -1e12

    _, max_val, _, max_loc = cv2.minMaxLoc(sum_scores)
    cx, cy = int(max_loc[0]), int(max_loc[1])
    rect_points = cv2.boxPoints(((cx, cy), (w, h), float(angle)))

    if user_search_mask is not None:
        pts_idx = rect_points.astype(np.int32)
        for px, py in pts_idx:
            if not (0 <= px < cols and 0 <= py < rows) or user_search_mask[py, px] == 0:
                return None

    # 头尾判定：对旋转后略出网格边界的顶点做裁剪，避免如 x=-1 导致误判为尾部
    cls_x = np.clip(rect_points[:, 0], 0, cols - 1)
    extreme_x = float(np.max(cls_x) if orientation == 0 else np.min(cls_x))
    in_head = head_crop_rect.x <= extreme_x <= head_crop_rect.x2

    res = SelectionResult(
        area_score=max_val / area_pixels,
        cell_count=int(sum_cells[cy, cx]),
        angle=-angle,
        center_grid=(cx, cy),
        rect_size_grid=(w, h),
        vertices_grid=rect_points,
    )
    return order_idx, in_head, res

def find_candidate_regions(
    grid: HeatmapGrid,
    cell_matrix: np.ndarray,
    search_rects: List[Tuple[int, int]],
    head_crop_rect: Rect,
    config: BM40Config,
    user_search_mask: Optional[np.ndarray] = None
) -> Dict[str, List[SelectionResult]]:
    """
    在热力图网格上搜索候选区域，支持用户选区强制约束。
    """
    orientation = config.heatmap_orientation   # 0: 头部在右，1: 头部在左
    
    # A. 准备基础分值图
    avg_score_map = grid.finalize(fill_value=config.heatmap_penalty_value)
    rows, cols = avg_score_map.shape

    # B. 海岸线过滤：逐格惩罚写入热力图后再参与窗口卷积。
    if config.coast_penalty_enabled:
        adjusted_score_map, background_floor = apply_coast_penalty(
            avg_score_map, grid, config
        )
    else:
        adjusted_score_map = avg_score_map.copy()
        valid = grid.weights > 0
        background_floor = (
            float(np.min(adjusted_score_map[valid]) - 1e-6)
            if np.any(valid)
            else float(config.heatmap_penalty_value)
        )
        adjusted_score_map[~valid] = background_floor

    # 如果有用户选区，初步压低底图分数
    if user_search_mask is not None:
        adjusted_score_map[user_search_mask == 0] = config.heatmap_penalty_value * 100

    head_results: List[SelectionResult] = []
    tail_results: List[SelectionResult] = []

    tasks = [
        (
            order_idx,
            w,
            h,
            angle,
            adjusted_score_map,
            cell_matrix,
            user_search_mask,
            head_crop_rect,
            orientation,
            background_floor,
            config.kernel_margin,
            rows,
            cols,
        )
        for order_idx, (w, h, angle) in enumerate(
            (w, h, angle) for (w, h) in search_rects for angle in config.get_angles()
        )
    ]

    cpu_count = os.cpu_count() or 1
    max_workers = min(len(tasks), max(1, min(8, cpu_count)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        ordered_results = list(executor.map(_search_one_window_angle, tasks))

    for item in ordered_results:
        if item is None:
            continue
        _, in_head, res = item
        if in_head:
            head_results.append(res)
        else:
            tail_results.append(res)

    return {"head_results": head_results, "tail_results": tail_results}



def filter_candidates(
    results: Dict[str, List[SelectionResult]], 
    config: BM40Config,
    all_cell_count: int
) -> List[SelectionResult]:
    """
    结果过滤
    根据细胞数量目标筛选候选区域。
    """
    # 选区目标：基础目标 * 冗余系数
    target_num = config.target_cell_num_WBC * config.target_ratio
    
    # 优先看尾部结果（体尾交界），没有则看头部
    candidates = results.get("tail_results", []) or results.get("head_results", [])
    if not candidates:
        print("警告：没有可用的候选选区，返回空列表。")
        return []

    select_ratio = config.select_ratio_init
    selected = []
    
    while not selected:
        # 定义当前的搜索上限
        upper_limit = target_num * (1 + select_ratio)
        
        # 寻找细胞数落在 [target_num, upper_limit] 的区域
        selected = [item for item in candidates if target_num <= item.cell_count <= upper_limit]
        
        if selected:
            break
            
        # 如果当前上限已经超过了全图总细胞数，说明再扩大比例也没有意义了
        if upper_limit > all_cell_count:
            print(f"提示：搜索上限({upper_limit:.0f})已超过全图总数({all_cell_count})，返回细胞数最多的选区。")
            # 此时返回细胞数量最接近全图（即最大）的选区
            return [max(candidates, key=lambda x: x.cell_count)]
            
        # 找不到则扩大步进比例
        select_ratio *= 2
            
    return selected


def prepare_uniformity_map(cell_matrix: np.ndarray, config: BM40Config) -> np.ndarray:
    """
    预计算全局均匀性底图（覆盖率图）。
    """
    # 只要格子里有细胞，就认为该空间被占用
    presence_map = (cell_matrix > 0).astype(np.float32)

    # 计算百倍视野对应的网格尺寸
    kw = max(1, int(config.x100_rect_width // config.cell_size))
    kh = max(1, int(config.x100_rect_height // config.cell_size))

    # 均值滤波：计算窗口内的“占用网格占比”
    return cv2.boxFilter(presence_map, -1, (kw, kh), normalize=True)


def get_shape_priority(result: SelectionResult, config: BM40Config) -> int:
    """0=横向矩形，1=近正方形，2=纵向矩形。"""
    width, height = result.rect_size_grid
    short_side = min(width, height)
    long_side = max(width, height)
    if short_side <= 0:
        return 2
    if long_side / short_side <= float(config.near_square_aspect_ratio):
        return 1
    return 0 if width > height else 2


def _score_close_tolerance(score_range: float, config: BM40Config) -> float:
    return max(0.0, float(score_range)) * float(config.shape_score_close_ratio)


def _order_candidates_by_score(
    candidates: List[SelectionResult],
) -> List[SelectionResult]:
    return sorted(candidates, key=lambda item: item.area_score, reverse=True)


def _split_score_close_prefix(
    ordered: List[SelectionResult],
    tolerance: float,
) -> Tuple[List[SelectionResult], List[SelectionResult]]:
    """按与最高分的差值切分：前缀为接近分候选，后缀为明显低分候选。"""
    if not ordered:
        return [], []

    best_score = ordered[0].area_score
    close_count = 1
    while (
        close_count < len(ordered)
        and best_score - ordered[close_count].area_score <= tolerance
    ):
        close_count += 1
    return ordered[:close_count], ordered[close_count:]


def sort_candidates_by_score_and_shape(
    candidates: List[SelectionResult],
    config: BM40Config,
    score_range: float,
) -> List[SelectionResult]:
    """
    先按分值降序；仅在“接近分”前缀内按形状重排（横向 > 近正方 > 纵向），
    明显低分候选保持原分数序且不参与形状重排。
    """
    ordered = _order_candidates_by_score(candidates)
    if len(ordered) <= 1:
        return ordered

    close_prefix, rest = _split_score_close_prefix(
        ordered, _score_close_tolerance(score_range, config)
    )
    close_prefix = sorted(
        close_prefix,
        key=lambda item: (get_shape_priority(item, config), -item.area_score),
    )
    return close_prefix + rest


def _uniformity_score(
    result: SelectionResult,
    u_map: np.ndarray,
) -> float:
    mask = np.zeros(u_map.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [result.vertices_grid.astype(np.int32)], 1)
    if not np.any(mask):
        return -1.0
    return float(np.mean(u_map[mask > 0]))


def select_best_uniform_region(
    selected_results: List[SelectionResult],
    cell_matrix: np.ndarray,
    config: BM40Config,
    score_range: float = 0.0,
) -> SelectionResult:
    """
    从候选选区中选出最佳。

    决策链（仅对 filter_candidates 已筛出的候选）：
    1. 按 area_score 降序；
    2. 与最高分差 <= score_range * shape_score_close_ratio 的进入“接近分”子集；
    3. 子集只有 1 个 → 直接返回（最高分胜出）；
    4. 子集多个 → 比均匀性 u_score（选区内百倍窗细胞占用率均值）；
    5. u_score 相同 → 形状优先（横向 > 近正方 > 纵向）；
    6. 仍相同 → area_score 更高者。
    """
    if not selected_results:
        raise ValueError("根据细胞数量目标筛选候选区域为空。")
    if len(selected_results) == 1:
        return selected_results[0]

    ordered = _order_candidates_by_score(selected_results)
    close_group, _ = _split_score_close_prefix(
        ordered, _score_close_tolerance(score_range, config)
    )
    if len(close_group) == 1:
        return close_group[0]

    u_map = prepare_uniformity_map(cell_matrix, config)
    return min(
        close_group,
        key=lambda res: (
            -_uniformity_score(res, u_map),
            get_shape_priority(res, config),
            -res.area_score,
        ),
    )