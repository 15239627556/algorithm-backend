from typing import Dict, List

import numpy as np
import torch
from torchvision.ops import nms as torch_nms

from project.cells import Cell
from project.tiles import Tile


def compute_effective_scores(
    boxes_t: torch.Tensor,
    scores_t_opt: torch.Tensor | None,
    tile_rect_xyxy: List[float],
) -> torch.Tensor:
    """提取到外部，用于在预处理阶段计算一次 effective scores"""
    if boxes_t.numel() == 0:
        return boxes_t.new_zeros((0,), dtype=torch.float32)

    # 1. 计算框的宽高
    wh = (boxes_t[:, 2:4] - boxes_t[:, 0:2]).clamp_min(0.0)
    wh_w, wh_h = wh[:, 0], wh[:, 1]

    # 2. 面积归一化
    area = wh_w * wh_h
    tile_w = max(tile_rect_xyxy[2] - tile_rect_xyxy[0], 1.0)
    tile_h = max(tile_rect_xyxy[3] - tile_rect_xyxy[1], 1.0)
    area_norm = area / (tile_w * tile_h)

    # 3. 动态计算 tau_pixels
    box_short_side = torch.minimum(wh_w, wh_h)
    dynamic_tau = (0.5 * box_short_side).clamp(min=30.0, max=100.0)

    # 4. 到边界距离
    dist_left = (boxes_t[:, 0] - tile_rect_xyxy[0]).clamp_min(0.0)
    dist_top = (boxes_t[:, 1] - tile_rect_xyxy[1]).clamp_min(0.0)
    dist_right = (tile_rect_xyxy[2] - boxes_t[:, 2]).clamp_min(0.0)
    dist_bottom = (tile_rect_xyxy[3] - boxes_t[:, 3]).clamp_min(0.0)

    dist_min = torch.minimum(
        torch.minimum(dist_left, dist_right),
        torch.minimum(dist_top, dist_bottom),
    )

    # 5. 完整度因子 [0, 1]
    completeness_norm = (dist_min / dynamic_tau).clamp(0.0, 1.0)

    # 6. 最终得分融合
    if scores_t_opt is None:
        return area_norm.to(torch.float32)

    return scores_t_opt.to(torch.float32) * (0.5 + 0.5 * completeness_norm) + area_norm.to(torch.float32)


def _pairwise_ios(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """计算 IoS 矩阵，shape=(na, nb)。IoS = inter / min(area_a, area_b)。"""
    aw = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0])
    ah = np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    bw = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0])
    bh = np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])
    area_a = aw * ah
    area_b = bw * bh

    xx1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    yy1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    xx2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    yy2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    return inter / np.maximum(np.minimum(area_a[:, None], area_b[None, :]), 1e-6)


def _cross_tile_ios_filter(
    boxes_np: np.ndarray,
    scores_np: np.ndarray,
    keep_idx: np.ndarray,
    num_left: int,
    ios_thresh: float,
    iou_thresh: float,
) -> np.ndarray:
    """
    在 IoU-NMS 存活框上，仅对跨 tile 对做一次向量化 IoS 抑制。
    对每个 IoS >= 阈值的左右框对，丢弃分数较低者。
    """
    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    left = keep_idx[keep_idx < num_left]
    right = keep_idx[keep_idx >= num_left]
    if left.size == 0 or right.size == 0:
        return keep_idx

    bi = boxes_np[left]
    bj = boxes_np[right]
    area_i = np.maximum(0.0, bi[:, 2] - bi[:, 0]) * np.maximum(0.0, bi[:, 3] - bi[:, 1])
    area_j = np.maximum(0.0, bj[:, 2] - bj[:, 0]) * np.maximum(0.0, bj[:, 3] - bj[:, 1])
    area_min = float(min(area_i.min(), area_j.min()))
    area_max = float(max(area_i.max(), area_j.max()))
    # 面积足够接近时，高 IoS 会被 IoU-NMS 覆盖
    if area_max <= 0.0 or (area_min / area_max) >= float(iou_thresh):
        return keep_idx

    ios = _pairwise_ios(bi, bj)
    a_idx, b_idx = np.nonzero(ios >= float(ios_thresh))
    if a_idx.size == 0:
        return keep_idx

    si = scores_np[left]
    sj = scores_np[right]
    left_wins = si[a_idx] >= sj[b_idx]
    drop_left = np.zeros(left.size, dtype=bool)
    drop_right = np.zeros(right.size, dtype=bool)
    drop_right[b_idx[left_wins]] = True
    drop_left[a_idx[~left_wins]] = True
    return np.concatenate([left[~drop_left], right[~drop_right]], axis=0)


def nms_iou_then_ios(
    boxes_t: torch.Tensor,
    scores_t: torch.Tensor,
    boxes_np: np.ndarray,
    scores_np: np.ndarray,
    num_left: int,
    iou_thresh: float,
    ios_thresh: float,
) -> np.ndarray:
    """
    快速两阶段：
    1) torchvision IoU-NMS（C++/CUDA，保持原速度主体）
    2) 跨 tile 向量化 IoS 过滤（无 Python 贪心循环）
    """
    if boxes_t.numel() == 0:
        return np.empty((0,), dtype=np.int64)

    keep_t = torch_nms(boxes_t, scores_t, float(iou_thresh))
    keep_np = keep_t.detach().cpu().numpy().astype(np.int64, copy=False)
    if ios_thresh <= 0 or keep_np.size <= 1 or num_left <= 0:
        return keep_np

    return _cross_tile_ios_filter(
        boxes_np=boxes_np,
        scores_np=scores_np,
        keep_idx=keep_np,
        num_left=num_left,
        ios_thresh=float(ios_thresh),
        iou_thresh=float(iou_thresh),
    )


def dedup_cells_across_tiles(
        tiles_40x: List[Tile],
        tile_w: int = 2448,
        tile_h: int = 2048,
        iou_thresh: float = 0.2,
        ios_thresh: float = 0.5,
) -> List[Tile]:
    """
    在相邻 40x tile 的重叠带内做 NMS 去重（先 IoU，再跨 tile IoS）。
    - 输入：List[Tile]，每个 tile 的 cells 列表中的细胞坐标为局部坐标（相对于瓦片）
    - 相邻关系：右、下、右下、左下
    - NMS 分数：使用 Cell.class_confidence，并结合“离 tile 边界的完整度”与“面积归一化”联合打分；
      若该候选缺分数，则仅用面积归一化兜底
    - iou_thresh: 交并比阈值（IoU = 交集 / 并集）
    - ios_thresh: 交集/较小框面积阈值（IoS = 交集 / min(两框面积)）；<=0 时关闭 IoS
    - 输出：List[Tile]，去重后的 tiles（cells 坐标仍为局部坐标，与输入一致）
    """
    # 重叠带单次 NMS 规模通常不大；IoS 在 numpy/CPU 上做。
    # 若走 CUDA，每对 neighbor 都要 .cpu()，同步开销远大于 NMS 收益（实测更慢）。
    device = torch.device("cpu")

    tiles_data = []

    # ---------- 1. 预处理：向量化提取与预计算 ----------
    for idx, tile in enumerate(tiles_40x):
        fx = int(tile.meta.get("col_index", 0))
        fy = int(tile.meta.get("row_index", 0))
        ax = int(tile.x) if tile.x is not None else 0
        ay = int(tile.y) if tile.y is not None else 0

        tw = tile.w if tile.w is not None else tile_w
        th = tile.h if tile.h is not None else tile_h
        image_rect = [ax, ay, ax + int(tw), ay + int(th)]

        cells = tile.cells or []
        num_cells = len(cells)

        if num_cells == 0:
            rects_xyxy = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
            rects_t = torch.empty((0, 4), dtype=torch.float32, device=device)
            eff_scores_t = torch.empty((0,), dtype=torch.float32, device=device)
            eff_scores_np = np.empty((0,), dtype=np.float32)
        else:
            # 向量化获取坐标与分数
            rects_list = [[c.cell_xmin, c.cell_ymin, c.cell_xmax, c.cell_ymax] for c in cells]
            scores_list = [
                float(c.class_confidence) if c.class_confidence is not None else 0.0
                for c in cells
            ]

            rects_xyxy = np.array(rects_list, dtype=np.float32)
            # 矩阵加法一次性转换为全局坐标
            rects_xyxy[:, [0, 2]] += ax
            rects_xyxy[:, [1, 3]] += ay
            scores = np.array(scores_list, dtype=np.float32)

            # 转为 tensor (放到指定设备上)
            rects_t = torch.as_tensor(rects_xyxy, dtype=torch.float32, device=device)
            scores_t = torch.as_tensor(scores, dtype=torch.float32, device=device)

            # 【核心优化】：在此处一次性预计算 effective_scores，而不是在双循环里
            eff_scores_t = compute_effective_scores(rects_t, scores_t, image_rect)
            eff_scores_np = eff_scores_t.detach().cpu().numpy()

        tiles_data.append({
            "orig_idx": idx,
            "tile": tile,
            "rowID": fy,
            "colID": fx,
            "imageRect_xyxy": np.array(image_rect, dtype=np.int32),
            "rects_np": rects_xyxy,
            "rects_t": rects_t,             # 用于 NMS (GPU/CPU)
            "eff_scores_t": eff_scores_t,   # 已预计算好的融合打分
            "eff_scores_np": eff_scores_np,  # IoS 过滤用，避免循环内反复 .cpu()
        })

    # 排序 & 建立空间索引
    tiles_sorted = sorted(tiles_data, key=lambda d: (d["rowID"], d["colID"]))
    rowcol_to_index = {(t["rowID"], t["colID"]): i for i, t in enumerate(tiles_sorted)}
    to_delete = [np.zeros(t["rects_np"].shape[0], dtype=bool) for t in tiles_sorted]

    # ---------- 2. 核心 NMS 循环 ----------
    neighbors = [(0, 1), (-1, 0), (-1, 1), (-1, -1)]

    for i, ti in enumerate(tiles_sorted):
        r, c = ti["rowID"], ti["colID"]
        for dr, dc in neighbors:
            j = rowcol_to_index.get((r + dr, c + dc))
            if j is None:
                continue

            tj = tiles_sorted[j]
            A, B = ti["imageRect_xyxy"], tj["imageRect_xyxy"]

            # 计算重叠框
            ix1, iy1 = max(A[0], B[0]), max(A[1], B[1])
            ix2, iy2 = min(A[2], B[2]), min(A[3], B[3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            rects_i, rects_j = ti["rects_np"], tj["rects_np"]
            if rects_i.size == 0 and rects_j.size == 0:
                continue

            # 利用 numpy 极速筛选交集
            mask_i = (
                (rects_i[:, 0] < ix2)
                & (rects_i[:, 2] > ix1)
                & (rects_i[:, 1] < iy2)
                & (rects_i[:, 3] > iy1)
            )
            mask_j = (
                (rects_j[:, 0] < ix2)
                & (rects_j[:, 2] > ix1)
                & (rects_j[:, 1] < iy2)
                & (rects_j[:, 3] > iy1)
            )

            if not (mask_i.any() or mask_j.any()):
                continue

            idx_i_np = np.nonzero(mask_i)[0]
            idx_j_np = np.nonzero(mask_j)[0]
            num_left, num_right = len(idx_i_np), len(idx_j_np)
            if num_left + num_right <= 1:
                continue

            # 取出参与 NMS 的 Tensor (已在 Device 上)
            idx_i_t = torch.tensor(idx_i_np, dtype=torch.long, device=device)
            idx_j_t = torch.tensor(idx_j_np, dtype=torch.long, device=device)

            boxes_i_t = ti["rects_t"].index_select(0, idx_i_t)
            boxes_j_t = tj["rects_t"].index_select(0, idx_j_t)
            # 直接使用预计算的 scores
            scores_i_t = ti["eff_scores_t"].index_select(0, idx_i_t)
            scores_j_t = tj["eff_scores_t"].index_select(0, idx_j_t)

            boxes_t = torch.cat([boxes_i_t, boxes_j_t], dim=0)
            scores_t = torch.cat([scores_i_t, scores_j_t], dim=0)
            boxes_np = np.concatenate(
                [rects_i[idx_i_np], rects_j[idx_j_np]], axis=0
            )
            scores_np = np.concatenate(
                [ti["eff_scores_np"][idx_i_np], tj["eff_scores_np"][idx_j_np]],
                axis=0,
            )

            # 执行快速两阶段 NMS：IoU (torchvision) + 跨 tile IoS (向量化)
            keep_np = nms_iou_then_ios(
                boxes_t=boxes_t,
                scores_t=scores_t,
                boxes_np=boxes_np,
                scores_np=scores_np,
                num_left=num_left,
                iou_thresh=float(iou_thresh),
                ios_thresh=float(ios_thresh),
            )

            kept_mask_np = np.zeros(num_left + num_right, dtype=bool)
            kept_mask_np[keep_np] = True

            if num_left > 0:
                drop_i = ~kept_mask_np[:num_left]
                if drop_i.any():
                    to_delete[i][idx_i_np[drop_i]] = True
            if num_right > 0:
                drop_j = ~kept_mask_np[num_left:]
                if drop_j.any():
                    to_delete[j][idx_j_np[drop_j]] = True

    # ---------- 3. 后处理：极速写回机制 ----------
    for sorted_i, t_data in enumerate(tiles_sorted):
        tile = t_data["tile"]

        if not tile.cells:
            continue

        keep_mask = ~to_delete[sorted_i]
        kept_indices = np.nonzero(keep_mask)[0]

        # 【核心优化】：直接复用原始的 Cell 对象，避免转换坐标和重新实例化！
        kept_cells = [tile.cells[idx] for idx in kept_indices]

        # 兜底情况：如果原来就没有 score，则按面积写回
        if len(kept_cells) > 0 and kept_cells[0].class_confidence is None:
            r_kept = t_data["rects_np"][keep_mask]
            w = np.clip(r_kept[:, 2] - r_kept[:, 0], 0.0, None)
            h = np.clip(r_kept[:, 3] - r_kept[:, 1], 0.0, None)
            fallback_scores = w * h
            for c_obj, f_score in zip(kept_cells, fallback_scores):
                c_obj.class_confidence = float(f_score)

        tile.cells = kept_cells

    return tiles_40x


def dedup_cells_across_tiles_per_type(
        tiles_40x: List[Tile],
        tile_w: int = 2448,
        tile_h: int = 2048,
        iou_thresh: float = 0.2,
        ios_thresh: float = 0.5,
        cell_types: List[int] | None = None,
) -> List[Tile]:
    """
    按 cell_type 分组分别去重，再将结果写回原始 tile。

    适用场景：
    - 不同细胞类型之间不应互相抑制；
    - 希望对若干类型分别做跨 tile 去重，并保留其他类型细胞。

    参数：
    - cell_types=None: 自动扫描 tiles 中出现过的全部 cell_type，并逐类去重。
    - cell_types=[...]: 只对指定类型去重；未指定的类型原样保留。
    - ios_thresh: 传给 dedup_cells_across_tiles；<=0 时关闭 IoS。
    """
    if not tiles_40x:
        return tiles_40x

    original_cells_by_uid: Dict[str, List[Cell]] = {
        tile.image_uid: list(tile.cells or []) for tile in tiles_40x
    }

    if cell_types is None:
        seen_types = set()
        ordered_types: List[int] = []
        for tile in tiles_40x:
            for cell in tile.cells or []:
                cell_type = cell.cell_type
                if cell_type in seen_types:
                    continue
                seen_types.add(cell_type)
                ordered_types.append(cell_type)
        target_cell_types = ordered_types
    else:
        # 去重并保留用户传入顺序
        target_cell_types = list(dict.fromkeys(cell_types))

    if not target_cell_types:
        return tiles_40x

    # 逐 tile 收集各类型去重后的结果，最后统一写回，避免中间态互相污染。
    deduped_cells_by_uid: Dict[str, List[Cell]] = {
        tile.image_uid: [] for tile in tiles_40x
    }

    for cell_type in target_cell_types:
        # 仅保留当前 cell_type，复用已有去重逻辑。
        for tile in tiles_40x:
            original_cells = original_cells_by_uid.get(tile.image_uid, [])
            tile.cells = [c for c in original_cells if c.cell_type == cell_type]

        dedup_cells_across_tiles(
            tiles_40x=tiles_40x,
            tile_w=tile_w,
            tile_h=tile_h,
            iou_thresh=iou_thresh,
            ios_thresh=ios_thresh,
        )

        for tile in tiles_40x:
            if tile.cells:
                deduped_cells_by_uid[tile.image_uid].extend(tile.cells)

    target_type_set = set(target_cell_types)
    for tile in tiles_40x:
        original_cells = original_cells_by_uid.get(tile.image_uid, [])

        # 按原始 tile.cells 的出现顺序合并，尽量保持每类细胞的相对顺序稳定。
        merged_cells: List[Cell] = []
        pending_by_type: Dict[int, List[Cell]] = {}
        for cell in deduped_cells_by_uid[tile.image_uid]:
            pending_by_type.setdefault(cell.cell_type, []).append(cell)

        for cell in original_cells:
            if cell.cell_type in target_type_set:
                bucket = pending_by_type.get(cell.cell_type)
                if bucket:
                    merged_cells.append(bucket.pop(0))
            else:
                merged_cells.append(cell)

        # 兜底：如果某些类型的保留框数量比原始更多，补到末尾。
        for cells_left in pending_by_type.values():
            if cells_left:
                merged_cells.extend(cells_left)

        tile.cells = merged_cells

    return tiles_40x
