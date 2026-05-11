from typing import List

import numpy as np
import torch
from torchvision.ops import nms as torch_nms
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
        torch.minimum(dist_top, dist_bottom)
    )

    # 5. 完整度因子 [0, 1]
    completeness_norm = (dist_min / dynamic_tau).clamp(0.0, 1.0)

    # 6. 最终得分融合
    if scores_t_opt is None:
        return area_norm.to(torch.float32)

    return scores_t_opt.to(torch.float32) * (0.5 + 0.5 * completeness_norm) + area_norm.to(torch.float32)


def dedup_cells_across_tiles(
        tiles_40x: List[Tile],
        tile_w: int = 2448,
        tile_h: int = 2048,
        iou_thresh: float = 0.2,
) -> List[Tile]:
    
    # 检测 GPU 以加速 NMS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tiles_data = []
    
    # ---------- 1. 预处理：向量化提取与预计算 ----------
    for idx, tile in enumerate(tiles_40x):
        fx = int(tile.meta.get('col_index', 0))
        fy = int(tile.meta.get('row_index', 0))
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
        else:
            # 向量化获取坐标与分数 (列表推导式提取属性最快)
            rects_list = [[c.cell_xmin, c.cell_ymin, c.cell_xmax, c.cell_ymax] for c in cells]
            scores_list = [c.class_confidence for c in cells]
            
            rects_xyxy = np.array(rects_list, dtype=np.float32)
            # 矩阵加法一次性转换为全局坐标
            rects_xyxy[:, [0, 2]] += ax
            rects_xyxy[:, [1, 3]] += ay
            scores = np.array(scores_list, dtype=np.float32)
            
            # 转为 tensor (放到指定设备上)
            rects_t = torch.as_tensor(rects_xyxy, dtype=torch.float32, device=device)
            scores_t = torch.as_tensor(scores, dtype=torch.float32, device=device) if scores.size > 0 else None
            
            # 【核心优化】：在此处一次性预计算 effective_scores，而不是在双循环里
            eff_scores_t = compute_effective_scores(rects_t, scores_t, image_rect)

        tiles_data.append({
            "orig_idx": idx,
            "tile": tile,
            "rowID": fy,
            "colID": fx,
            "imageRect_xyxy": np.array(image_rect, dtype=np.int32),
            "rects_np": rects_xyxy, 
            "rects_t": rects_t,             # 用于 NMS (GPU/CPU)
            "eff_scores_t": eff_scores_t,   # 已预计算好的融合打分
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
            if j is None: continue
            
            tj = tiles_sorted[j]
            A, B = ti["imageRect_xyxy"], tj["imageRect_xyxy"]
            
            # 计算重叠框
            ix1, iy1 = max(A[0], B[0]), max(A[1], B[1])
            ix2, iy2 = min(A[2], B[2]), min(A[3], B[3])
            if ix2 <= ix1 or iy2 <= iy1: continue
            
            rects_i, rects_j = ti["rects_np"], tj["rects_np"]
            if rects_i.size == 0 and rects_j.size == 0: continue

            # 利用 numpy 极速筛选交集
            mask_i = (rects_i[:, 0] < ix2) & (rects_i[:, 2] > ix1) & (rects_i[:, 1] < iy2) & (rects_i[:, 3] > iy1)
            mask_j = (rects_j[:, 0] < ix2) & (rects_j[:, 2] > ix1) & (rects_j[:, 1] < iy2) & (rects_j[:, 3] > iy1)
            
            if not (mask_i.any() or mask_j.any()): continue

            idx_i_np = np.nonzero(mask_i)[0]
            idx_j_np = np.nonzero(mask_j)[0]
            num_left, num_right = len(idx_i_np), len(idx_j_np)
            if num_left + num_right <= 1: continue

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

            # 执行 NMS
            keep = torch_nms(boxes_t, scores_t, iou_thresh)
            
            # 将 keep 掩码转回 CPU 计算丢弃项
            kept_mask = torch.zeros(num_left + num_right, dtype=torch.bool, device=device)
            kept_mask[keep] = True
            kept_mask_np = kept_mask.cpu().numpy()

            if num_left > 0:
                drop_i = ~kept_mask_np[:num_left]
                if drop_i.any(): to_delete[i][idx_i_np[drop_i]] = True
            if num_right > 0:
                drop_j = ~kept_mask_np[num_left:]
                if drop_j.any(): to_delete[j][idx_j_np[drop_j]] = True

    # ---------- 3. 后处理：极速写回机制 ----------
    for sorted_i, t_data in enumerate(tiles_sorted):
        tile = t_data["tile"]
        
        if not tile.cells:
            continue

        keep_mask = ~to_delete[sorted_i]
        kept_indices = np.nonzero(keep_mask)[0]

        # 【核心优化】：直接复用原始的 Cell 对象，避免转换坐标和重新实例化！
        kept_cells = [tile.cells[idx] for idx in kept_indices]
        
        # 兜底情况：如果原来就没有 score，但用户依然希望按“面积打分”进行结果写回
        # (通常不需要，因为 NMS 用于剔除，不会修改原本就存在框的属性，这里仅针对你的 fallback 逻辑)
        if len(kept_cells) > 0 and kept_cells[0].class_confidence is None:
            r_kept = t_data["rects_np"][keep_mask]
            w = np.clip(r_kept[:, 2] - r_kept[:, 0], 0.0, None)
            h = np.clip(r_kept[:, 3] - r_kept[:, 1], 0.0, None)
            fallback_scores = w * h
            for c_obj, f_score in zip(kept_cells, fallback_scores):
                c_obj.class_confidence = float(f_score)

        tile.cells = kept_cells

    return tiles_40x