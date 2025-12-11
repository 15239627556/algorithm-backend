from typing import Any, Dict, List
import numpy as np
import torch
from torchvision.ops import nms as torch_nms

def dedup_cells_across_tiles(
    infos_40xtile: List[Dict[str, Any]],
    tile_w: int = 2448,
    tile_h: int = 2048,
    iou_thresh: float = 0.2,
):
    """
    在相邻 40x tile 的重叠带内做 NMS 去重（torchvision.ops.nms）。
    - 输入：每个 tile 的 t['global_cell_rects'] = [[x1,y1,x2,y2,score], ...]
    - 相邻关系：右、下、右下、左下
    - NMS 分数：优先使用提供的 score（第 5 列）；若缺，则在该对候选上临时面积兜底
    - 输出：t['global_cell_rects_dedup'] = [[x1,y1,x2,y2,score], ...]
    """

    # ---------- 预处理：float32，避免循环内 dtype 转换 ----------
    tiles = []
    for idx, t in enumerate(infos_40xtile):
        fx = int(t.get("index_40xtile_x", 0))
        fy = int(t.get("index_40xtile_y", 0))
        ax = int(t.get("abs_40xtile_x", 0))
        ay = int(t.get("abs_40xtile_y", 0))

        raw = t.get("global_cell_rects") or []
        if len(raw) == 0:
            rects_xyxy = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
        else:
            a = np.asarray(raw, dtype=np.float32)
            if a.ndim == 1:
                a = a.reshape(1, -1)
            if a.shape[1] >= 5:
                rects_xyxy = np.ascontiguousarray(a[:, :4].astype(np.float32, copy=False))
                scores     = np.ascontiguousarray(a[:, 4].astype(np.float32, copy=False))
            elif a.shape[1] >= 4:
                rects_xyxy = np.ascontiguousarray(a[:, :4].astype(np.float32, copy=False))
                scores     = np.empty((rects_xyxy.shape[0],), dtype=np.float32)  # 缺 score，后续临时兜底
            else:
                rects_xyxy = np.empty((0, 4), dtype=np.float32)
                scores     = np.empty((0,), dtype=np.float32)

        tiles.append({
            "orig_idx": idx,
            "rowID": fy,
            "colID": fx,
            "imageRect_xyxy": np.array([ax, ay, ax + int(tile_w), ay + int(tile_h)], dtype=np.int32),

            # numpy 缓存
            "rects_np": rects_xyxy,          # (N,4) float32
            "scores_np": scores,              # (N,)  float32（可能是空数组或长度为0表示缺 score）

            # 预构造 torch 张量（CPU），循环内只索引
            "rects_t": torch.from_numpy(rects_xyxy),               # (N,4) float32
            "scores_t": torch.from_numpy(scores) if scores.size > 0 else None,  # (N,) 或 None
        })

    # 排序 & 定位映射
    tiles_sorted = sorted(tiles, key=lambda d: (d["rowID"], d["colID"]))
    rowcol_to_index = {(t["rowID"], t["colID"]): i for i, t in enumerate(tiles_sorted)}

    # 每 tile 的删除掩码
    to_delete = [np.zeros(t["rects_np"].shape[0], dtype=bool) for t in tiles_sorted]

    # 相邻（右、下、右下、左下）
    neighbors = [(0, +1), (-1, 0), (-1, +1), (-1, -1)]

    # 与 ROI 相交的布尔掩码
    def mask_intersect_rects(rects_xyxy: np.ndarray, roi_xyxy: np.ndarray) -> np.ndarray:
        if rects_xyxy.size == 0:
            return np.zeros((0,), dtype=bool)
        rx1, ry1, rx2, ry2 = roi_xyxy
        x1 = rects_xyxy[:, 0]; y1 = rects_xyxy[:, 1]
        x2 = rects_xyxy[:, 2]; y2 = rects_xyxy[:, 3]
        return (x1 < rx2) & (x2 > rx1) & (y1 < ry2) & (y2 > ry1)

    # 主循环：tile 对级别（候选在 torch 侧索引，NMS 用 score）
    for i, ti in enumerate(tiles_sorted):
        r, c = ti["rowID"], ti["colID"]
        for dr, dc in neighbors:
            j = rowcol_to_index.get((r + dr, c + dc))
            if j is None:
                continue
            tj = tiles_sorted[j]

            # tile 的重叠带
            A = ti["imageRect_xyxy"]; B = tj["imageRect_xyxy"]
            ix1, iy1 = max(A[0], B[0]), max(A[1], B[1])
            ix2, iy2 = min(A[2], B[2]), min(A[3], B[3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            overlap_xyxy = np.array([ix1, iy1, ix2, iy2], dtype=np.int32)

            rects_i = ti["rects_np"]; rects_j = tj["rects_np"]
            if rects_i.shape[0] == 0 and rects_j.shape[0] == 0:
                continue

            mask_i = mask_intersect_rects(rects_i, overlap_xyxy)
            mask_j = mask_intersect_rects(rects_j, overlap_xyxy)
            if not (mask_i.any() or mask_j.any()):
                continue

            idx_i_np = np.nonzero(mask_i)[0]
            idx_j_np = np.nonzero(mask_j)[0]
            num_left  = idx_i_np.shape[0]
            num_right = idx_j_np.shape[0]
            total = num_left + num_right
            if total <= 1:
                continue  # 无需 NMS

            idx_i_t = torch.from_numpy(idx_i_np.astype(np.int64))
            idx_j_t = torch.from_numpy(idx_j_np.astype(np.int64))

            boxes_i_t = ti["rects_t"].index_select(0, idx_i_t)
            boxes_j_t = tj["rects_t"].index_select(0, idx_j_t)

            # 分数优先用原 score；若该 tile 缺 score，则对该对候选临时按面积兜底（只算子集，开销很小）
            if ti["scores_t"] is not None:
                scores_i_t = ti["scores_t"].index_select(0, idx_i_t)
            else:
                # (x2-x1)*(y2-y1)
                wh_i = (boxes_i_t[:, 2] - boxes_i_t[:, 0]).clamp_min_(0) * (boxes_i_t[:, 3] - boxes_i_t[:, 1]).clamp_min_(0)
                scores_i_t = wh_i

            if tj["scores_t"] is not None:
                scores_j_t = tj["scores_t"].index_select(0, idx_j_t)
            else:
                wh_j = (boxes_j_t[:, 2] - boxes_j_t[:, 0]).clamp_min_(0) * (boxes_j_t[:, 3] - boxes_j_t[:, 1]).clamp_min_(0)
                scores_j_t = wh_j

            boxes_t  = torch.cat([boxes_i_t,  boxes_j_t],  dim=0) if num_left and num_right else \
                       (boxes_i_t if num_left else boxes_j_t)
            scores_t = torch.cat([scores_i_t, scores_j_t], dim=0) if num_left and num_right else \
                       (scores_i_t if num_left else scores_j_t)

            keep = torch_nms(boxes_t, scores_t, float(iou_thresh))
            kept_mask = torch.zeros((total,), dtype=torch.bool)
            kept_mask[keep] = True
            kept_mask_np = kept_mask.numpy()

            # 前半段 i，后半段 j
            if num_left > 0:
                drop_i = ~kept_mask_np[:num_left]
                if np.any(drop_i):
                    to_delete[i][idx_i_np[drop_i]] = True
            if num_right > 0:
                drop_j = ~kept_mask_np[num_left:]
                if np.any(drop_j):
                    to_delete[j][idx_j_np[drop_j]] = True

    # ---------- 写回：global_cell_rects_dedup ----------
    for sorted_i, t in enumerate(tiles_sorted):
        orig_i = t["orig_idx"]
        rects  = t["rects_np"]          # float32
        scores = t["scores_np"]         # float32
        if rects.shape[0] == 0:
            kept_rects = []
        else:
            keep_mask = ~to_delete[sorted_i]
            r_kept = rects[keep_mask]   # (M,4) float32
            if scores.shape[0] == rects.shape[0]:
                s_kept = scores[keep_mask]  # (M,) float32
            else:
                # 兜底：面积作为 score（float32）
                w = np.clip(r_kept[:, 2] - r_kept[:, 0], 0.0, None)
                h = np.clip(r_kept[:, 3] - r_kept[:, 1], 0.0, None)
                s_kept = (w * h).astype(np.float32, copy=False)

            # 直接以 float32 保存，不再转 int32
            kept_rects = np.column_stack((r_kept, s_kept)).astype(np.float32, copy=False).tolist()

        infos_40xtile[orig_i]["global_cell_rects_dedup"] = kept_rects


    return infos_40xtile