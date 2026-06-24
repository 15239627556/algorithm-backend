from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tritonclient.grpc as grpcclient

logger = logging.getLogger("dpi714756_bm_pb_local_pipeline")

WBC_DET_MODEL = "DPI714756_BM_PB_WBC_detector"
WBC_CLS_MODEL = "DPI714756_BM_PB_WBC_classifier"
RED_DET_MODEL = "DPI714756_BM_PB_RED_cell_detection"
RED_CLS_MODEL = "DPI714756_BM_PB_RED_cell_classifier"
PLAT_DET_MODEL = "DPI714756_BM_PB_PLAT_detection"

RED_CLS_BATCH = 32
RED_CLS_IMG_SIZE = 224

# 红细胞 YOLO 检测（动态模型：输入按原图 x0.5 缩放，宽高向上取整到 32 的倍数后补边）
RED_LETTERBOX_R = 0.5
RED_STRIDE = 32  # 模型输入宽高需被 32 整除
RED_CONF_THR = 0.20
RED_IOU_THR = 0.40
RED_OVER_THR = 0.8
RED_MAX_DET = 1000
RED_NUM_CLASSES = 3
RED_AGNOSTIC_NMS = True
RED_EXTRA_DEDUP = True
RED_MERGE_TARGET_CLS = 1  # CR 凝集红细胞
RED_MERGE_OVER_THR = 0.30
RED_MERGE_SIZE_SIM_THR = 0.65
# 红细胞检测类别名（与 Red-cell_dete 训练标签一致，用于可视化/调试）
RED_DET_CLASS_NAMES = ("SR", "CR", "OR")
RED_DET_OR_CLASS = 2  # OR(其它)：不参与分类、不返回

# 血小板 YOLO 检测（与 100x_血小板_dete/auto_infer.py 对齐）
PLAT_IMGSZ_WH = (512, 384)  # letterbox new_shape (W, H)
PLAT_CONF_THR = 0.10
PLAT_IOU_THR = 0.50
PLAT_OVERLAP_SMALL_THR = 0.80
PLAT_SKIP_LABEL = 1  # 破碎细胞，不返回

VALID_TASKS = frozenset({"wbc", "red", "plat"})

# 兼容旧客户端：0=WBC, 1=RED, 2=WBC+RED（已无 ALL=三合一，需显式传 tasks=wbc,red,plat）
TASK_WBC_ONLY = 0
TASK_RED_ONLY = 1
TASK_WBC_RED = 2

WBC_CLASS_ID_REMAP = np.array(
    [16, 13, 2, 5, 3, 4, 33, 24, 22, 25, 18, 0, 11, 21, 30, 10, 6, 9, 7, 8, 23, 26, 19, 27, 20, 32, 15, 1, 12, 17, 14, 29, 28, 31, 34],
    dtype=np.int32,
)

RED_KEEP_SCORE_THR_DEDUP = 0.7


def parse_tasks(tasks: str | list[str] | int | None = None) -> tuple[bool, bool, bool]:
    """
    解析任务组合，返回 (enable_wbc, enable_red, enable_plat)。

    tasks 可为逗号分隔字符串或列表，例如 "wbc,red,plat"。
    仍接受旧版整数 task_mode：0=WBC, 1=RED, 2=WBC+RED。
    """
    if tasks is None or tasks == "":
        return True, False, False
    if isinstance(tasks, int):
        if tasks == TASK_WBC_ONLY:
            return True, False, False
        if tasks == TASK_RED_ONLY:
            return False, True, False
        if tasks == TASK_WBC_RED:
            return True, True, False
        raise ValueError(f"invalid task_mode: {tasks}")
    if isinstance(tasks, str):
        parts = [p.strip().lower() for p in tasks.replace(";", ",").split(",") if p.strip()]
    else:
        parts = [str(p).strip().lower() for p in tasks if str(p).strip()]
    if not parts:
        return True, False, False
    unknown = sorted(set(parts) - VALID_TASKS)
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}, valid: {sorted(VALID_TASKS)}")
    return "wbc" in parts, "red" in parts, "plat" in parts


def _plat_letterbox(
    im: np.ndarray,
    new_shape: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
    auto_shape: bool = False,
    scale_fill: bool = False,
    scaleup: bool = False,
    stride: int = 32,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """与 auto_infer.letterbox 一致。"""
    shape = im.shape[:2]
    new_h = new_shape[1]
    new_w = new_shape[0]
    r = min(new_w / shape[1], new_h / shape[0])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_w - new_unpad[0], new_h - new_unpad[1]
    if auto_shape:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0, 0
        new_unpad = (new_w, new_h)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, float(r), (left, top)


def _plat_scale_boxes(
    boxes: np.ndarray,
    r: float,
    pad: tuple[int, int],
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    boxes = boxes.astype(np.float32, copy=True)
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= r
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
    return boxes


def _plat_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    overlap_small_thr: float = 0.8,
) -> list[int]:
    """与 auto_infer.nms 一致。"""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        small_areas = np.minimum(areas[i], areas[order[1:]])
        overlap_small = inter / (small_areas + 1e-7)
        inds = np.where((ovr <= iou_threshold) & (overlap_small <= overlap_small_thr))[0]
        order = order[inds + 1]
    return keep


def _red_xyxy_crop_bounds(box: np.ndarray, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """
    与 detect_class_one_image_onnx.crop_xyxy_round 一致：四角分别 round 后 clip。
    返回 (x1, y1, x2, y2)；无效框返回 None。
    """
    x1 = max(0, int(round(float(box[0]))))
    y1 = max(0, int(round(float(box[1]))))
    x2 = min(img_w, int(round(float(box[2]))))
    y2 = min(img_h, int(round(float(box[3]))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _red_xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    """原图坐标 xyxy -> xywh，供 API 返回（与历史 red_detections 格式一致）。"""
    if xyxy.shape[0] == 0:
        return np.zeros((0, 4), np.float32)
    out = np.empty_like(xyxy, dtype=np.float32)
    out[:, 0] = xyxy[:, 0]
    out[:, 1] = xyxy[:, 1]
    out[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    out[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return out


def _red_letterbox(
    im: np.ndarray,
    new_shape: tuple[int, int],
    r: float = 0.5,
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scale_fill: bool = False,
    scaleup: bool = True,
    center: bool = True,
    stride: int = 32,
    interpolation: int = cv2.INTER_LINEAR,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """与 Red-cell_dete_onnx_inference_model_not_nms.letterbox 一致：resize + pad 到 new_shape (H,W)。"""
    h, w = im.shape[:2]
    new_h, new_w = new_shape
    new_unpad = (int(round(w * r)), int(round(h * r)))
    if scale_fill:
        new_unpad = (new_w, new_h)
        r = 1.0
        dw, dh = 0.0, 0.0
    else:
        dw, dh = new_w - new_unpad[0], new_h - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    if center:
        dw /= 2.0
        dh /= 2.0
    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=interpolation)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, float(r), (float(left), float(top))


def _red_scale_boxes(boxes: np.ndarray, r: float, pad: tuple[float, float], orig_shape: tuple[int, int]) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    boxes = boxes.astype(np.float32, copy=False)
    pad_x, pad_y = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / r
    H, W = int(orig_shape[0]), int(orig_shape[1])
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
    boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0])
    boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1])
    return boxes


def _red_compute_iou_and_overlap_ratio(
    box: np.ndarray, boxes: np.ndarray, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    box = np.asarray(box, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    if box.ndim == 1:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter_w = np.maximum(0.0, x2 - x1)
        inter_h = np.maximum(0.0, y2 - y1)
        inter_area = inter_w * inter_h
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = area1 + area2 - inter_area + eps
        iou = inter_area / union_area
        overlap_ratio = inter_area / (np.minimum(area1, area2) + eps)
        return iou, overlap_ratio
    if box.ndim == 2:
        A, B = box, boxes
        x1 = np.maximum(A[:, None, 0], B[None, :, 0])
        y1 = np.maximum(A[:, None, 1], B[None, :, 1])
        x2 = np.minimum(A[:, None, 2], B[None, :, 2])
        y2 = np.minimum(A[:, None, 3], B[None, :, 3])
        inter_w = np.maximum(0.0, x2 - x1)
        inter_h = np.maximum(0.0, y2 - y1)
        inter_area = inter_w * inter_h
        area_a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
        area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
        union_area = area_a[:, None] + area_b[None, :] - inter_area + eps
        iou = inter_area / union_area
        min_area = np.minimum(area_a[:, None], area_b[None, :]) + eps
        overlap_ratio = inter_area / min_area
        return iou, overlap_ratio
    raise ValueError(f"box.ndim must be 1 or 2, got {box.ndim}")


def _red_nms_keep_largest_box(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    overlap_threshold: float,
    keep_score_thr: float,
) -> list[int]:
    keep: list[int] = []
    if boxes is None or len(boxes) == 0:
        return keep
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = areas.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]
    inds = np.arange(len(boxes))[order]
    while len(boxes) > 0:
        ref_box = boxes[0]
        ref_idx = inds[0]
        keep.append(int(ref_idx))
        if len(boxes) == 1:
            break
        iou, overlap = _red_compute_iou_and_overlap_ratio(ref_box, boxes[1:])
        dup_mask = (iou > iou_threshold) | (overlap > overlap_threshold)
        score_mask = scores[1:] >= keep_score_thr
        remove_mask = dup_mask & (~score_mask)
        mask = np.ones(len(boxes), dtype=bool)
        mask[0] = False
        mask[1:] = ~remove_mask
        boxes = boxes[mask]
        scores = scores[mask]
        inds = inds[mask]
    return keep


def _red_yolov_like_nms(
    xyxy: np.ndarray,
    conf: np.ndarray,
    cls: np.ndarray,
    iou_thr: float,
    max_det: int,
    agnostic: bool,
    max_wh: float = 7680.0,
    max_nms: int = 30000,
    classes: np.ndarray | None = None,
    extra_dedup: bool = False,
    overlap_thr: float = 0.70,
    keep_score_thr: float = 0.70,
) -> np.ndarray:
    if xyxy is None or len(xyxy) == 0:
        return np.empty((0,), dtype=np.int64)
    xyxy = np.asarray(xyxy)
    conf = np.asarray(conf)
    cls = np.asarray(cls)
    n = xyxy.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.int64)
    if classes is not None:
        cls_i = cls.astype(np.int32, copy=False)
        mask = np.isin(cls_i, np.asarray(classes, dtype=np.int32))
        if not np.any(mask):
            return np.empty((0,), dtype=np.int64)
        xyxy_f = xyxy[mask]
        conf_f = conf[mask]
        cls_f = cls_i[mask]
        orig_idx = np.nonzero(mask)[0]
    else:
        xyxy_f, conf_f, cls_f = xyxy, conf, cls.astype(np.int32, copy=False)
        orig_idx = np.arange(n, dtype=np.int64)
    m = xyxy_f.shape[0]
    if m > max_nms:
        topk = conf_f.argsort()[::-1][:max_nms]
        xyxy_f = xyxy_f[topk]
        conf_f = conf_f[topk]
        cls_f = cls_f[topk]
        orig_idx = orig_idx[topk]
    if not agnostic:
        c = (cls_f.astype(np.float32) * float(max_wh))[:, None]
        boxes_for_nms = xyxy_f + c
    else:
        boxes_for_nms = xyxy_f
    order = conf_f.argsort()[::-1]
    kept_local: list[int] = []
    while order.size > 0 and len(kept_local) < max_det:
        i = int(order[0])
        kept_local.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        iou, _ = _red_compute_iou_and_overlap_ratio(boxes_for_nms[i], boxes_for_nms[rest])
        suppr = iou > iou_thr
        order = rest[~suppr]
    kept_local_arr = np.asarray(kept_local, dtype=np.int64)
    if extra_dedup and kept_local_arr.size > 0:
        sub_boxes = xyxy_f[kept_local_arr]
        sub_scores = conf_f[kept_local_arr]
        keep2 = _red_nms_keep_largest_box(
            sub_boxes,
            sub_scores,
            iou_threshold=iou_thr,
            overlap_threshold=overlap_thr,
            keep_score_thr=keep_score_thr,
        )
        kept_local_arr = kept_local_arr[np.asarray(keep2, dtype=np.int64)]
        if kept_local_arr.size > max_det:
            kept_local_arr = kept_local_arr[:max_det]
    return orig_idx[kept_local_arr].astype(np.int64)


def _red_merge_close_or_touching_boxes_for_class(
    bboxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    target_cls: int,
    score_mode: str,
    over_merge_thr: float,
    size_sim_thr: float,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """与参考脚本 merge_close_or_touching_boxes_for_class 一致（连通域合并 target_cls）。"""
    if bboxes is None or len(bboxes) == 0:
        return bboxes, scores, labels
    bboxes = np.asarray(bboxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    idx_target = np.where(labels == target_cls)[0]
    idx_other = np.where(labels != target_cls)[0]
    out_boxes = bboxes[idx_other].copy() if len(idx_other) else np.zeros((0, 4), np.float32)
    out_scores = scores[idx_other].copy() if len(idx_other) else np.zeros((0,), np.float32)
    out_labels = labels[idx_other].copy() if len(idx_other) else np.zeros((0,), np.int32)
    if len(idx_target) == 0:
        return out_boxes, out_scores, out_labels
    B = bboxes[idx_target]
    S = scores[idx_target]
    n = B.shape[0]
    if n == 1:
        out_boxes = np.concatenate([out_boxes, B], axis=0)
        out_scores = np.concatenate([out_scores, S], axis=0)
        out_labels = np.concatenate([out_labels, np.array([target_cls], dtype=np.int32)], axis=0)
        return out_boxes, out_scores, out_labels
    _, over_mat = _red_compute_iou_and_overlap_ratio(B, B, eps=eps)
    w = np.maximum(eps, B[:, 2] - B[:, 0])
    h = np.maximum(eps, B[:, 3] - B[:, 1])
    sim_w = np.minimum(w[:, None], w[None, :]) / np.maximum(w[:, None], w[None, :])
    sim_h = np.minimum(h[:, None], h[None, :]) / np.maximum(h[:, None], h[None, :])
    sim_mat = np.minimum(sim_w, sim_h)
    adj = (over_mat >= over_merge_thr) & (sim_mat >= size_sim_thr)
    np.fill_diagonal(adj, True)
    unassigned = np.ones(n, dtype=bool)
    merged_boxes: list[list[float]] = []
    merged_scores: list[float] = []
    while unassigned.any():
        seed = int(np.argmax(unassigned))
        reach = adj[seed] & unassigned
        reach[seed] = True
        while True:
            new_reach = adj[reach].any(axis=0) & unassigned
            new_reach[seed] = True
            if np.array_equal(new_reach, reach):
                break
            reach = new_reach
        members = np.where(reach)[0]
        unassigned[members] = False
        boxes_m = B[members]
        scores_m = S[members]
        x1 = float(np.min(boxes_m[:, 0]))
        y1 = float(np.min(boxes_m[:, 1]))
        x2 = float(np.max(boxes_m[:, 2]))
        y2 = float(np.max(boxes_m[:, 3]))
        merged_boxes.append([x1, y1, x2, y2])
        if score_mode == "mean":
            merged_scores.append(float(np.mean(scores_m)))
        else:
            merged_scores.append(float(np.max(scores_m)))
    merged_boxes_arr = np.asarray(merged_boxes, dtype=np.float32)
    merged_scores_arr = np.asarray(merged_scores, dtype=np.float32)
    merged_labels_arr = np.full((len(merged_boxes_arr),), target_cls, dtype=np.int32)
    out_boxes = np.concatenate([out_boxes, merged_boxes_arr], axis=0)
    out_scores = np.concatenate([out_scores, merged_scores_arr], axis=0)
    out_labels = np.concatenate([out_labels, merged_labels_arr], axis=0)
    return out_boxes, out_scores, out_labels


def _red_postprocess_no_nms(
    pred: np.ndarray,
    conf_thr: float,
    iou_thr: float,
    max_det: int,
    agnostic: bool,
    num_classes: int,
    extra_dedup: bool,
    overlap_thr: float,
    keep_score_thr: float = RED_KEEP_SCORE_THR_DEDUP,
    apply_sigmoid: bool = False,
    classes: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """不带 NMS 的 YOLO 头输出 (B, 4+C, N) 或 (B, N, 4+C) -> 每图 (xyxy, conf, cls)。"""
    if pred.ndim != 3:
        raise RuntimeError(f"Unexpected pred ndim={pred.ndim}, expect 3.")
    a_expect = 4 + num_classes
    b, a, n = pred.shape
    if a == a_expect:
        pred = pred.transpose(0, 2, 1)
    elif n == a_expect:
        pass
    else:
        raise RuntimeError(
            f"Unexpected shape {pred.shape}, neither (B,{a_expect},N) nor (B,N,{a_expect})."
        )
    boxes_xywh = pred[..., :4].astype(np.float32)
    scores_all = pred[..., 4:].astype(np.float32)
    c = scores_all.shape[-1]
    if c != num_classes:
        num_classes = c
    if apply_sigmoid:
        scores_all = 1.0 / (1.0 + np.exp(-scores_all))
    cls_ids = scores_all.argmax(-1)
    conf_all = scores_all.max(-1)
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for bi in range(pred.shape[0]):
        m = conf_all[bi] >= conf_thr
        if not m.any():
            results.append(
                (np.empty((0, 4), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int32))
            )
            continue
        bx_xywh = boxes_xywh[bi, m]
        bx = np.empty_like(bx_xywh, dtype=np.float32)
        bx[:, 0] = bx_xywh[:, 0] - bx_xywh[:, 2] / 2
        bx[:, 1] = bx_xywh[:, 1] - bx_xywh[:, 3] / 2
        bx[:, 2] = bx_xywh[:, 0] + bx_xywh[:, 2] / 2
        bx[:, 3] = bx_xywh[:, 1] + bx_xywh[:, 3] / 2
        cf = conf_all[bi, m].astype(np.float32)
        cl = cls_ids[bi, m].astype(np.int32)
        keep = _red_yolov_like_nms(
            bx,
            cf,
            cl,
            iou_thr=iou_thr,
            max_det=min(max_det, bx.shape[0]),
            agnostic=agnostic,
            extra_dedup=extra_dedup,
            overlap_thr=overlap_thr,
            keep_score_thr=keep_score_thr,
            classes=classes,
        )
        results.append((bx[keep], cf[keep], cl[keep]))
    return results


class LocalBmpbPipeline714756:
    _bgr_npy_save_lock = threading.Lock()
    _bgr_npy_save_seq = 0

    WBC_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    WBC_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # 红细胞 4-head 分类（与训练 onnx_infer_multilabel_4head_bs32_cv.py 对齐）
    RED_CLS_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    RED_CLS_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        *,
        batch_size: int = 8,
        timeout_ms: int = 10000,
        device_id: int = 0,
        parallel_preprocess: bool = True,
        preprocess_workers: int = 8,
        verbose: bool = False,
        bgr_npy_dump_dir: str | None = None,
        preprocess_executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._cls_batch_size = max(1, int(batch_size))
        self._timeout_s = max(1, int(timeout_ms)) / 1000.0
        self._device_id = int(device_id)
        self._parallel_preprocess = bool(parallel_preprocess)
        self._preprocess_workers = max(1, int(preprocess_workers))
        self._verbose = bool(verbose)
        dump = (bgr_npy_dump_dir or "").strip()
        if not dump and self._verbose:
            dump = "debug_bgr_npy_714756"
        self._bgr_npy_dump_dir: str | None = dump or None
        # 与 147246 一致：多请求共享同一预处理池，总线程为 preprocess_workers，不随 num_slots 倍增。
        # 为 None 时每批仍临时建池，供独立调用脚本回退用。
        self._preprocess_executor = preprocess_executor

    @staticmethod
    def _decode(image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode failed")
        return bgr

    @staticmethod
    def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, iom_thr: float = 0.85) -> np.ndarray:
        if boxes.shape[0] == 0:
            return np.array([], dtype=np.int64)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
            iom = inter / (np.minimum(areas[i], areas[order[1:]]) + 1e-7)
            order = order[np.where((iou <= iou_thr) & (iom <= iom_thr))[0] + 1]
        return np.asarray(keep, dtype=np.int64)

    def _run_wbc(self, client: grpcclient.InferenceServerClient, bgr: np.ndarray):
        self._set_cuda_device_if_available()
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        resized = cv2.resize(rgb, (int(round(w * 0.25)), int(round(h * 0.25))), interpolation=cv2.INTER_LINEAR)
        tw = resized.shape[1] + (32 - resized.shape[1] % 32) % 32
        th = resized.shape[0] + (32 - resized.shape[0] % 32) % 32
        canvas = np.full((th, tw, 3), 114.0, dtype=np.float32)
        yoff = (th - resized.shape[0]) // 2
        xoff = (tw - resized.shape[1]) // 2
        canvas[yoff : yoff + resized.shape[0], xoff : xoff + resized.shape[1]] = resized
        x = (canvas / 255.0).transpose(2, 0, 1)[None, ...].astype(np.float32)

        inp = grpcclient.InferInput("images", list(x.shape), "FP32")
        inp.set_data_from_numpy(np.ascontiguousarray(x))
        out = grpcclient.InferRequestedOutput("output0")
        pred = client.infer(
            model_name=WBC_DET_MODEL,
            inputs=[inp],
            outputs=[out],
            client_timeout=self._timeout_s,
        ).as_numpy("output0")[0].T

        conf = pred[:, 4:].max(axis=1)
        mask = conf > 0.25
        if not mask.any():
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0, 5), np.int32), np.zeros((0, 5), np.float32)
        p = pred[mask]
        scores = conf[mask]
        boxes = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], axis=1)
        keep = self._nms_xyxy(boxes, scores, 0.50, 0.85)
        boxes, scores = boxes[keep], scores[keep]
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - xoff) / 0.25
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - yoff) / 0.25
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

        boxes, scores = self._filter_side_wbc_cells(bgr, boxes, scores)
        if boxes.shape[0] == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0, 5), np.int32), np.zeros((0, 5), np.float32)

        probs_all = []
        ids_all = []
        rgb_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for i in range(0, boxes.shape[0], self._cls_batch_size):
            box_chunk = boxes[i : i + self._cls_batch_size].astype(np.int32)
            if self._parallel_preprocess and len(box_chunk) > 1:
                ex = self._preprocess_executor
                if ex is not None:
                    batch = list(
                        ex.map(lambda b: self._prep_cls_patch(rgb_full, b), box_chunk, chunksize=1)
                    )
                else:
                    with ThreadPoolExecutor(max_workers=min(self._preprocess_workers, len(box_chunk))) as pool:
                        batch = list(pool.map(lambda b: self._prep_cls_patch(rgb_full, b), box_chunk))
            else:
                batch = [self._prep_cls_patch(rgb_full, bx) for bx in box_chunk]
            batch_np = np.ascontiguousarray(np.stack(batch, axis=0).astype(np.float32))
            inp = grpcclient.InferInput("input", list(batch_np.shape), "FP32")
            inp.set_data_from_numpy(batch_np)
            out = grpcclient.InferRequestedOutput("output")
            logits = client.infer(
                model_name=WBC_CLS_MODEL,
                inputs=[inp],
                outputs=[out],
                client_timeout=self._timeout_s,
            ).as_numpy("output")
            logits = np.nan_to_num(logits)
            logits = np.clip(logits, -np.inf, 32.0)
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
            topk = np.argsort(-probs, axis=1)[:, :5]
            topv = np.take_along_axis(probs, topk, axis=1)
            ids_all.append(WBC_CLASS_ID_REMAP[topk])
            probs_all.append(topv.astype(np.float32))
        class_ids = np.concatenate(ids_all, axis=0).astype(np.int32) if ids_all else np.zeros((0, 5), np.int32)
        class_probs = np.concatenate(probs_all, axis=0).astype(np.float32) if probs_all else np.zeros((0, 5), np.float32)
        return boxes.astype(np.float32), scores.astype(np.float32), class_ids, class_probs

    def _prep_cls_patch(self, rgb_full: np.ndarray, bx: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bx
        if x2 <= x1 or y2 <= y1:
            patch = np.zeros((224, 224, 3), np.float32)
        else:
            crop = rgb_full[y1:y2, x1:x2]
            m = max(crop.shape[:2])
            canvas_crop = np.zeros((m, m, 3), np.uint8)
            oy = (m - crop.shape[0]) // 2
            ox = (m - crop.shape[1]) // 2
            canvas_crop[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] = crop
            patch = cv2.resize(canvas_crop, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        return ((patch - self.WBC_MEAN) / self.WBC_STD).transpose(2, 0, 1)

    def _run_red(self, client: grpcclient.InferenceServerClient, bgr: np.ndarray):
        """
        动态模型预处理：原图直接 x0.5 缩放，缩放后宽高向上取整到 32 的倍数，
        不足部分用 (114,114,114) 居中补边 -> RGB /255 -> NCHW；解析 output0 -> NMS ->
        scale_boxes -> merge_close_or_touching_boxes_for_class(CR)；返回原图坐标下 xyxy 及检测分数/类别。
        """
        self._set_cuda_device_if_available()
        h, w = bgr.shape[:2]
        # x0.5 后的尺寸，再向上取整到 32 的倍数作为模型输入尺寸
        new_w = int(round(w * RED_LETTERBOX_R))
        new_h = int(round(h * RED_LETTERBOX_R))
        target_w = new_w + (RED_STRIDE - new_w % RED_STRIDE) % RED_STRIDE
        target_h = new_h + (RED_STRIDE - new_h % RED_STRIDE) % RED_STRIDE
        im_lb, r, pad = _red_letterbox(
            bgr,
            (target_h, target_w),
            r=RED_LETTERBOX_R,
            color=(114, 114, 114),
            auto=False,
            scale_fill=False,
            scaleup=True,
            center=True,
            stride=RED_STRIDE,
        )
        rgb = cv2.cvtColor(im_lb, cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        x = np.ascontiguousarray(x, dtype=np.float32)

        inp = grpcclient.InferInput("images", list(x.shape), "FP32")
        inp.set_data_from_numpy(x)
        out_req = grpcclient.InferRequestedOutput("output0")
        pred = client.infer(
            model_name=RED_DET_MODEL,
            inputs=[inp],
            outputs=[out_req],
            client_timeout=self._timeout_s,
        ).as_numpy("output0")

        per = _red_postprocess_no_nms(
            pred,
            conf_thr=RED_CONF_THR,
            iou_thr=RED_IOU_THR,
            max_det=RED_MAX_DET,
            agnostic=RED_AGNOSTIC_NMS,
            num_classes=RED_NUM_CLASSES,
            extra_dedup=RED_EXTRA_DEDUP,
            overlap_thr=RED_OVER_THR,
            keep_score_thr=RED_KEEP_SCORE_THR_DEDUP,
        )
        xyxy, conf, cls = per[0]
        orig_hw = (h, w)
        xyxy = _red_scale_boxes(xyxy, r, pad, orig_hw)
        xyxy, conf, cls = _red_merge_close_or_touching_boxes_for_class(
            xyxy,
            conf,
            cls,
            target_cls=RED_MERGE_TARGET_CLS,
            score_mode="mean",
            over_merge_thr=RED_MERGE_OVER_THR,
            size_sim_thr=RED_MERGE_SIZE_SIM_THR,
        )
        if xyxy.shape[0] == 0:
            return (
                np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.int32),
            )
        return xyxy.astype(np.float32), conf.astype(np.float32), cls.astype(np.int32)

    def _run_plat(self, client: grpcclient.InferenceServerClient, bgr: np.ndarray):
        """
        预处理 / 后处理与 100x_血小板_dete/auto_infer.py 一致：
        letterbox(512x384, auto_shape=False, scaleup=False) -> RGB /255 -> NCHW；
        解析 output0 -> NMS -> scale_boxes；过滤 label=1（破碎细胞）。
        """
        self._set_cuda_device_if_available()
        h, w = bgr.shape[:2]
        im_lb, r, pad = _plat_letterbox(
            bgr,
            PLAT_IMGSZ_WH,
            color=(114, 114, 114),
            auto_shape=False,
            scale_fill=False,
            scaleup=False,
        )
        blob = cv2.cvtColor(im_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.ascontiguousarray(blob.transpose(2, 0, 1)[None, ...], dtype=np.float32)

        inp = grpcclient.InferInput("images", list(x.shape), "FP32")
        inp.set_data_from_numpy(x)
        out_req = grpcclient.InferRequestedOutput("output0")
        pred = client.infer(
            model_name=PLAT_DET_MODEL,
            inputs=[inp],
            outputs=[out_req],
            client_timeout=self._timeout_s,
        ).as_numpy("output0")[0].T  # (anchors, 5)

        scores = pred[:, 4:].max(axis=1)
        mask = scores > PLAT_CONF_THR
        if not mask.any():
            return (
                np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.int32),
            )

        m_pred = pred[mask]
        m_scores = scores[mask]
        m_labels = np.argmax(m_pred[:, 4:], axis=1).astype(np.int32)

        boxes = np.zeros((len(m_pred), 4), dtype=np.float32)
        boxes[:, 0] = m_pred[:, 0] - m_pred[:, 2] / 2
        boxes[:, 1] = m_pred[:, 1] - m_pred[:, 3] / 2
        boxes[:, 2] = m_pred[:, 0] + m_pred[:, 2] / 2
        boxes[:, 3] = m_pred[:, 1] + m_pred[:, 3] / 2

        keep = _plat_nms(boxes, m_scores, PLAT_IOU_THR, overlap_small_thr=PLAT_OVERLAP_SMALL_THR)
        if not keep:
            return (
                np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.int32),
            )

        final_boxes = _plat_scale_boxes(boxes[keep].copy(), r, pad, w, h)
        final_scores = m_scores[keep].astype(np.float32)
        final_labels = m_labels[keep].astype(np.int32)

        valid = final_labels != PLAT_SKIP_LABEL
        final_boxes = final_boxes[valid]
        final_scores = final_scores[valid]
        final_labels = final_labels[valid]
        return final_boxes, final_scores, final_labels

    @staticmethod
    def _filter_side_wbc_cells(
        bgr: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        border_ratio: float = 0.015,
        sat_threshold: int = 100,
        sat_area_ratio: float = 0.2,
        strip_width: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        过滤靠图像边缘的 WBC 细胞（移植自 libX100CellLoc.cpp filterSideCell 逻辑）。

        对每个检测框，若其任意一侧边界落在图像边缘 border_ratio 范围内，
        则在该侧取宽度为 strip_width 像素的条带（原图 HSV 饱和度通道），
        若条带内饱和度 > sat_threshold 的像素占比超过 sat_area_ratio，
        则判定为靠边细胞并过滤（说明边缘处仍有细胞内容延伸到视野外）。

        参数
        ----
        border_ratio  : 判定"靠边"的阈值，默认 1.5%（与 C++ 0.015 一致）。
        sat_threshold : HSV-S 饱和度阈值，默认 100（与 C++ threshold=100 一致）。
        sat_area_ratio: 饱和像素占条带面积的比例阈值，默认 20%（与 C++ .2 一致）。
        strip_width   : 边缘条带宽度（像素），默认 5（与 C++ rect 宽/高 5 一致）。
        """
        if boxes.shape[0] == 0:
            return boxes, scores

        img_h, img_w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]  # 饱和度通道，与 C++ channels[1] 一致

        keep_mask = np.ones(boxes.shape[0], dtype=bool)

        left_thr   = img_w * border_ratio
        right_thr  = img_w * (1.0 - border_ratio)
        top_thr    = img_h * border_ratio
        bottom_thr = img_h * (1.0 - border_ratio)

        for i in range(boxes.shape[0]):
            x1 = max(0, int(boxes[i, 0]))
            y1 = max(0, int(boxes[i, 1]))
            x2 = min(img_w, int(boxes[i, 2]))
            y2 = min(img_h, int(boxes[i, 3]))

            is_border = False
            L_flag = R_flag = T_flag = B_flag = False

            if x1 <= left_thr:
                strip = sat[y1:y2, 0:strip_width]
                if strip.size > 0:
                    is_border = True
                    L_flag = int(np.count_nonzero(strip > sat_threshold)) > strip.size * sat_area_ratio

            if x2 >= right_thr:
                strip = sat[y1:y2, img_w - strip_width:img_w]
                if strip.size > 0:
                    is_border = True
                    R_flag = int(np.count_nonzero(strip > sat_threshold)) > strip.size * sat_area_ratio

            if y1 <= top_thr:
                strip = sat[0:strip_width, x1:x2]
                if strip.size > 0:
                    is_border = True
                    T_flag = int(np.count_nonzero(strip > sat_threshold)) > strip.size * sat_area_ratio

            if y2 >= bottom_thr:
                strip = sat[img_h - strip_width:img_h, x1:x2]
                if strip.size > 0:
                    is_border = True
                    B_flag = int(np.count_nonzero(strip > sat_threshold)) > strip.size * sat_area_ratio

            if is_border and (L_flag or R_flag or T_flag or B_flag):
                keep_mask[i] = False

        return boxes[keep_mask], scores[keep_mask]

    @staticmethod
    def _softmax_2d(logits: np.ndarray) -> np.ndarray:
        l = np.nan_to_num(logits)
        l = l - l.max(axis=1, keepdims=True)
        e = np.exp(l)
        return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)

    @staticmethod
    def _red_agg_top1(la: np.ndarray, loc_classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        聚集 head 后处理（与 detect_class_one_image_onnx.classify_cells_onnx 一致）：
        - SR(0)：固定 agg_idx=0，prob=1.0
        - CR(1)：对原始 logits 的 la[1:] 做 softmax（不是对全 3 类 softmax 后再切片），再 argmax，索引 +1。
          若误用全类 softmax 后的 pa[1:] 取值，概率只是边际而非「缗钱状 vs 凝集」上的归一化置信度，会明显偏小。
        """
        valid = int(la.shape[0])
        ai = np.zeros(valid, dtype=np.int32)
        a_probs = np.ones(valid, dtype=np.float32)
        for j in range(valid):
            if int(loc_classes[j]) == 1:
                sub = np.nan_to_num(la[j, 1:].astype(np.float64))
                sub = sub - np.max(sub)
                e = np.exp(sub)
                prob_a_12 = (e / np.maximum(e.sum(), 1e-12)).astype(np.float32)
                sub_idx = int(np.argmax(prob_a_12))
                ai[j] = sub_idx + 1
                a_probs[j] = float(prob_a_12[sub_idx])
        return ai, a_probs

    def _prep_red_cls_patch(self, rgb_full: np.ndarray, box_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        从整图按 xyxy 裁切（与 detect_class_one_image_onnx.crop_xyxy_round 一致），
        ResizeLongestSide -> PadToSquare -> /255 -> ImageNet 归一化 -> CHW；
        几何特征 [w/W, h/H, w*h/(W*H), w/h] 使用 round 后的 bbox 宽高。
        """
        img_h, img_w = rgb_full.shape[:2]
        bounds = _red_xyxy_crop_bounds(box_xyxy, img_w, img_h)
        if bounds is None:
            x1 = y1 = x2 = y2 = 0
            cw = ch = 0
        else:
            x1, y1, x2, y2 = bounds
            cw = x2 - x1
            ch = y2 - y1

        size = RED_CLS_IMG_SIZE
        if cw == 0 or ch == 0:
            patch = np.zeros((size, size, 3), np.float32)
        else:
            crop = rgb_full[y1:y2, x1:x2]
            long_side = max(cw, ch)
            scale = size / float(long_side)
            new_w = max(1, int(round(cw * scale)))
            new_h = max(1, int(round(ch * scale)))
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            pad_w = size - new_w
            pad_h = size - new_h
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )
            patch = padded.astype(np.float32) / 255.0

        chw = ((patch - self.RED_CLS_MEAN) / self.RED_CLS_STD).transpose(2, 0, 1)

        full_w_f = float(img_w) if img_w > 0 else 1.0
        full_h_f = float(img_h) if img_h > 0 else 1.0
        cw_f = float(cw)
        ch_f = float(ch)
        geom = np.array(
            [
                cw_f / full_w_f,
                ch_f / full_h_f,
                (cw_f * ch_f) / (full_w_f * full_h_f),
                (cw_f / ch_f) if ch_f > 0 else 0.0,
            ],
            dtype=np.float32,
        )
        return chw.astype(np.float32), geom

    def _run_red_cls(
        self,
        client: grpcclient.InferenceServerClient,
        bgr: np.ndarray,
        red_boxes_xyxy: np.ndarray,
        red_det_class_ids: np.ndarray,
    ):
        """
        对 _run_red 输出的 [M, 4] xyxy 检测框逐个裁切并送入分类器。
        red_det_class_ids: [M] 检测定位类别（0=SR, 1=CR）；OR 应在调用前已过滤。
        返回 4 个 head 的 top-1 id 和概率（每个 shape=[M]）。
        """
        empty_id = np.zeros((0,), np.int32)
        empty_p = np.zeros((0,), np.float32)
        m = int(red_boxes_xyxy.shape[0])
        if m == 0:
            return empty_id, empty_p, empty_id, empty_p, empty_id, empty_p, empty_id, empty_p

        self._set_cuda_device_if_available()
        rgb_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        s_ids, s_probs = [], []
        c_ids, c_probs = [], []
        m_ids, m_probs = [], []
        a_ids, a_probs = [], []

        for i in range(0, m, RED_CLS_BATCH):
            chunk = red_boxes_xyxy[i : i + RED_CLS_BATCH]
            valid = int(chunk.shape[0])

            if self._parallel_preprocess and valid > 1:
                ex = self._preprocess_executor
                if ex is not None:
                    pairs = list(
                        ex.map(lambda b: self._prep_red_cls_patch(rgb_full, b), chunk, chunksize=1)
                    )
                else:
                    with ThreadPoolExecutor(max_workers=min(self._preprocess_workers, valid)) as pool:
                        pairs = list(pool.map(lambda b: self._prep_red_cls_patch(rgb_full, b), chunk))
            else:
                pairs = [self._prep_red_cls_patch(rgb_full, bx) for bx in chunk]

            patches = [p[0] for p in pairs]
            geoms = [p[1] for p in pairs]

            # 模型固定 batch=32：不足补齐（用最后一个），多余的丢弃
            while len(patches) < RED_CLS_BATCH:
                patches.append(patches[-1])
                geoms.append(geoms[-1])

            x_batch = np.ascontiguousarray(np.stack(patches, axis=0).astype(np.float32))
            g_batch = np.ascontiguousarray(np.stack(geoms, axis=0).astype(np.float32))

            inp_imgs = grpcclient.InferInput("images", list(x_batch.shape), "FP32")
            inp_imgs.set_data_from_numpy(x_batch)
            inp_geom = grpcclient.InferInput("geom_feat", list(g_batch.shape), "FP32")
            inp_geom.set_data_from_numpy(g_batch)
            outs = [
                grpcclient.InferRequestedOutput(n)
                for n in ("logits_struct", "logits_color", "logits_morph", "logits_agg")
            ]
            resp = client.infer(
                model_name=RED_CLS_MODEL,
                inputs=[inp_imgs, inp_geom],
                outputs=outs,
                client_timeout=self._timeout_s,
            )

            ls = resp.as_numpy("logits_struct")[:valid]
            lc = resp.as_numpy("logits_color")[:valid]
            lm = resp.as_numpy("logits_morph")[:valid]
            la = resp.as_numpy("logits_agg")[:valid]

            ps = self._softmax_2d(ls)
            pc = self._softmax_2d(lc)
            pm = self._softmax_2d(lm)
            si = np.argmax(ps, axis=1).astype(np.int32)
            ci = np.argmax(pc, axis=1).astype(np.int32)
            mi = np.argmax(pm, axis=1).astype(np.int32)
            chunk_loc = red_det_class_ids[i : i + RED_CLS_BATCH][:valid]
            ai, ap = self._red_agg_top1(la, chunk_loc)

            s_ids.append(si)
            s_probs.append(np.take_along_axis(ps, si[:, None].astype(np.intp), axis=1).squeeze(1).astype(np.float32))
            c_ids.append(ci)
            c_probs.append(np.take_along_axis(pc, ci[:, None].astype(np.intp), axis=1).squeeze(1).astype(np.float32))
            m_ids.append(mi)
            m_probs.append(np.take_along_axis(pm, mi[:, None].astype(np.intp), axis=1).squeeze(1).astype(np.float32))
            a_ids.append(ai)
            a_probs.append(ap.astype(np.float32))

        return (
            np.concatenate(s_ids).astype(np.int32),
            np.concatenate(s_probs).astype(np.float32),
            np.concatenate(c_ids).astype(np.int32),
            np.concatenate(c_probs).astype(np.float32),
            np.concatenate(m_ids).astype(np.int32),
            np.concatenate(m_probs).astype(np.float32),
            np.concatenate(a_ids).astype(np.int32),
            np.concatenate(a_probs).astype(np.float32),
        )

    def _save_decoded_bgr_npy(self, bgr: np.ndarray) -> Path | None:
        """
        将 cv2.imdecode 得到的 BGR 整图保存为 .npy。

        数组约定（与 OpenCV 一致，便于与参考脚本 cv2.imread 对比）：
          shape = (H, W, 3)，dtype=uint8
          内存顺序 C-contiguous：先 row(行/y)，再 col(列/x)，再 channel
          channel 顺序 BGR：bgr[row, col, 0]=B, [..., 1]=G, [..., 2]=R
        读取：bgr = np.load(path)  # 不要用 np.load(..., allow_pickle=True)
        """
        if not self._bgr_npy_dump_dir:
            return None
        dump_dir = Path(self._bgr_npy_dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        with type(self)._bgr_npy_save_lock:
            type(self)._bgr_npy_save_seq += 1
            seq = type(self)._bgr_npy_save_seq
        h, w = int(bgr.shape[0]), int(bgr.shape[1])
        ts = int(time.time() * 1000)
        path = dump_dir / f"decoded_bgr_{h}x{w}_{ts}_{seq:06d}.npy"
        np.save(path, np.ascontiguousarray(bgr))
        meta_path = dump_dir / "README_bgr_npy.txt"
        if not meta_path.is_file():
            meta_path.write_text(
                "decoded_bgr_*.npy 由 dpi714756 LocalBmpbPipeline714756 保存。\n"
                "shape: (H, W, 3), dtype=uint8, C-order, 像素遍历顺序 row->col->channel。\n"
                "channel: BGR (OpenCV), bgr[y,x,0]=B, bgr[y,x,1]=G, bgr[y,x,2]=R。\n"
                "加载: bgr = np.load('xxx.npy')\n",
                encoding="utf-8",
            )
        if self._verbose:
            logger.info(
                "[714756] saved decoded BGR npy: %s shape=%s dtype=%s order=C row,col,BGR",
                path.resolve(),
                bgr.shape,
                bgr.dtype,
            )
        return path

    @staticmethod
    def _debug_print_bgr(
        bgr: np.ndarray,
        *,
        label: str = "BGRImage",
        rows: int = 4,
        cols: int = 4,
        corner: str = "topleft",
    ) -> None:
        """
        记录 BGR 图像部分像素供两边对比。

        打印格式（每像素一行）：
            [label] row=R col=C  B=b G=g R=r
        坐标系：row=0 对应图像最顶行，col=0 对应最左列。

        参数
        ----
        rows / cols  : 从所选角落取多少行/列。
        corner       : 从哪个角取样，可选 "topleft"（默认）/ "topright" /
                       "bottomleft" / "bottomright"。
        """
        h, w = bgr.shape[:2]
        r_start = h - rows if corner in ("bottomleft", "bottomright") else 0
        c_start = w - cols if corner in ("topright", "bottomright") else 0
        r_end = min(r_start + rows, h)
        c_end = min(c_start + cols, w)
        logger.debug(
            "[DEBUG][%s] shape=(%s,%s,3) dtype=%s corner=%s sample_rows=%s..%s sample_cols=%s..%s",
            label,
            h,
            w,
            bgr.dtype,
            corner,
            r_start,
            r_end - 1,
            c_start,
            c_end - 1,
        )
        logger.debug("[DEBUG][%s] 格式: row=行索引(0=顶部) col=列索引(0=左侧) B G R（OpenCV BGR顺序）", label)
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                b, g, rv = int(bgr[r, c, 0]), int(bgr[r, c, 1]), int(bgr[r, c, 2])
                logger.debug("[DEBUG][%s] row=%5d col=%5d  B=%3d G=%3d R=%3d", label, r, c, b, g, rv)

    def _set_cuda_device_if_available(self) -> None:
        cuda_mod = getattr(cv2, "cuda", None)
        if cuda_mod is None:
            return
        try:
            if cuda_mod.getCudaEnabledDeviceCount() <= 0:
                return
            wanted = max(0, self._device_id)
            dev = min(wanted, int(cuda_mod.getCudaEnabledDeviceCount()) - 1)
            cuda_mod.setDevice(dev)
        except Exception:
            if self._verbose:
                logger.warning("[714756] set cv2 cuda device failed: %s", self._device_id)

    def run(
        self,
        client: grpcclient.InferenceServerClient,
        image_bytes: bytes,
        tasks: str | list[str] | int | None = "wbc",
        *,
        task_mode: int | None = None,
    ) -> dict:
        """
        tasks: 逗号分隔任务名 wbc / red / plat，可组合，如 "wbc,red,plat"。
        task_mode: 已弃用，仅兼容旧客户端（0/1/2）；显式传入 tasks 时忽略 task_mode。
        """
        if task_mode is not None:
            enable_wbc, enable_red, enable_plat = parse_tasks(task_mode)
        elif tasks is not None:
            enable_wbc, enable_red, enable_plat = parse_tasks(tasks)
        else:
            enable_wbc, enable_red, enable_plat = True, False, False

        bgr = self._decode(image_bytes)
        self._save_decoded_bgr_npy(bgr)
        if self._verbose:
            self._debug_print_bgr(bgr, label="decoded_bgr")

        if enable_wbc:
            boxes, scores, class_ids, class_probs = self._run_wbc(client, bgr)
        else:
            boxes = np.zeros((0, 4), np.float32)
            scores = np.zeros((0,), np.float32)
            class_ids = np.zeros((0, 5), np.int32)
            class_probs = np.zeros((0, 5), np.float32)

        empty_id = np.zeros((0,), np.int32)
        empty_p = np.zeros((0,), np.float32)
        if enable_red:
            red_xyxy, red_det_scores, red_det_class_ids = self._run_red(client, bgr)
            if red_xyxy.shape[0] > 0:
                keep = red_det_class_ids != RED_DET_OR_CLASS
                red_xyxy = red_xyxy[keep]
                red_det_scores = red_det_scores[keep]
                red_det_class_ids = red_det_class_ids[keep]
            (
                red_struct, red_struct_p,
                red_color, red_color_p,
                red_morph, red_morph_p,
                red_agg, red_agg_p,
            ) = self._run_red_cls(client, bgr, red_xyxy, red_det_class_ids)
            red = _red_xyxy_to_xywh(red_xyxy)
        else:
            red = np.zeros((0, 4), np.float32)
            red_det_scores, red_det_class_ids = empty_p, empty_id
            red_struct, red_struct_p = empty_id, empty_p
            red_color, red_color_p = empty_id, empty_p
            red_morph, red_morph_p = empty_id, empty_p
            red_agg, red_agg_p = empty_id, empty_p

        if enable_plat:
            plat_xyxy, plat_det_scores, plat_det_class_ids = self._run_plat(client, bgr)
            plat = _red_xyxy_to_xywh(plat_xyxy)
        else:
            plat = np.zeros((0, 4), np.float32)
            plat_det_scores, plat_det_class_ids = empty_p, empty_id

        return {
            "tasks": {
                "wbc": enable_wbc,
                "red": enable_red,
                "plat": enable_plat,
            },
            "boxes": boxes,
            "scores": scores,
            "class_ids": class_ids,
            "class_probs": class_probs,
            "num_detections": int(boxes.shape[0]),
            "red_detections": red,
            "red_det_scores": red_det_scores,
            "red_det_class_ids": red_det_class_ids,
            "red_num_detections": int(red.shape[0]),
            "red_class_struct": red_struct,
            "red_class_struct_prob": red_struct_p,
            "red_class_color": red_color,
            "red_class_color_prob": red_color_p,
            "red_class_morph": red_morph,
            "red_class_morph_prob": red_morph_p,
            "red_class_agg": red_agg,
            "red_class_agg_prob": red_agg_p,
            "plat_detections": plat,
            "plat_det_scores": plat_det_scores,
            "plat_det_class_ids": plat_det_class_ids,
            "plat_num_detections": int(plat.shape[0]),
        }
