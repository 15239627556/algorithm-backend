from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import tritonclient.grpc as grpcclient

logger = logging.getLogger("dpi357378_local_pipeline")

DET_MODEL = "DPI357378_BM_MEG_cell_detection"
CLS_MODEL = "DPI357378_BM_MEG_cell_classifier"


class LocalMegPipeline357378:
    DET_CONF_THRESHOLD = 0.3
    DET_NMS_SCORE_THRESH = 0.6
    DET_NMS_IOU_THRESH = 0.3
    NUM_CLASSES = 14
    REMAPPED_NUM_CLASSES = 6
    TOPK = 5
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        *,
        batch_size: int = 1,
        timeout_ms: int = 10000,
        device_id: int = 0,
        parallel_preprocess: bool = True,
        preprocess_workers: int = 8,
        verbose: bool = False,
        preprocess_executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._cls_batch_size = max(1, int(batch_size))
        self._timeout_s = max(1, int(timeout_ms)) / 1000.0
        self._device_id = int(device_id)
        self._parallel_preprocess = bool(parallel_preprocess)
        self._preprocess_workers = max(1, int(preprocess_workers))
        self._verbose = bool(verbose)
        # 多请求共享同一预处理池（与 147246 同思路），总线程不随 num_slots 倍增
        self._preprocess_executor = preprocess_executor

    @staticmethod
    def _decode(image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode failed")
        return bgr

    @staticmethod
    def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
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
            order = order[np.where(iou <= iou_thr)[0] + 1]
        return np.asarray(keep, dtype=np.int64)

    def _run_detection(self, client: grpcclient.InferenceServerClient, bgr: np.ndarray):
        self._set_cuda_device_if_available()
        orig_h, orig_w = bgr.shape[:2]
        resized = cv2.resize(bgr, (306, 256), interpolation=cv2.INTER_NEAREST)
        canvas = np.ones((256, 320, 3), dtype=np.uint8) * 255
        canvas[:, :306] = resized
        x = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        x = np.ascontiguousarray(x, dtype=np.float32)

        inp = grpcclient.InferInput("input.1", list(x.shape), "FP32")
        inp.set_data_from_numpy(x)
        outs = [grpcclient.InferRequestedOutput(n) for n in ("330", "329", "325", "328")]
        resp = client.infer(
            model_name=DET_MODEL,
            inputs=[inp],
            outputs=outs,
            client_timeout=self._timeout_s,
        )

        pool_cls = resp.as_numpy("330")[0, 0]
        pred_cls = resp.as_numpy("329")[0, 0]
        pred_size = resp.as_numpy("325")[0]
        pred_off = resp.as_numpy("328")[0]

        mask = (pred_cls == pool_cls) & (pred_cls >= self.DET_CONF_THRESHOLD)
        ys, xs = np.where(mask)
        if xs.size == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

        off_x = pred_off[0, ys, xs]
        off_y = pred_off[1, ys, xs]
        sz_w = pred_size[0, ys, xs]
        sz_h = pred_size[1, ys, xs]

        cx = xs.astype(np.float32) * 4.0 + off_x
        cy = ys.astype(np.float32) * 4.0 + off_y
        half_w = sz_w * 80.0 / 2.0
        half_h = sz_h * 64.0 / 2.0

        x1 = (cx - half_w) / 306.0 * orig_w
        y1 = (cy - half_h) / 256.0 * orig_h
        x2 = (cx + half_w) / 306.0 * orig_w
        y2 = (cy + half_h) / 256.0 * orig_h
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        scores = pred_cls[ys, xs].astype(np.float32)

        score_keep = np.where(scores >= self.DET_NMS_SCORE_THRESH)[0]
        boxes, scores = boxes[score_keep], scores[score_keep]
        keep = self._nms_xyxy(boxes, scores, self.DET_NMS_IOU_THRESH)
        return boxes[keep], scores[keep]

    def _classify(self, client: grpcclient.InferenceServerClient, bgr: np.ndarray, boxes: np.ndarray):
        n = boxes.shape[0]
        if n == 0:
            return np.zeros((0, self.TOPK), np.int32), np.zeros((0, self.TOPK), np.float32)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._set_cuda_device_if_available()
        logits_rows = []
        for i in range(0, n, self._cls_batch_size):
            idx_chunk = list(range(i, min(i + self._cls_batch_size, n)))
            if self._parallel_preprocess and len(idx_chunk) > 1:
                ex = self._preprocess_executor
                if ex is not None:
                    batch = list(
                        ex.map(lambda j: self._prep_cls_patch(rgb, boxes[j]), idx_chunk, chunksize=1)
                    )
                else:
                    with ThreadPoolExecutor(max_workers=min(self._preprocess_workers, len(idx_chunk))) as pool:
                        batch = list(pool.map(lambda j: self._prep_cls_patch(rgb, boxes[j]), idx_chunk))
            else:
                batch = [self._prep_cls_patch(rgb, boxes[j]) for j in idx_chunk]
            batch_np = np.ascontiguousarray(np.stack(batch, axis=0), dtype=np.float32)
            inp = grpcclient.InferInput("input.1", list(batch_np.shape), "FP32")
            inp.set_data_from_numpy(batch_np)
            out = grpcclient.InferRequestedOutput("226")
            resp = client.infer(
                model_name=CLS_MODEL,
                inputs=[inp],
                outputs=[out],
                client_timeout=self._timeout_s,
            )
            logits_rows.extend(list(resp.as_numpy("226")))

        logits = np.stack(logits_rows, axis=0)
        logits = np.clip(logits, -np.inf, 32.0)
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
        remapped_probs = self._remap_probs(probs.astype(np.float32))
        topk = np.argsort(-remapped_probs, axis=1)[:, : self.TOPK]
        topv = np.take_along_axis(remapped_probs, topk, axis=1)
        return topk.astype(np.int32), topv.astype(np.float32)

    @staticmethod
    def _remap_probs(probs: np.ndarray) -> np.ndarray:
        """
        将 14 类概率按业务规则重映射为 6 类：
        新0 <- 旧2
        新1 <- 旧3
        新2 <- 旧4
        新3 <- 旧0 + 旧1
        新4 <- 旧6
        新5 <- 旧5 + 旧7 + 旧8 + 旧9 + 旧10 + 旧11 + 旧12 + 旧13
        """
        out = np.zeros((probs.shape[0], LocalMegPipeline357378.REMAPPED_NUM_CLASSES), dtype=np.float32)
        out[:, 0] = probs[:, 2]
        out[:, 1] = probs[:, 3]
        out[:, 2] = probs[:, 4]
        out[:, 3] = probs[:, 0] + probs[:, 1]
        out[:, 4] = probs[:, 6]
        out[:, 5] = (
            probs[:, 5]
            + probs[:, 7]
            + probs[:, 8]
            + probs[:, 9]
            + probs[:, 10]
            + probs[:, 11]
            + probs[:, 12]
            + probs[:, 13]
        )
        return out

    def _prep_cls_patch(self, rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = box.astype(np.int32)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, rgb.shape[1]), min(y2, rgb.shape[0])
        if x2 <= x1 or y2 <= y1:
            crop = np.zeros((128, 128, 3), dtype=np.uint8)
        else:
            patch = rgb[y1:y2, x1:x2]
            crop = cv2.resize(patch, (128, 128), interpolation=cv2.INTER_LINEAR)
        x = crop.astype(np.float32) / 255.0
        return ((x - self.MEAN) / self.STD).transpose(2, 0, 1)

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
                logger.warning("[357378] set cv2 cuda device failed: %s", self._device_id)

    def run(self, client: grpcclient.InferenceServerClient, image_bytes: bytes) -> dict:
        bgr = self._decode(image_bytes)
        boxes, scores = self._run_detection(client, bgr)
        class_ids, probs = self._classify(client, bgr, boxes)
        return {
            "boxes": boxes.astype(np.float32),
            "scores": scores.astype(np.float32),
            "class_ids": class_ids,
            "class_probs": probs,
            "num_detections": int(boxes.shape[0]),
        }
