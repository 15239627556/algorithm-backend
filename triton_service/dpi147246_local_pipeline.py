"""
DPI147246_BM_PB_pipeline 的客户端编排版本。

服务端 Python backend (`model_repository/DPI147246_BM_PB_pipeline/1/model.py`)
把 JPEG 解码、resize、NMS、像素统计等 CPU 工作放在 Triton 里做，并通过 BLS
调用 4 个 TensorRT 子模型。这会让每个 Python stub 进程都初始化自己的
CUDA context / PyTorch caching allocator，VRAM 随 `instance_group.count`
线性增长（单 stub 约 0.9–1.3 GB）。

本文件保持**推理语义完全一致**，只是把编排挪到客户端：
    - CPU 前后处理在客户端 Python 里跑
    - 子模型 TensorRT engine 仍然通过 gRPC 在 Triton 服务端的 GPU 上推理
    - 同一 pipeline 内的 4 个子模型请求用 `async_infer` 并行发起，等价于原
      BLS `exec()` 并发

对外接口 `LocalWbcMegPipeline.run(client, image_bytes, enable_meg)` 返回与
`client.py::infer_wbc_meg` 完全一致的 dict，可以直接替换使用。

注意：
    - pre-processing 中的 "612x512 resize" 与 "MEG_SCALE_RATIO=0.25" 是
      server 原始实现的硬编码（假设原图 2448x2048），本文件严格保留；
      原图如果是 4896x4096，MEG 坐标会映射到 2448x2048 坐标系（与
      server 行为一致）。
    - Cell Analysis 像素计数的 `orig_sizes` 使用解码后 BGR 的真实
      (H, W)，因此 4896x4096 与 2448x2048 的 count 比例是对的。
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import cv2
import numpy as np
import tritonclient.grpc as grpcclient

# 可选：Numba 加速 NMS（与纯 NumPy 路径语义一致；未安装时自动回退）
try:
    from numba import njit

    @njit(cache=True)
    def _nms_numpy_numba(
        boxes: np.ndarray, scores: np.ndarray, iou_thr: float
    ) -> np.ndarray:
        n = boxes.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.int64)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = np.argsort(-scores)
        keep = np.empty(n, dtype=np.int64)
        ord_buf = np.empty(n, dtype=np.int64)
        ord_next = np.empty(n, dtype=np.int64)
        for j in range(n):
            ord_buf[j] = order[j]
        ln = n
        nk = 0
        eps = 1e-7
        while ln > 0:
            i = ord_buf[0]
            keep[nk] = i
            nk += 1
            if ln == 1:
                break
            new_ln = 0
            ai = areas[i]
            xi1 = x1[i]
            yi1 = y1[i]
            xi2 = x2[i]
            yi2 = y2[i]
            for k in range(1, ln):
                j = ord_buf[k]
                xx1 = max(x1[j], xi1)
                yy1 = max(y1[j], yi1)
                xx2 = min(x2[j], xi2)
                yy2 = min(y2[j], yi2)
                iw = max(0.0, xx2 - xx1)
                ih = max(0.0, yy2 - yy1)
                inter = iw * ih
                uni = areas[j] + ai - inter + eps
                iou = inter / uni
                if iou <= iou_thr:
                    ord_next[new_ln] = j
                    new_ln += 1
            for t in range(new_ln):
                ord_buf[t] = ord_next[t]
            ln = new_ln
        out = np.empty(nk, dtype=np.int64)
        for t in range(nk):
            out[t] = keep[t]
        return out

    @njit(cache=True)
    def _nms_overlap_numba(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_thr: float,
        overlap_thr: float,
    ) -> np.ndarray:
        n = boxes.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.int64)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = np.argsort(-scores)
        suppressed = np.zeros(n, dtype=np.uint8)
        keep = np.empty(n, dtype=np.int64)
        nk = 0
        eps = 1e-7
        for t in range(n):
            idx = order[t]
            if suppressed[idx] != 0:
                continue
            keep[nk] = idx
            nk += 1
            aidx = areas[idx]
            x1i = x1[idx]
            y1i = y1[idx]
            x2i = x2[idx]
            y2i = y2[idx]
            for j in range(n):
                xx1 = max(x1[j], x1i)
                yy1 = max(y1[j], y1i)
                xx2 = min(x2[j], x2i)
                yy2 = min(y2[j], y2i)
                iw = max(0.0, xx2 - xx1)
                ih = max(0.0, yy2 - yy1)
                inter = iw * ih
                iou = inter / (areas[j] + aidx - inter + eps)
                ovl = inter / (min(areas[j], aidx) + eps)
                if iou > iou_thr or ovl > overlap_thr:
                    suppressed[j] = 1
            suppressed[idx] = 0
        out = np.empty(nk, dtype=np.int64)
        for t in range(nk):
            out[t] = keep[t]
        return out

    _USE_NUMBA_NMS = True
except ImportError:  # pragma: no cover - 无 numba 时走纯 NumPy
    _USE_NUMBA_NMS = False

# 可选：torch GPU letterbox（与 OpenCV 路径在 resize 上可能略有数值差，后处理 meta 一致）
try:
    import torch
    import torch.nn.functional as F_torch

    def wbc_torch_cuda_available() -> bool:
        return bool(torch.cuda.is_available())
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F_torch = None  # type: ignore[assignment]

    def wbc_torch_cuda_available() -> bool:
        return False


# ── 子模型名称 ─────────────────────────────────────────────────────────
MODEL_WBC = "DPI147246_BM_PB_WBC_cell_detection"
MODEL_MEG = "DPI147246_BM_PB_MEG_cell_detection"
MODEL_CON = "DPI147246_BM_PB_constituency_score"
MODEL_CA = "DPI147246_BM_PB_cell_analysis"


class LocalWbcMegPipeline:
    """DPI147246_BM_PB_pipeline 的客户端编排实现。

    与 server 版一一对应的常量、前后处理；只把 BLS 改为 gRPC。"""

    # WBC
    WBC_INPUT_H, WBC_INPUT_W = 1024, 1248
    WBC_CONF_THR = 0.35
    WBC_IOU_THR = 0.50
    WBC_AREA_MIN, WBC_AREA_MAX = 200, 100000
    WBC_MAX_BATCH = 4

    # MEG
    MEG_INPUT_H, MEG_INPUT_W = 512, 640
    MEG_SCALE_RATIO = 0.25
    MEG_CONF_THR = 0.35
    MEG_IOU_THR = 0.60
    MEG_AREA_MIN, MEG_AREA_MAX = 7000, 350000
    MEG_MAX_DET = 1000
    MEG_MAX_BATCH = 4

    # Constituency
    CONST_INPUT_H, CONST_INPUT_W = 224, 224
    CONST_NUM_CLASSES = 7
    CONST_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    CONST_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    CONST_MAX_BATCH = 16
    CONST_CLASS_DICT = np.array([64, 32, 16, 8, 4, 2, 1], dtype=np.float32)

    # Cell Analysis
    CA_INPUT_H, CA_INPUT_W = 512, 640
    CA_RESIZE_W, CA_RESIZE_H = 612, 512
    CA_BATCH_SIZE = 4
    CA_PAD_VALUE = 230
    CA_PIXEL_THR = 128

    # ──────────────────────────────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────────────────────────────
    def run(
        self,
        client: grpcclient.InferenceServerClient,
        image_bytes: bytes,
        enable_meg: bool = True,
    ) -> dict:
        """执行完整 pipeline，返回与 server 版 infer_wbc_meg 一致的 dict。"""
        bgr = self._decode(image_bytes)
        if bgr is None:
            raise RuntimeError("cv2.imdecode 失败：请确认输入是合法 JPEG/PNG bytes")
        bgr_612 = cv2.resize(bgr, (612, 512), interpolation=cv2.INTER_LINEAR)

        # ── 各子模型预处理 ────────────────────────────────────────────
        wbc_in, wbc_meta = self._wbc_preprocess_batch([bgr])
        con_in, con_regions = self._constituency_preprocess_batch([bgr_612])
        ca_in = self._cell_analysis_preprocess_batch([bgr_612])
        ca_orig_sizes = [(bgr.shape[0], bgr.shape[1])]

        meg_in = None
        meg_meta = None
        if enable_meg:
            meg_in, meg_meta = self._meg_preprocess_batch([bgr_612])

        # ── 并行发起 4 路推理 ────────────────────────────────────────
        callers: dict[str, _AsyncCall] = {}
        callers["wbc"] = _async_infer(
            client, MODEL_WBC,
            inputs=[("images", wbc_in, "FP32")],
            outputs=["output0"],
        )
        callers["con"] = _async_infer(
            client, MODEL_CON,
            inputs=[("input", con_in, "FP32")],
            outputs=["output"],
        )
        callers["ca"] = _async_infer(
            client, MODEL_CA,
            inputs=[("data", ca_in, "FP32")],
            outputs=["222"],
        )
        if enable_meg:
            callers["meg"] = _async_infer(
                client, MODEL_MEG,
                inputs=[("images", meg_in, "FP32")],
                outputs=["output0"],
            )

        # 等待全部完成
        results = {key: c.wait() for key, c in callers.items()}

        # ── 后处理 ──────────────────────────────────────────────────
        wbc_out = results["wbc"].as_numpy("output0")
        con_out = results["con"].as_numpy("output")
        ca_out = results["ca"].as_numpy("222")

        wbc_res = self._wbc_postprocess_batch(wbc_out, wbc_meta)[0]
        con_res = self._constituency_postprocess_batch(con_out, 1, con_regions)[0]
        ca_counts_wbc, ca_counts_red = self._cell_analysis_postprocess_and_count(
            ca_out, ca_orig_sizes
        )

        if enable_meg:
            meg_out = results["meg"].as_numpy("output0")
            meg_res_list = self._meg_postprocess_batch(meg_out, meg_meta)
            meg_res = meg_res_list[0]
        else:
            meg_res = np.zeros((0, 5), dtype=np.float32)

        con_regions_arr, con_grades_arr, con_scores_arr = con_res

        return {
            "wbc_detections": wbc_res if wbc_res.shape[0] > 0 else np.array([]),
            "wbc_num": int(wbc_res.shape[0]),
            "meg_detections": meg_res if meg_res.shape[0] > 0 else np.array([]),
            "meg_num": int(meg_res.shape[0]),
            "constituency_regions": con_regions_arr,
            "constituency_grades": con_grades_arr,
            "constituency_scores": con_scores_arr,
            "cell_analysis_wbc_pixel_count": int(ca_counts_wbc[0]),
            "cell_analysis_red_pixel_count": int(ca_counts_red[0]),
        }

    # ──────────────────────────────────────────────────────────────────
    # 工具
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _decode(jpg_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(jpg_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # ═══════════════════════════════════════════════════════════════════
    # WBC
    # ═══════════════════════════════════════════════════════════════════
    def _wbc_preprocess_batch(self, bgr_imgs: list) -> tuple:
        blobs = []
        metas = []
        for img in bgr_imgs:
            orig_h, orig_w = img.shape[:2]
            new_h, new_w = self.WBC_INPUT_H, self.WBC_INPUT_W
            r = min(new_w / orig_w, new_h / orig_h, 1.0)
            nw, nh = int(round(orig_w * r)), int(round(orig_h * r))
            resized = cv2.resize(img, (nw, nh)) if (orig_w, orig_h) != (nw, nh) else img
            dw, dh = (new_w - nw) / 2, (new_h - nh) / 2
            top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
            padded = cv2.copyMakeBorder(
                resized, top, int(round(dh + 0.1)), left, int(round(dw + 0.1)),
                cv2.BORDER_CONSTANT, value=(114, 114, 114),
            )
            blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blobs.append(blob.transpose(2, 0, 1))
            metas.append((r, (left, top), (orig_w, orig_h)))
        return np.ascontiguousarray(np.stack(blobs, axis=0)), metas

    def _wbc_preprocess_batch_cuda(
        self,
        bgr_imgs: list,
        *,
        device: Any,
        stream: Any,
    ) -> tuple[Any, list]:
        """WBC letterbox 在 GPU 上完成（torch 双线性 resize + 与 CPU 相同的 pad 整数）。

        与 `_wbc_preprocess_batch` 相比：OpenCV 对 uint8 做 INTER_LINEAR，本路径在
        float 域双线性，框坐标可能有亚像素级差异；`metas` 的 r / (left,top) 与
        CPU 实现一致，便于后处理对齐原图尺度。
        """
        if torch is None or F_torch is None:
            raise RuntimeError("未安装 torch，无法使用 GPU WBC 预处理")
        blobs: list = []
        metas: list = []
        new_h, new_w = self.WBC_INPUT_H, self.WBC_INPUT_W
        pad_v = 114.0 / 255.0
        with torch.cuda.stream(stream):
            for img in bgr_imgs:
                orig_h, orig_w = int(img.shape[0]), int(img.shape[1])
                r = min(new_w / orig_w, new_h / orig_h, 1.0)
                nw = int(round(orig_w * r))
                nh = int(round(orig_h * r))
                dw = (new_w - nw) * 0.5
                dh = (new_h - nh) * 0.5
                top = int(round(dh - 0.1))
                bottom = int(round(dh + 0.1))
                left = int(round(dw - 0.1))
                right = int(round(dw + 0.1))

                t = torch.from_numpy(img).to(
                    device=device, dtype=torch.uint8, non_blocking=False
                )
                x = t.to(dtype=torch.float32).mul_(1.0 / 255.0).permute(2, 0, 1).unsqueeze(0)
                if orig_w != nw or orig_h != nh:
                    x = F_torch.interpolate(
                        x, size=(nh, nw), mode="bilinear", align_corners=False
                    )
                x = F_torch.pad(x, (left, right, top, bottom), value=pad_v)
                x = x[:, [2, 1, 0], :, :]
                blobs.append(x.squeeze(0))
                metas.append((r, (left, top), (orig_w, orig_h)))
            stacked = torch.stack(blobs, dim=0).contiguous()
        return stacked, metas

    def _wbc_preprocess_batch_cuda_from_gpu(
        self,
        bgr_imgs_u8_gpu: list,
        *,
        stream: Any,
    ) -> tuple[Any, list]:
        """WBC 前处理在 GPU 完成（输入为 GPU uint8 BGR 列表）。"""
        if torch is None or F_torch is None:
            raise RuntimeError("未安装 torch，无法使用 GPU WBC 预处理")
        blobs: list = []
        metas: list = []
        new_h, new_w = self.WBC_INPUT_H, self.WBC_INPUT_W
        pad_v = 114.0 / 255.0
        with torch.cuda.stream(stream):
            for t_u8 in bgr_imgs_u8_gpu:
                orig_h, orig_w = int(t_u8.shape[0]), int(t_u8.shape[1])
                r = min(new_w / orig_w, new_h / orig_h, 1.0)
                nw = int(round(orig_w * r))
                nh = int(round(orig_h * r))
                dw = (new_w - nw) * 0.5
                dh = (new_h - nh) * 0.5
                top = int(round(dh - 0.1))
                bottom = int(round(dh + 0.1))
                left = int(round(dw - 0.1))
                right = int(round(dw + 0.1))

                x = t_u8.to(dtype=torch.float32).mul_(1.0 / 255.0).permute(2, 0, 1).unsqueeze(0)
                if orig_w != nw or orig_h != nh:
                    x = F_torch.interpolate(
                        x, size=(nh, nw), mode="bilinear", align_corners=False
                    )
                x = F_torch.pad(x, (left, right, top, bottom), value=pad_v)
                x = x[:, [2, 1, 0], :, :]
                blobs.append(x.squeeze(0))
                metas.append((r, (left, top), (orig_w, orig_h)))
            stacked = torch.stack(blobs, dim=0).contiguous()
        return stacked, metas

    def _resize_612_batch_cuda(
        self,
        bgr_imgs: list,
        *,
        device: Any,
        stream: Any,
    ) -> list:
        """在 GPU 上把 BGR 图 resize 到 612x512，并回传 uint8 numpy。"""
        if torch is None or F_torch is None:
            raise RuntimeError("未安装 torch，无法使用 GPU resize")
        out: list = []
        with torch.cuda.stream(stream):
            for img in bgr_imgs:
                t = torch.from_numpy(img).to(device=device, dtype=torch.uint8, non_blocking=False)
                x = t.to(dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                x = F_torch.interpolate(
                    x, size=(self.CA_RESIZE_H, self.CA_RESIZE_W), mode="bilinear", align_corners=False
                )
                x = x.round().clamp_(0.0, 255.0).to(dtype=torch.uint8).squeeze(0).permute(1, 2, 0)
                out.append(x.contiguous().cpu().numpy())
        return out

    def _resize_612_batch_cuda_from_gpu(
        self,
        bgr_imgs_u8_gpu: list,
        *,
        stream: Any,
    ) -> Any:
        """在 GPU 上把 BGR 图 resize 到 612x512，返回 GPU float32 NCHW 张量。

        若 batch 内各图 H×W 一致，则走单次 batched interpolate（例子 B），减少
        kernel 启动次数；尺寸不一致时回退逐张，语义与逐张路径一致。
        """
        if torch is None or F_torch is None:
            raise RuntimeError("未安装 torch，无法使用 GPU resize")
        if not bgr_imgs_u8_gpu:
            return torch.empty((0, 3, self.CA_RESIZE_H, self.CA_RESIZE_W))

        def _one(t_u8: Any) -> Any:
            x = t_u8.to(dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            x = F_torch.interpolate(
                x, size=(self.CA_RESIZE_H, self.CA_RESIZE_W), mode="bilinear", align_corners=False
            )
            return x.squeeze(0).contiguous()

        ref = bgr_imgs_u8_gpu[0]
        ref_hw = (int(ref.shape[0]), int(ref.shape[1]))
        same_hw = all(
            int(t.shape[0]) == ref_hw[0] and int(t.shape[1]) == ref_hw[1]
            for t in bgr_imgs_u8_gpu
        )

        with torch.cuda.stream(stream):
            if same_hw:
                batch_u8 = torch.stack(bgr_imgs_u8_gpu, dim=0)
                x = batch_u8.permute(0, 3, 1, 2).to(dtype=torch.float32)
                out = F_torch.interpolate(
                    x, size=(self.CA_RESIZE_H, self.CA_RESIZE_W), mode="bilinear", align_corners=False
                ).contiguous()
            else:
                out_list: list = []
                for t_u8 in bgr_imgs_u8_gpu:
                    out_list.append(_one(t_u8))
                out = torch.stack(out_list, dim=0).contiguous()
        return out

    def _meg_preprocess_batch_cuda(
        self,
        bgr_imgs: list,
        *,
        device: Any,
        stream: Any,
    ) -> tuple[Any, list]:
        """MEG 前处理在 GPU 完成，返回 torch.Tensor(N,3,512,640) + metas。"""
        if torch is None or F_torch is None:
            raise RuntimeError("未安装 torch，无法使用 GPU MEG 预处理")
        blobs: list = []
        metas: list = []
        with torch.cuda.stream(stream):
            for img in bgr_imgs:
                curr_h, curr_w = int(img.shape[0]), int(img.shape[1])
                r = self.MEG_SCALE_RATIO
                top = (self.MEG_INPUT_H - curr_h) // 2
                left = (self.MEG_INPUT_W - curr_w) // 2
                bottom = self.MEG_INPUT_H - curr_h - top
                right = self.MEG_INPUT_W - curr_w - left

                t = torch.from_numpy(img).to(device=device, dtype=torch.uint8, non_blocking=False)
                x = t.to(dtype=torch.float32).mul_(1.0 / 255.0).permute(2, 0, 1).unsqueeze(0)
                x = F_torch.pad(x, (left, right, top, bottom), value=114.0 / 255.0)
                x = x[:, [2, 1, 0], :, :]
                blobs.append(x.squeeze(0))
                metas.append((r, (left, top), (curr_w, curr_h)))
            stacked = torch.stack(blobs, dim=0).contiguous()
        return stacked, metas

    def _meg_preprocess_batch_cuda_from_gpu(
        self,
        bgr_612_f32_nchw: Any,
        *,
        stream: Any,
    ) -> tuple[Any, list]:
        """MEG 前处理在 GPU 完成（输入为 GPU float32 NCHW，值域 0..255）。"""
        if torch is None or F_torch is None:
            raise RuntimeError("未安装 torch，无法使用 GPU MEG 预处理")
        metas: list = []
        if int(bgr_612_f32_nchw.shape[0]) == 0:
            return torch.empty((0, 3, self.MEG_INPUT_H, self.MEG_INPUT_W)), metas
        curr_h = int(bgr_612_f32_nchw.shape[2])
        curr_w = int(bgr_612_f32_nchw.shape[3])
        r = self.MEG_SCALE_RATIO
        top = (self.MEG_INPUT_H - curr_h) // 2
        left = (self.MEG_INPUT_W - curr_w) // 2
        bottom = self.MEG_INPUT_H - curr_h - top
        right = self.MEG_INPUT_W - curr_w - left
        with torch.cuda.stream(stream):
            x = bgr_612_f32_nchw.mul(1.0 / 255.0)
            x = F_torch.pad(x, (left, right, top, bottom), value=114.0 / 255.0)
            stacked = x[:, [2, 1, 0], :, :].contiguous()
            metas = [(r, (left, top), (curr_w, curr_h)) for _ in range(int(stacked.shape[0]))]
        return stacked, metas

    def _constituency_preprocess_batch_cuda(
        self,
        bgr_imgs: list,
        *,
        device: Any,
        stream: Any,
    ) -> tuple[Any, list]:
        """Constituency 前处理在 GPU 完成，返回 torch.Tensor(N*4,3,224,224) + regions。"""
        if torch is None:
            raise RuntimeError("未安装 torch，无法使用 GPU Constituency 预处理")
        all_crops: list = []
        all_regions: list = []
        wi, hi = 2448.0 / 612.0, 2048.0 / 512.0
        mean = torch.tensor(self.CONST_MEAN, device=device, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.CONST_STD, device=device, dtype=torch.float32).view(3, 1, 1)
        with torch.cuda.stream(stream):
            for img in bgr_imgs:
                t = torch.from_numpy(img).to(device=device, dtype=torch.uint8, non_blocking=False)
                x = t.to(dtype=torch.float32).permute(2, 0, 1).mul_(1.0 / 255.0)
                x = (x - mean) / std
                for i in range(2):
                    for j in range(2):
                        xx, yy = j * 212, i * 200 + 88
                        crop = x[:, yy:yy + self.CONST_INPUT_H, xx:xx + self.CONST_INPUT_W]
                        all_crops.append(crop.contiguous())
                        all_regions.append([
                            int(xx * wi), int(yy * hi),
                            int(self.CONST_INPUT_W * wi), int(self.CONST_INPUT_H * hi),
                        ])
            stacked = torch.stack(all_crops, dim=0).contiguous()
        return stacked, all_regions

    def _constituency_preprocess_batch_cuda_from_gpu(
        self,
        bgr_612_f32_nchw: Any,
        *,
        stream: Any,
    ) -> tuple[Any, list]:
        """Constituency 前处理在 GPU 完成（输入为 GPU float32 NCHW，值域 0..255）。"""
        if torch is None:
            raise RuntimeError("未安装 torch，无法使用 GPU Constituency 预处理")
        all_regions: list = []
        wi, hi = 2448.0 / 612.0, 2048.0 / 512.0
        if int(bgr_612_f32_nchw.shape[0]) == 0:
            return torch.empty((0, 3, self.CONST_INPUT_H, self.CONST_INPUT_W)), all_regions
        device = bgr_612_f32_nchw.device
        mean = torch.tensor(self.CONST_MEAN, device=device, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.CONST_STD, device=device, dtype=torch.float32).view(3, 1, 1)
        with torch.cuda.stream(stream):
            x = bgr_612_f32_nchw.mul(1.0 / 255.0)
            x = (x - mean) / std

            # 固定 2x2 切块坐标，按 batch 批量拼接为 [N*4, C, H, W]
            xx0, xx1 = 0, 212
            yy0, yy1 = 88, 288
            c00 = x[:, :, yy0:yy0 + self.CONST_INPUT_H, xx0:xx0 + self.CONST_INPUT_W]
            c01 = x[:, :, yy0:yy0 + self.CONST_INPUT_H, xx1:xx1 + self.CONST_INPUT_W]
            c10 = x[:, :, yy1:yy1 + self.CONST_INPUT_H, xx0:xx0 + self.CONST_INPUT_W]
            c11 = x[:, :, yy1:yy1 + self.CONST_INPUT_H, xx1:xx1 + self.CONST_INPUT_W]
            stacked = torch.stack([c00, c01, c10, c11], dim=1).reshape(
                -1, 3, self.CONST_INPUT_H, self.CONST_INPUT_W
            ).contiguous()

            one_img_regions = [
                [int(xx0 * wi), int(yy0 * hi), int(self.CONST_INPUT_W * wi), int(self.CONST_INPUT_H * hi)],
                [int(xx1 * wi), int(yy0 * hi), int(self.CONST_INPUT_W * wi), int(self.CONST_INPUT_H * hi)],
                [int(xx0 * wi), int(yy1 * hi), int(self.CONST_INPUT_W * wi), int(self.CONST_INPUT_H * hi)],
                [int(xx1 * wi), int(yy1 * hi), int(self.CONST_INPUT_W * wi), int(self.CONST_INPUT_H * hi)],
            ]
            all_regions = one_img_regions * int(bgr_612_f32_nchw.shape[0])
        return stacked, all_regions

    def _wbc_postprocess_batch(self, outputs: np.ndarray, metas: list) -> list:
        results = []
        for i in range(outputs.shape[0]):
            r, pad, (orig_w, orig_h) = metas[i]
            pred = outputs[i].T
            scores = pred[:, 4:].max(axis=1)
            labels = pred[:, 4:].argmax(axis=1)
            mask = (scores > self.WBC_CONF_THR) & (labels == 0)
            if not mask.any():
                results.append(np.zeros((0, 5), dtype=np.float32))
                continue
            m_pred, m_scores = pred[mask], scores[mask]
            boxes = np.stack([
                m_pred[:, 0] - m_pred[:, 2] / 2,
                m_pred[:, 1] - m_pred[:, 3] / 2,
                m_pred[:, 0] + m_pred[:, 2] / 2,
                m_pred[:, 1] + m_pred[:, 3] / 2,
            ], axis=1)
            keep = self._nms_numpy(boxes, m_scores, self.WBC_IOU_THR)
            if len(keep) == 0:
                results.append(np.zeros((0, 5), dtype=np.float32))
                continue
            boxes, m_scores = boxes[keep], m_scores[keep]
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / r
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / r
            w = np.clip(boxes[:, 2] - boxes[:, 0], 0, None)
            h = np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
            area_mask = (w * h > self.WBC_AREA_MIN) & (w * h < self.WBC_AREA_MAX)
            results.append(np.stack([
                boxes[area_mask, 0], boxes[area_mask, 1],
                w[area_mask], h[area_mask], m_scores[area_mask],
            ], axis=1).astype(np.float32))
        return results

    # ═══════════════════════════════════════════════════════════════════
    # MEG
    # ═══════════════════════════════════════════════════════════════════
    def _meg_preprocess_batch(self, bgr_imgs: list) -> tuple:
        blobs = []
        metas = []
        for img in bgr_imgs:
            curr_h, curr_w = img.shape[:2]
            r = self.MEG_SCALE_RATIO

            top = (self.MEG_INPUT_H - curr_h) // 2
            left = (self.MEG_INPUT_W - curr_w) // 2
            bottom = self.MEG_INPUT_H - curr_h - top
            right = self.MEG_INPUT_W - curr_w - left

            padded = cv2.copyMakeBorder(
                img, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=(114, 114, 114),
            )
            blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blobs.append(blob.transpose(2, 0, 1))
            metas.append((r, (left, top), (curr_w, curr_h)))
        return np.ascontiguousarray(np.stack(blobs, axis=0)), metas

    def _meg_postprocess_batch(self, outputs: np.ndarray, metas: list) -> list:
        results = []
        for i in range(outputs.shape[0]):
            r, pad, (_, _) = metas[i]
            pred = outputs[i].T
            conf = pred[:, 4:].max(axis=1)
            mask = conf >= self.MEG_CONF_THR
            if not mask.any():
                results.append(np.zeros((0, 5), dtype=np.float32))
                continue
            m_boxes = pred[mask, :4]
            m_conf = conf[mask]
            boxes_xyxy = np.stack([
                m_boxes[:, 0] - m_boxes[:, 2] * 0.5,
                m_boxes[:, 1] - m_boxes[:, 3] * 0.5,
                m_boxes[:, 0] + m_boxes[:, 2] * 0.5,
                m_boxes[:, 1] + m_boxes[:, 3] * 0.5,
            ], axis=1)
            keep = self._nms_numpy_with_overlap(boxes_xyxy, m_conf, self.MEG_IOU_THR)
            boxes_xyxy = boxes_xyxy[keep[:self.MEG_MAX_DET]]
            m_conf = m_conf[keep[:self.MEG_MAX_DET]]
            boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad[0]) / r
            boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad[1]) / r
            w = np.clip(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], 0, None)
            h = np.clip(boxes_xyxy[:, 3] - boxes_xyxy[:, 1], 0, None)
            area_mask = (w * h > self.MEG_AREA_MIN) & (w * h < self.MEG_AREA_MAX)
            results.append(np.stack([
                boxes_xyxy[area_mask, 0], boxes_xyxy[area_mask, 1],
                w[area_mask], h[area_mask], m_conf[area_mask],
            ], axis=1).astype(np.float32))
        return results

    # ═══════════════════════════════════════════════════════════════════
    # Constituency
    # ═══════════════════════════════════════════════════════════════════
    def _constituency_preprocess_batch(self, bgr_imgs: list) -> tuple:
        all_crops, all_regions = [], []
        for img in bgr_imgs:
            norm_img = (img.astype(np.float32) / 255.0 - self.CONST_MEAN) / self.CONST_STD
            wi, hi = 2448.0 / 612.0, 2048.0 / 512.0
            for i in range(2):
                for j in range(2):
                    x, y = j * 212, i * 200 + 88
                    crop = norm_img[y:y + self.CONST_INPUT_H, x:x + self.CONST_INPUT_W]
                    all_crops.append(crop.transpose(2, 0, 1))
                    all_regions.append([
                        int(x * wi), int(y * hi),
                        int(self.CONST_INPUT_W * wi), int(self.CONST_INPUT_H * hi),
                    ])
        return np.ascontiguousarray(np.stack(all_crops, axis=0)), all_regions

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    @staticmethod
    def _compute_log_prior(scores: np.ndarray) -> np.ndarray:
        v = np.where(
            (scores > 45) & (scores <= 64),
            8.5 * scores * 0.001,
            np.where(
                (scores > 24) & (scores <= 45),
                5.5 * scores * 0.001,
                0.16875 * scores * 0.001,
            ),
        )
        return np.log(np.clip(v, 1e-10, None))

    def _constituency_postprocess_batch(
        self, output: np.ndarray, num_imgs: int, regions_list: list
    ) -> list:
        probs = self._softmax(output, axis=1)
        grades = probs.argmax(axis=1)
        weighted_scores = (probs * self.CONST_CLASS_DICT).sum(axis=1)
        direct_scores = self.CONST_CLASS_DICT[grades]
        scores = np.where(grades <= 2, weighted_scores, direct_scores)
        log_priors = self._compute_log_prior(scores)

        results = []
        for i in range(num_imgs):
            idx_s, idx_e = i * 4, (i + 1) * 4
            results.append((
                np.array(regions_list[idx_s:idx_e], dtype=np.int32),
                grades[idx_s:idx_e].astype(np.int32),
                log_priors[idx_s:idx_e].astype(np.float32),
            ))
        return results

    # ═══════════════════════════════════════════════════════════════════
    # Cell Analysis
    # ═══════════════════════════════════════════════════════════════════
    def _cell_analysis_preprocess_batch(self, bgr_imgs: list) -> np.ndarray:
        inputC, inputH, inputW = 4, self.CA_INPUT_H, self.CA_INPUT_W
        batchSize = self.CA_BATCH_SIZE

        blobs = []
        for img in bgr_imgs:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            white_mean_gray = float(np.mean(gray))
            if white_mean_gray < 160:
                flat = gray.ravel()
                kth = min(int(np.ceil(flat.size * 0.85)), flat.size - 1)
                highval = np.partition(flat, kth)[kth]
                white_mask = (gray >= highval).astype(np.uint8)
                white_mean = cv2.mean(img, mask=white_mask)[:3]
                img = img.astype(np.float32)
                for c in range(3):
                    scale = 250.0 / (float(white_mean[c]) + 1e-6)
                    img[:, :, c] = np.clip(img[:, :, c] * scale, 0, 255)
                img = img.astype(np.uint8)

            canvas = np.full((inputH, inputW, 3), self.CA_PAD_VALUE, dtype=np.uint8)
            canvas[0:self.CA_RESIZE_H, 0:self.CA_RESIZE_W] = img
            img = canvas

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            _, s, _ = cv2.split(hsv)
            b, g, r = cv2.split(img)
            img_4ch = np.stack([b, g, r, s], axis=-1).astype(np.float32) / 255.0
            blobs.append(img_4ch.transpose(2, 0, 1))

        if len(blobs) < batchSize:
            filler_blob = np.full(
                (inputC, inputH, inputW), self.CA_PAD_VALUE / 255.0, dtype=np.float32
            )
            # np.stack 会对每行拷贝，无需 filler_blob+0 触发整幅重复分配
            blobs.extend([filler_blob] * (batchSize - len(blobs)))
        return np.ascontiguousarray(np.stack(blobs, axis=0))

    def _cell_analysis_preprocess_batch_cuda(
        self,
        bgr_imgs: list,
        *,
        device: Any,
        stream: Any,
    ) -> Any:
        """Cell Analysis 前处理在 GPU 完成，返回 torch.Tensor(B,4,512,640)。"""
        if torch is None:
            raise RuntimeError("未安装 torch，无法使用 GPU Cell Analysis 预处理")
        input_c, input_h, input_w = 4, self.CA_INPUT_H, self.CA_INPUT_W
        batch_size = self.CA_BATCH_SIZE
        pad_val_u8 = float(self.CA_PAD_VALUE)
        pad_val_f32 = pad_val_u8 / 255.0
        thr = float(self.CA_PIXEL_THR)

        blobs: list = []
        with torch.cuda.stream(stream):
            for img in bgr_imgs:
                # t_u8: [H,W,3] BGR uint8
                t_u8 = torch.from_numpy(img).to(device=device, dtype=torch.uint8, non_blocking=False)
                t = t_u8.to(dtype=torch.float32)

                # gray = 0.114*B + 0.587*G + 0.299*R
                gray = 0.114 * t[:, :, 0] + 0.587 * t[:, :, 1] + 0.299 * t[:, :, 2]
                white_mean_gray = float(gray.mean().item())
                if white_mean_gray < 160.0:
                    flat = gray.reshape(-1)
                    # 与 CPU 的 85% 分位近似对齐
                    q = torch.quantile(flat, torch.tensor(0.85, device=device))
                    white_mask = (gray >= q)
                    if bool(white_mask.any().item()):
                        wm = white_mask.to(dtype=torch.float32)
                        den = wm.sum()
                        # B/G/R 三通道 white_mean
                        white_mean = (t * wm.unsqueeze(-1)).sum(dim=(0, 1)) / den
                        scale = 250.0 / (white_mean + 1e-6)
                        t = torch.clamp(t * scale.view(1, 1, 3), 0.0, 255.0)

                # pad 到 (512,640)
                canvas = torch.full((input_h, input_w, 3), pad_val_u8, device=device, dtype=torch.float32)
                canvas[0:self.CA_RESIZE_H, 0:self.CA_RESIZE_W, :] = t

                # HSV 的 S 通道: S = (max-min)/max (max>0)
                cmax = torch.max(canvas, dim=2).values
                cmin = torch.min(canvas, dim=2).values
                s = torch.zeros_like(cmax)
                nz = cmax > 0.0
                s[nz] = (cmax[nz] - cmin[nz]) / cmax[nz]

                # B,G,R,S 组 4 通道并归一化
                b = canvas[:, :, 0] * (1.0 / 255.0)
                g = canvas[:, :, 1] * (1.0 / 255.0)
                r = canvas[:, :, 2] * (1.0 / 255.0)
                img_4ch = torch.stack([b, g, r, s], dim=2)
                blobs.append(img_4ch.permute(2, 0, 1).contiguous())

            if len(blobs) < batch_size:
                filler_blob = torch.full(
                    (input_c, input_h, input_w),
                    pad_val_f32,
                    device=device,
                    dtype=torch.float32,
                )
                blobs.extend([filler_blob] * (batch_size - len(blobs)))
            stacked = torch.stack(blobs, dim=0).contiguous()
        return stacked

    def _cell_analysis_preprocess_batch_cuda_from_gpu(
        self,
        bgr_612_f32_nchw: Any,
        *,
        stream: Any,
    ) -> Any:
        """Cell Analysis 前处理在 GPU 完成（输入为 GPU float32 NCHW，值域 0..255）。"""
        if torch is None:
            raise RuntimeError("未安装 torch，无法使用 GPU Cell Analysis 预处理")
        input_c, input_h, input_w = 4, self.CA_INPUT_H, self.CA_INPUT_W
        batch_size = self.CA_BATCH_SIZE
        pad_val_u8 = float(self.CA_PAD_VALUE)
        pad_val_f32 = pad_val_u8 / 255.0

        if int(bgr_612_f32_nchw.shape[0]) == 0:
            raise RuntimeError("bgr_612_f32_nchw 不能为空")
        device = bgr_612_f32_nchw.device
        blobs: list = []
        with torch.cuda.stream(stream):
            for i in range(int(bgr_612_f32_nchw.shape[0])):
                # 按原 CA 语义先回到 uint8，再进入后续流程。
                t_u8 = bgr_612_f32_nchw[i].permute(1, 2, 0).round().clamp_(0.0, 255.0).to(dtype=torch.uint8)
                t = t_u8.to(dtype=torch.float32)
                gray = 0.114 * t[:, :, 0] + 0.587 * t[:, :, 1] + 0.299 * t[:, :, 2]
                white_mean_gray = float(gray.mean().item())
                if white_mean_gray < 160.0:
                    flat = gray.reshape(-1)
                    q = torch.quantile(flat, torch.tensor(0.85, device=device))
                    white_mask = (gray >= q)
                    if bool(white_mask.any().item()):
                        wm = white_mask.to(dtype=torch.float32)
                        den = wm.sum()
                        white_mean = (t * wm.unsqueeze(-1)).sum(dim=(0, 1)) / den
                        scale = 250.0 / (white_mean + 1e-6)
                        t = torch.clamp(t * scale.view(1, 1, 3), 0.0, 255.0)

                canvas = torch.full(
                    (input_h, input_w, 3), pad_val_u8, device=device, dtype=torch.float32
                )
                canvas[0:self.CA_RESIZE_H, 0:self.CA_RESIZE_W, :] = t

                cmax = torch.max(canvas, dim=2).values
                cmin = torch.min(canvas, dim=2).values
                s = torch.zeros_like(cmax)
                nz = cmax > 0.0
                s[nz] = (cmax[nz] - cmin[nz]) / cmax[nz]

                b = canvas[:, :, 0] * (1.0 / 255.0)
                g = canvas[:, :, 1] * (1.0 / 255.0)
                r = canvas[:, :, 2] * (1.0 / 255.0)
                img_4ch = torch.stack([b, g, r, s], dim=2)
                blobs.append(img_4ch.permute(2, 0, 1).contiguous())

            if len(blobs) < batch_size:
                filler_blob = torch.full(
                    (input_c, input_h, input_w),
                    pad_val_f32,
                    device=device,
                    dtype=torch.float32,
                )
                blobs.extend([filler_blob] * (batch_size - len(blobs)))
            stacked = torch.stack(blobs, dim=0).contiguous()
        return stacked

    def _cell_analysis_postprocess_and_count(self, output: np.ndarray, orig_sizes: list):
        outH, outW = self.CA_RESIZE_H, self.CA_RESIZE_W
        num_valid = len(orig_sizes)
        counts_wbc = np.zeros(num_valid, dtype=np.int32)
        counts_red = np.zeros(num_valid, dtype=np.int32)
        base_area = float(outH * outW)
        for b in range(num_valid):
            out = output[b]
            g_ch, b_ch, r_ch = out[0], out[1], out[2]
            result = np.stack([b_ch, g_ch, r_ch], axis=-1)
            result = result[:outH, :outW] * 255.0
            result = np.clip(result, 0, 255).astype(np.uint8)
            b_channel = result[:, :, 0]
            r_channel = result[:, :, 2]
            thr = int(self.CA_PIXEL_THR)
            raw_wbc = int(np.sum(b_channel > thr))
            raw_red = int(np.sum(r_channel > thr))
            orig_h, orig_w = orig_sizes[b]
            ratio = float(orig_h * orig_w) / base_area if base_area > 0 else 1.0
            counts_wbc[b] = int(np.rint(raw_wbc * ratio))
            counts_red[b] = int(np.rint(raw_red * ratio))
        return counts_wbc, counts_red

    # ═══════════════════════════════════════════════════════════════════
    # NMS
    # ═══════════════════════════════════════════════════════════════════
    @staticmethod
    def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
        if boxes.shape[0] == 0:
            return np.array([], dtype=np.int64)
        if _USE_NUMBA_NMS:
            return _nms_numpy_numba(
                np.ascontiguousarray(boxes, dtype=np.float32),
                np.ascontiguousarray(scores, dtype=np.float32),
                float(iou_thr),
            )
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[order[1:]], x1[i])
            yy1 = np.maximum(y1[order[1:]], y1[i])
            xx2 = np.minimum(x2[order[1:]], x2[i])
            yy2 = np.minimum(y2[order[1:]], y2[i])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[order[1:]] + areas[i] - inter + 1e-7)
            inds = np.where(iou <= iou_thr)[0]
            order = order[inds + 1]
        return np.array(keep, dtype=np.int64)

    @staticmethod
    def _nms_numpy_with_overlap(
        boxes: np.ndarray, scores: np.ndarray,
        iou_thr: float, overlap_thr: float = 0.7,
    ) -> np.ndarray:
        if boxes.shape[0] == 0:
            return np.array([], dtype=np.int64)
        if _USE_NUMBA_NMS:
            return _nms_overlap_numba(
                np.ascontiguousarray(boxes, dtype=np.float32),
                np.ascontiguousarray(scores, dtype=np.float32),
                float(iou_thr),
                float(overlap_thr),
            )
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        suppressed = np.zeros(boxes.shape[0], dtype=bool)
        for idx in order:
            if suppressed[idx]:
                continue
            keep.append(idx)
            xx1 = np.maximum(x1, x1[idx])
            yy1 = np.maximum(y1, y1[idx])
            xx2 = np.minimum(x2, x2[idx])
            yy2 = np.minimum(y2, y2[idx])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas + areas[idx] - inter + 1e-7)
            overlap = inter / (np.minimum(areas, areas[idx]) + 1e-7)
            suppressed |= (iou > iou_thr) | (overlap > overlap_thr)
            suppressed[idx] = False
        return np.array(keep, dtype=np.int64)


# ────────────────────────────────────────────────────────────────────────
# async_infer 的轻量包装：event-based wait，返回 InferResult
# ────────────────────────────────────────────────────────────────────────
class _AsyncCall:
    __slots__ = ("_event", "_result", "_error", "_model")

    def __init__(self, model: str):
        self._event = threading.Event()
        self._result: Any = None
        self._error: Exception | None = None
        self._model = model

    def _callback(self, result, error):
        if error is not None:
            self._error = error
        else:
            self._result = result
        self._event.set()

    def wait(self, timeout: float | None = None):
        if not self._event.wait(timeout=timeout):
            raise TimeoutError(f"async_infer timed out: {self._model}")
        if self._error is not None:
            raise RuntimeError(f"{self._model} inference error: {self._error}")
        return self._result


def _async_infer(
    client: grpcclient.InferenceServerClient,
    model_name: str,
    inputs: list[tuple[str, np.ndarray, str]],
    outputs: list[str],
) -> _AsyncCall:
    """用 async_infer 发起请求，返回 _AsyncCall，调 .wait() 拿 InferResult。

    inputs: list of (name, ndarray, datatype_str)
    """
    grpc_inputs = []
    for name, arr, dtype in inputs:
        arr = np.ascontiguousarray(arr)
        inp = grpcclient.InferInput(name, list(arr.shape), dtype)
        inp.set_data_from_numpy(arr)
        grpc_inputs.append(inp)
    grpc_outputs = [grpcclient.InferRequestedOutput(n) for n in outputs]

    call = _AsyncCall(model_name)
    client.async_infer(
        model_name=model_name,
        inputs=grpc_inputs,
        outputs=grpc_outputs,
        callback=call._callback,
    )
    return call
