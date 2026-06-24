"""
DPI147246_BM_PB_pipeline 的客户端编排 **批量 + CUDA 共享内存** 版本。

与 `dpi147246_local_pipeline.LocalWbcMegPipeline` 对比：
    - 不再"一张图走一次 pipeline"，而是把多个并发请求在客户端 coalesce
      成 batch（默认 4 张或 20ms 超时）再发给 Triton；
    - 所有子模型的输入/输出张量通过 **CUDA shared memory** 交给 Triton，
      彻底消除 gRPC 大张量序列化和 TCP/loopback 拷贝；
    - Cell Analysis 仍然固定 batch=4，但真实图片不足 4 张时才用 filler
      补位；凑够 4 张不做任何填充，GPU 算力和 SHM 带宽都不浪费。
    - `use_gpu_preprocess_mode2=True` 时，预处理输出经
      `set_shared_memory_region_from_dlpack` **D2D 写入** CUDA SHM；缺少该 API
      的 tritonclient 版本会在启动时直接报错（不回退 numpy）。
    - WBC letterbox 可选在**本进程**用 torch.cuda 完成（`use_gpu_wbc_preprocess=True`），
      与 MEG/CON/CA 的 CPU 预处理线程池重叠。默认 **关闭**（走 OpenCV CPU）：
      当前实现最后仍要把张量 `cpu().numpy()` 做大块 D2H 再写 CUDA SHM，与原先
      在 CPU 上算好再 H2D 往往总耗时接近甚至更差；真正提速需后续做 **D2D 直写
      SHM** 等改造。

前提：
    - 客户端进程能看到同一块 GPU（tritonclient 会在当前进程用 cuda-python
      /pycuda 起一个 CUDA context，调 cuIpcGetMemHandle）；
    - Triton server 的 `ipc: host` 已开，能接受 CUDA IPC handle；
    - 4 个子模型 engine 的 I/O 类型为 FP32（通过 `curl /v2/models/.../config`
      已确认），所以 SHM 区按 FP32 分配。

对外 API：
    pipeline = BatchedCudaWbcMegPipeline(url, enable_meg=True)
    result = pipeline.submit(image_bytes)
    # result 结构与 `infer_wbc_meg` 完全一致，可直接替换
"""

from __future__ import annotations

import atexit
import logging
import signal
import sys
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from queue import Queue
from typing import Any

import numpy as np
import tritonclient.grpc as grpcclient

try:
    import tritonclient.utils.cuda_shared_memory as cudashm
except ImportError as e:  # pragma: no cover - 友好报错
    raise ImportError(
        "tritonclient.utils.cuda_shared_memory 不可用。\n"
        "请在当前进程/容器里安装带 cuda 扩展的 tritonclient：\n"
        "    pip install 'tritonclient[cuda]'\n"
        "同时确认容器内能看到 GPU（nvidia-smi 可用）。"
    ) from e

import cv2  # type: ignore

from dpi147246_cpu_profile import ACTIVE_PROFILE, DEFAULT_KWARGS
from dpi147246_local_pipeline import (
    LocalWbcMegPipeline,
    MODEL_WBC,
    MODEL_MEG,
    MODEL_CON,
    MODEL_CA,
    wbc_torch_cuda_available,
)
from p147246_lifecycle import (
    Trace,
    set_batch,
    set_result,
    stage_end,
    stage_start,
)

_PROFILE_DEFAULTS = DEFAULT_KWARGS
logger = logging.getLogger("dpi147246_cuda_pipeline")


def _item_trace(item: tuple) -> Trace | None:
    return item[4] if len(item) > 4 else None


def _stage_start_items(items: list, name: str, *, scope: str = "batch") -> None:
    for item in items:
        stage_start(_item_trace(item), name, scope=scope)


def _stage_end_items(items: list, name: str) -> None:
    for item in items:
        stage_end(_item_trace(item), name)


# ─────────────────────────────────────────────────────────────────────────
# SHM 区尺寸计算：用 FP32 上界预分配，覆盖满 batch（CA 固定 4，其他取 max_batch）
# ─────────────────────────────────────────────────────────────────────────
FP32_BYTES = 4

# tritonclient.utils.cuda_shared_memory 不是线程安全：多线程同时
# set_shared_memory_region / get_contents_as_numpy 会触发 libcuda 段错误。
# 用单 worker 的 ThreadPoolExecutor 把所有 CUDA SHM 读写串到同一线程（方案 A）。
_SHM_IO_EXECUTOR: ThreadPoolExecutor | None = None
_SHM_IO_EXEC_LOCK = threading.Lock()


def _get_cuda_shm_io_executor() -> ThreadPoolExecutor:
    global _SHM_IO_EXECUTOR
    with _SHM_IO_EXEC_LOCK:
        if _SHM_IO_EXECUTOR is None:
            _SHM_IO_EXECUTOR = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="cuda_shm_io",
            )
        return _SHM_IO_EXECUTOR


def _shutdown_cuda_shm_io_executor() -> None:
    """在释放 SHM 区之前调用，确保不再有 in-flight 的 cudaMemcpy。"""
    global _SHM_IO_EXECUTOR
    with _SHM_IO_EXEC_LOCK:
        if _SHM_IO_EXECUTOR is not None:
            _SHM_IO_EXECUTOR.shutdown(wait=True, cancel_futures=False)
            _SHM_IO_EXECUTOR = None


# WBC/MEG 后处理里大量 Python/Numpy（含 NMS）。多 slot 若用同进程多线程并行，
# CPython GIL 下吞吐可能崩溃；``heavy_post_mode=serial`` 时用此锁全局串行。
# ``process`` 模式在子进程跑 NMS，可与多 slot 真并行且结果一致。
_HEAVY_POST_LOCK = threading.Lock()
_HEAVY_POST_PROC_HELPER: LocalWbcMegPipeline | None = None


def _init_heavy_post_proc_worker() -> None:
    global _HEAVY_POST_PROC_HELPER
    _HEAVY_POST_PROC_HELPER = LocalWbcMegPipeline()


def _heavy_post_proc_task(
    wbc_out: np.ndarray,
    wbc_meta: list,
    meg_out: np.ndarray | None,
    meg_meta: list | None,
    run_meg: bool,
    n: int,
) -> tuple[list, list]:
    h = _HEAVY_POST_PROC_HELPER
    if h is None:
        raise RuntimeError("heavy post worker 未初始化")
    wbc = h._wbc_postprocess_batch(wbc_out, wbc_meta)
    if run_meg and meg_out is not None and meg_meta is not None:
        meg = h._meg_postprocess_batch(meg_out, meg_meta)
    else:
        meg = [np.zeros((0, 5), dtype=np.float32) for _ in range(n)]
    return wbc, meg


def _prod(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _bytes_fp32(shape: list[int]) -> int:
    return _prod(shape) * FP32_BYTES


class _ShmRegion:
    """一块 CUDA 共享内存区域。负责分配、向 Triton 注册、释放。"""

    __slots__ = ("name", "byte_size", "handle", "registered", "_client")

    def __init__(
        self,
        client: grpcclient.InferenceServerClient,
        name: str,
        byte_size: int,
        device_id: int,
    ):
        self.name = name
        self.byte_size = byte_size
        self._client = client
        # 分配显存 + 拿 CUDA IPC handle
        self.handle = cudashm.create_shared_memory_region(name, byte_size, device_id)
        # 注册到 Triton 服务端
        try:
            client.register_cuda_shared_memory(
                name, cudashm.get_raw_handle(self.handle), device_id, byte_size
            )
        except Exception:
            cudashm.destroy_shared_memory_region(self.handle)
            raise
        self.registered = True

    def write_numpy(self, arr: np.ndarray) -> int:
        """把 numpy 数组写入 SHM，返回实际使用的字节数。

        实际 cudaMemcpy 在专用 I/O 线程执行（见 `_get_cuda_shm_io_executor`），
        调用方线程只负责准备 contiguous 数组并阻塞等待完成。
        """
        arr = np.ascontiguousarray(arr)
        assert arr.dtype == np.float32, f"期望 float32, 实际 {arr.dtype}"
        used = arr.nbytes
        if used > self.byte_size:
            raise RuntimeError(
                f"SHM 区 {self.name} 容量 {self.byte_size} 不足以写入 {used} 字节"
            )
        _get_cuda_shm_io_executor().submit(
            cudashm.set_shared_memory_region, self.handle, [arr]
        ).result()
        return used

    def read_numpy(self, dtype: np.dtype, shape: list[int]) -> np.ndarray:
        return _get_cuda_shm_io_executor().submit(
            cudashm.get_contents_as_numpy, self.handle, dtype, shape
        ).result()

    def close(self) -> None:
        if self.registered:
            try:
                self._client.unregister_cuda_shared_memory(self.name)
            except Exception as e:
                logger.warning(
                    "[CudaShmRegion] unregister %s 失败: %s；Triton 端可能残留注册，显存无法回收",
                    self.name,
                    e,
                )
            self.registered = False
        if self.handle is not None:
            try:
                cudashm.destroy_shared_memory_region(self.handle)
            except Exception as e:
                logger.warning(
                    "[CudaShmRegion] destroy %s 失败: %s；GPU 显存块可能泄漏直到进程退出",
                    self.name,
                    e,
                )
            self.handle = None


class _BatchSlot:
    """一个"合批处理位"：8 块 SHM 区（4 模型 × 输入/输出）。"""

    __slots__ = ("slot_id", "slot_ix", "wbc_in", "wbc_out", "meg_in", "meg_out",
                 "con_in", "con_out", "ca_in", "ca_out")

    def __init__(
        self,
        slot_id: str,
        slot_ix: int,
        wbc_in: _ShmRegion, wbc_out: _ShmRegion,
        meg_in: _ShmRegion, meg_out: _ShmRegion,
        con_in: _ShmRegion, con_out: _ShmRegion,
        ca_in: _ShmRegion, ca_out: _ShmRegion,
    ):
        self.slot_id = slot_id
        self.slot_ix = slot_ix
        self.wbc_in, self.wbc_out = wbc_in, wbc_out
        self.meg_in, self.meg_out = meg_in, meg_out
        self.con_in, self.con_out = con_in, con_out
        self.ca_in, self.ca_out = ca_in, ca_out

    def regions(self) -> list[_ShmRegion]:
        return [self.wbc_in, self.wbc_out, self.meg_in, self.meg_out,
                self.con_in, self.con_out, self.ca_in, self.ca_out]


def _shm_host_input_view(arr: np.ndarray, region: _ShmRegion, label: str) -> np.ndarray:
    """供 H2D 写入：仅在 dtype 非 FP32 或非 C 连续时拷贝，避免 ascontiguousarray 全量重复。"""
    if arr.dtype != np.float32:
        arr = np.asarray(arr, dtype=np.float32, order="C")
    elif not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    if arr.nbytes > region.byte_size:
        raise RuntimeError(
            f"SHM 区 {region.name} ({label}) 容量 {region.byte_size} 不足以写入 {arr.nbytes} 字节"
        )
    return arr


def _shm_io_write_slot_batch(
    slot: _BatchSlot,
    wbc_np: np.ndarray,
    con_np: np.ndarray,
    ca_np: np.ndarray,
    meg_np: np.ndarray | None,
    run_meg: bool,
) -> None:
    """在专用 I/O 线程里一次写完本 slot 全部输入，减少 submit 排队次数。"""
    wbc_np = _shm_host_input_view(wbc_np, slot.wbc_in, "wbc_in")
    con_np = _shm_host_input_view(con_np, slot.con_in, "con_in")
    ca_np = _shm_host_input_view(ca_np, slot.ca_in, "ca_in")
    if run_meg:
        if meg_np is None:
            raise RuntimeError("run_meg=True 但 meg_np 为 None")
        meg_np = _shm_host_input_view(meg_np, slot.meg_in, "meg_in")

    def _do() -> None:
        cudashm.set_shared_memory_region(slot.wbc_in.handle, [wbc_np])
        cudashm.set_shared_memory_region(slot.con_in.handle, [con_np])
        cudashm.set_shared_memory_region(slot.ca_in.handle, [ca_np])
        if run_meg:
            cudashm.set_shared_memory_region(slot.meg_in.handle, [meg_np])

    _get_cuda_shm_io_executor().submit(_do).result()


def _shm_io_write_slot_batch_dlpack(
    slot: _BatchSlot,
    wbc_t: Any,
    con_t: Any,
    ca_t: Any,
    meg_t: Any | None,
    run_meg: bool,
) -> None:
    """将已在 GPU 上的 FP32 contiguous 张量 D2D 写入 CUDA SHM（无 CPU numpy 中转）。

    依赖 `cudashm.set_shared_memory_region_from_dlpack`（构造 pipeline 时已校验存在）。
    """
    import torch

    fn = cudashm.set_shared_memory_region_from_dlpack

    def _as_dlpack_c_order(t: Any) -> Any:
        # PyTorch>=1.13 在 __dlpack__() 导出时会把 size==1 维度的 stride 归一化为 1，
        # 与 C-order 不符，导致 tritonclient 的连续性校验误判失败（容器内外 torch
        # 版本不同即表现不同）。SHM 写入只拷贝字节、形状由 InferInput 单独指定，
        # 因此展平成 1-D 连续张量：既保证字节序为 C-order，又彻底消除 size==1 维度
        # 的 stride 归一化问题，与 torch / tritonclient 版本无关。
        if not t.is_contiguous():
            t = t.contiguous()
        return t.reshape(-1)

    def _check(t: Any, region: _ShmRegion, label: str) -> Any:
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{label} 期望 torch.Tensor，实际 {type(t)}")
        if not t.is_cuda:
            raise RuntimeError(f"{label} 必须在 CUDA 上")
        if t.dtype != torch.float32:
            raise RuntimeError(f"{label} 期望 float32，实际 {t.dtype}")
        t = _as_dlpack_c_order(t.detach())
        nb = int(t.numel()) * FP32_BYTES
        if nb > int(region.byte_size):
            raise RuntimeError(
                f"SHM 区 {region.name} ({label}) 容量 {region.byte_size} "
                f"不足以写入 {nb} 字节"
            )
        return t

    wbc_t = _check(wbc_t, slot.wbc_in, "wbc_in")
    con_t = _check(con_t, slot.con_in, "con_in")
    ca_t = _check(ca_t, slot.ca_in, "ca_in")
    if run_meg:
        if meg_t is None:
            raise RuntimeError("run_meg=True 但 meg_t 为 None")
        meg_t = _check(meg_t, slot.meg_in, "meg_in")

    def _do() -> None:
        fn(slot.wbc_in.handle, [wbc_t])
        fn(slot.con_in.handle, [con_t])
        fn(slot.ca_in.handle, [ca_t])
        if run_meg:
            fn(slot.meg_in.handle, [meg_t])

    _get_cuda_shm_io_executor().submit(_do).result()


def _shm_io_read_slot_batch(
    slot: _BatchSlot,
    wbc_shape: list[int],
    con_shape: list[int],
    ca_shape: list[int],
    meg_shape: list[int] | None,
    run_meg: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """在专用 I/O 线程里一次读完本 slot 全部输出。"""

    def _do() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        wbc = cudashm.get_contents_as_numpy(slot.wbc_out.handle, np.float32, wbc_shape)
        con = cudashm.get_contents_as_numpy(slot.con_out.handle, np.float32, con_shape)
        ca = cudashm.get_contents_as_numpy(slot.ca_out.handle, np.float32, ca_shape)
        meg: np.ndarray | None = None
        if run_meg and meg_shape is not None:
            meg = cudashm.get_contents_as_numpy(slot.meg_out.handle, np.float32, meg_shape)
        return wbc, con, ca, meg

    return _get_cuda_shm_io_executor().submit(_do).result()


# ─────────────────────────────────────────────────────────────────────────
# 轻量级阶段计时器 + 聚合统计
# ─────────────────────────────────────────────────────────────────────────
class _StageTimer:
    """单 batch 内多次 .mark(name) 累加各阶段耗时。"""

    __slots__ = ("_t", "stages")

    def __init__(self):
        self._t = time.perf_counter()
        self.stages: dict[str, float] = {}

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.stages[name] = self.stages.get(name, 0.0) + (now - self._t) * 1000.0
        self._t = now

    def total_ms(self) -> float:
        return sum(self.stages.values())


# 阶段顺序（影响打印顺序，不影响正确性）
_STAGE_ORDER = (
    "wait_slot",
    "decode",
    "cpu_to_gpu_upload",
    "resize_612",
    "mode2_gpu_preprocess",
    "wbc_pre",
    "meg_pre",
    "con_pre",
    "ca_pre",
    "shm_write",
    "infer_launch",
    "infer_wait",
    "shm_read",
    "wbc_post",
    "meg_post",
    "con_post",
    "ca_post",
    "dispatch",
)


# ─────────────────────────────────────────────────────────────────────────
# async_infer 包装（同 dpi147246_local_pipeline，但返回 InferResult 供读
# metadata；输出数据通过 SHM 直接读取，不从 InferResult 拷贝）
# ─────────────────────────────────────────────────────────────────────────
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
            raise TimeoutError(f"async_infer 超时: {self._model}")
        if self._error is not None:
            raise RuntimeError(f"{self._model} 推理错误: {self._error}")
        return self._result


# ─────────────────────────────────────────────────────────────────────────
# 主类
# ─────────────────────────────────────────────────────────────────────────
class BatchedCudaWbcMegPipeline:
    """DPI147246 客户端编排 · batch=4 coalesce · CUDA SHM 版。

    - 所有工作线程共享本实例（单例即可）。
    - `submit(image_bytes)` 线程安全，阻塞到本张图片的结果就绪后返回。
    - 合批策略：凑满 `batch_size` 张立刻发；若 `timeout_ms` 内没凑满，
      以当前已攒的数量发出（CA 模型若不足 4 张才用 filler 补齐）。
    """

    # 子模型各自的 max_batch（与 config.pbtxt 一致）
    WBC_MAX_BATCH = 4
    MEG_MAX_BATCH = 4
    CON_MAX_BATCH = 16           # = batch_size(4) * 4 crops / 张
    CA_FIXED_BATCH = 4           # CA 固定 batch=4（max_batch_size: 0）

    def __init__(
        self,
        url: str,
        *,
        batch_size: int = _PROFILE_DEFAULTS["batch_size"],
        timeout_ms: int = _PROFILE_DEFAULTS["timeout_ms"],
        num_slots: int = _PROFILE_DEFAULTS["num_slots"],
        enable_meg: bool = True,
        enable_ca: bool = _PROFILE_DEFAULTS["enable_ca"],
        device_id: int = 0,
        verbose: bool = False,
        parallel_preprocess: bool = True,
        preprocess_workers: int = _PROFILE_DEFAULTS["preprocess_workers"],
        decode_workers: int | None = None,
        heavy_post_mode: str | None = None,
        heavy_post_workers: int | None = None,
        use_gpu_wbc_preprocess: bool | None = None,
        use_gpu_preprocess_mode2: bool = True,
        purge_stale_cuda_shm: bool = False,
        decode_only_return_empty: bool = False,
        cpu_to_gpu_only_return_empty: bool = False,
        preprocess_only_return_empty: bool = False,
        shm_write_only_return_empty: bool = False,
        coalesce_log_path: str | None = None,
    ):
        if batch_size <= 0 or batch_size > self.WBC_MAX_BATCH:
            raise ValueError(
                f"batch_size 必须在 (0, {self.WBC_MAX_BATCH}]，当前 {batch_size}"
            )
        if enable_ca and batch_size != self.CA_FIXED_BATCH:
            # 只有真实图 == 4 时 CA 才能"不填充"；非 4 的 batch size 相当于总是
            # 要给 CA 塞 filler，这里不支持以免用户踩坑。
            raise ValueError(
                "目前 batch_size 必须等于 4（与 CA engine 固定 batch 对齐）"
            )

        self._url = url
        self._batch_size = batch_size
        self._timeout_s = timeout_ms / 1000.0
        self._num_slots = num_slots
        self._enable_meg = enable_meg
        self._enable_ca = bool(enable_ca)
        self._device_id = device_id
        self._verbose = verbose
        self._parallel_preprocess = parallel_preprocess
        # 临时调试开关：仅做 decode，随后直接返回空结果，不执行后续推理/后处理。
        self._decode_only_return_empty = bool(decode_only_return_empty)
        # 临时调试开关：仅在 mode2 下生效；decode + CPU->GPU 上传后直接返回空结果。
        self._cpu_to_gpu_only_return_empty = bool(cpu_to_gpu_only_return_empty)
        # 临时调试开关：decode + 全部前处理做完后返回空结果，不执行后续推理/后处理。
        self._preprocess_only_return_empty = bool(preprocess_only_return_empty)
        # 临时调试开关：decode + 前处理 + SHM 写入后返回空结果，不执行后续推理/后处理。
        self._shm_write_only_return_empty = bool(shm_write_only_return_empty)
        self._use_gpu_preprocess_mode2 = bool(use_gpu_preprocess_mode2)
        self._cpu_profile = ACTIVE_PROFILE
        # 预处理 / SHM 写入并发线程池；多个 dispatcher 线程共享。
        # cv2 / numpy / cudaMemcpy 都释放 GIL，所以 thread pool 是合理的选择。
        self._pp_pool = ThreadPoolExecutor(
            max_workers=max(1, preprocess_workers),
            thread_name_prefix="dpi147246_pp",
        )
        # JPEG 解码单独池，避免与 batch 内其它预处理任务抢同一线程（3970X 上解码占带宽）
        _dec_w = (
            _PROFILE_DEFAULTS["decode_workers"]
            if decode_workers is None
            else int(decode_workers)
        )
        self._decode_pool: ThreadPoolExecutor | None = None
        if parallel_preprocess and _dec_w > 0:
            self._decode_pool = ThreadPoolExecutor(
                max_workers=max(1, _dec_w),
                thread_name_prefix="dpi147246_dec",
            )

        _hp_mode = (
            heavy_post_mode
            if heavy_post_mode is not None
            else str(_PROFILE_DEFAULTS.get("heavy_post_mode", "serial"))
        )
        if _hp_mode not in ("serial", "thread", "process"):
            raise ValueError(
                f"heavy_post_mode 必须是 serial|thread|process，当前 {_hp_mode!r}"
            )
        self._heavy_post_mode = _hp_mode
        _hp_w = (
            int(heavy_post_workers)
            if heavy_post_workers is not None
            else int(_PROFILE_DEFAULTS.get("heavy_post_workers", num_slots))
        )
        self._heavy_post_workers = max(1, _hp_w)
        self._heavy_post_proc_pool: ProcessPoolExecutor | None = None
        self._heavy_post_thread_pool: ThreadPoolExecutor | None = None
        if self._heavy_post_mode == "process":
            self._heavy_post_proc_pool = ProcessPoolExecutor(
                max_workers=self._heavy_post_workers,
                initializer=_init_heavy_post_proc_worker,
            )
        elif self._heavy_post_mode == "thread":
            self._heavy_post_thread_pool = ThreadPoolExecutor(
                max_workers=self._heavy_post_workers,
                thread_name_prefix="dpi147246_hpost",
            )

        if use_gpu_wbc_preprocess is False:
            self._use_gpu_wbc_pre = False
        elif use_gpu_wbc_preprocess is True:
            if not wbc_torch_cuda_available():
                raise RuntimeError(
                    "use_gpu_wbc_preprocess=True 需要已安装 torch 且 torch.cuda.is_available()"
                )
            self._use_gpu_wbc_pre = True
        else:
            # 默认不启用：GPU 版仍含 sync + 大块 D2H，压测多未缩短 wbc_pre
            self._use_gpu_wbc_pre = False
        if self._use_gpu_preprocess_mode2:
            if not wbc_torch_cuda_available():
                raise RuntimeError(
                    "use_gpu_preprocess_mode2=True 需要已安装 torch 且 torch.cuda.is_available()"
                )
            if getattr(cudashm, "set_shared_memory_region_from_dlpack", None) is None:
                raise RuntimeError(
                    "use_gpu_preprocess_mode2=True 需要 tritonclient 提供 "
                    "tritonclient.utils.cuda_shared_memory.set_shared_memory_region_from_dlpack "
                    "（用于 GPU 张量 D2D 写入 CUDA SHM，无此 API 请升级 tritonclient，"
                    "例如 pip install -U 'tritonclient[cuda]'，并安装 cuda-python）。"
                )
            # mode2 里 WBC/MEG/CON 都走 GPU 前处理，WBC 路径必须开启
            self._use_gpu_wbc_pre = True

        self._torch_device: Any = None
        self._wbc_streams: list[Any] = []
        self._wbc_stream_next = 0
        self._wbc_stream_lock = threading.Lock()
        if self._use_gpu_wbc_pre:
            import torch

            torch.cuda.set_device(device_id)
            self._torch_device = torch.device("cuda", device_id)
            self._wbc_streams = [
                torch.cuda.Stream(device=self._torch_device)
                for _ in range(num_slots)
            ]
            if verbose:
                logger.info("[CudaPipeline] WBC letterbox 使用 GPU (torch %s)", self._torch_device)

        if verbose:
            logger.info(
                "[CudaPipeline] cpu_profile=%s timeout_ms=%s num_slots=%s preprocess_workers=%s "
                "decode_workers=%s enable_ca=%s heavy_post_mode=%s heavy_post_workers=%s",
                self._cpu_profile,
                timeout_ms,
                num_slots,
                preprocess_workers,
                _dec_w,
                self._enable_ca,
                self._heavy_post_mode,
                self._heavy_post_workers,
            )

        self._client = grpcclient.InferenceServerClient(url=url, verbose=False)

        # 共用的预/后处理方法（先于任何用到 _helper 的步骤实例化）
        self._helper = LocalWbcMegPipeline()

        # 注意：双实例/多实例并发时，不能做全局 purge（会误删其它实例在用的 region）。
        # 仅在显式要求时执行，默认关闭。
        if purge_stale_cuda_shm:
            # 启动前先把 Triton 端残留的 CUDA SHM 注册全部清掉。
            # 上次运行如果异常退出（Ctrl+C / kill / 崩溃），Triton 端会残留一批
            # 注册，对应的 GPU 显存也会被 hold 住。干净启动能避免滚雪球泄漏。
            self._purge_stale_cuda_shm_registrations()

        # 确保所有依赖模型都已 load
        self._ensure_models_loaded()

        # 读 output shape（输入 shape 我们自己算，输出 shape 必须从 metadata 取）
        self._out_shapes = self._probe_output_shapes()

        # 计算每块 SHM 区需要的字节数（按 max batch 取上界）
        self._region_bytes = self._compute_region_bytes()

        # 分配 slot 池
        self._slots: list[_BatchSlot] = []
        for i in range(num_slots):
            self._slots.append(
                self._alloc_slot(slot_id=f"{uuid.uuid4().hex[:8]}_{i}", slot_ix=i)
            )
        self._slot_q: Queue[_BatchSlot] = Queue()
        for s in self._slots:
            self._slot_q.put(s)

        # 预热专用 CUDA SHM I/O 单线程（所有 memcpy 经此线程，避免多线程段错误）
        _get_cuda_shm_io_executor()

        # batch coalescer 状态
        self._co_lock = threading.Lock()
        self._pending: list = []
        self._batch_seq = 0

        # 性能统计（按阶段累加）
        self._stats_lock = threading.Lock()
        self._stats_count = 0          # batch 数
        self._stats_imgs = 0           # 真实图片数（不含 CA filler）
        self._stats_wbc_total = 0      # 累计 WBC 检测数
        self._stats_meg_batches = 0    # 跑了 MEG 的 batch 数
        self._stats_total_ms = 0.0     # batch 端到端总耗时（含 wait_slot）
        self._stats_stage_sum: dict[str, float] = {s: 0.0 for s in _STAGE_ORDER}
        self._stats_stage_count: dict[str, int] = {s: 0 for s in _STAGE_ORDER}
        # 合批：凑满 batch_size 立即发出 vs 超时 flush；以及按实际张数 n 的分布
        self._coalesce_full = 0
        self._coalesce_timeout = 0
        self._coalesce_timeout_n1 = 0
        self._coalesce_by_n: dict[int, int] = {}
        if coalesce_log_path:
            logger.warning("[CudaPipeline] coalesce_log_path 已废弃，合批日志统一写入服务日志文件")

        self._closed = False
        atexit.register(self.close)
        self._install_signal_handlers()

    def _purge_stale_cuda_shm_registrations(self) -> None:
        """让 Triton 反注册它记录的所有 CUDA SHM（不分 name）。

        注：这只清理 Triton 端的"记账"和它对 IPC handle 的引用，
        无法释放其它进程持有的 GPU 显存块。但只要 Triton 释放了
        handle 引用，原进程退出后 driver 会自动回收对应显存。
        """
        try:
            self._client.unregister_cuda_shared_memory()
            if self._verbose:
                logger.info("[CudaPipeline] 已清理 Triton 端残留 CUDA SHM 注册")
        except Exception as e:
            logger.warning("[CudaPipeline] 启动时清理残留 CUDA SHM 注册失败: %s", e)

    def _install_signal_handlers(self) -> None:
        """把 SIGINT / SIGTERM 接过来，确保 Ctrl+C 时也能清理 SHM。

        注意：signal.signal 只能在主线程注册。如果本 pipeline 是在
        子线程里构造的（比如 web server 的 worker），signal 注册
        会抛 ValueError，此时只靠 atexit 兜底。
        """
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            prev_int = signal.getsignal(signal.SIGINT)
            prev_term = signal.getsignal(signal.SIGTERM)

            def _handler(signum, frame, _self=self, _pi=prev_int, _pt=prev_term):
                try:
                    _self.close()
                finally:
                    prev = _pi if signum == signal.SIGINT else _pt
                    if callable(prev) and prev not in (signal.SIG_DFL, signal.SIG_IGN):
                        prev(signum, frame)
                    else:
                        # 恢复默认行为后重新抛出（保持原进程退出语义）
                        signal.signal(signum, signal.SIG_DFL)
                        sys.exit(128 + signum)

            signal.signal(signal.SIGINT, _handler)
            # SIGTERM 在 Windows 下不存在，做保护
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, _handler)
        except (ValueError, OSError) as e:
            # 子线程 / 嵌入式环境下注册失败，fallback 到 atexit
            if self._verbose:
                logger.info("[CudaPipeline] 无法安装信号处理（%s），仅依赖 atexit", e)

    # ------------------------------------------------------------------
    # 初始化阶段
    # ------------------------------------------------------------------
    def _next_wbc_stream(self):
        if not self._wbc_streams:
            return None
        with self._wbc_stream_lock:
            s = self._wbc_streams[self._wbc_stream_next % len(self._wbc_streams)]
            self._wbc_stream_next += 1
            return s

    def _ensure_models_loaded(self) -> None:
        models = [MODEL_WBC, MODEL_MEG, MODEL_CON]
        if self._enable_ca:
            models.append(MODEL_CA)
        for m in models:
            try:
                ready = self._client.is_model_ready(m)
            except Exception:
                ready = False
            if not ready:
                # explicit 模式下需要显式 load
                self._client.load_model(m)

    def _probe_output_shapes(self) -> dict[str, list[int]]:
        """跑一次真 warm-up 推理，直接从 output numpy 拿精确 shape。

        相比解析 get_model_metadata（对 dynamic dim 会返回 -1，YOLO 的 anchor
        维度会被误识别）/ get_model_config（不同 Triton 版本返回格式不一致），
        warm-up 推理 100% 准确；代价是启动时多跑 4 次 dummy 推理，总共约几百 ms。
        """
        shapes: dict[str, list[int]] = {}

        def _run(model: str, input_name: str, input_shape: list[int], out_name: str) -> list[int]:
            dummy = np.zeros(input_shape, dtype=np.float32)
            inp = grpcclient.InferInput(input_name, list(input_shape), "FP32")
            inp.set_data_from_numpy(dummy)
            out = grpcclient.InferRequestedOutput(out_name)
            res = self._client.infer(model, inputs=[inp], outputs=[out])
            arr = res.as_numpy(out_name)
            if arr is None:
                raise RuntimeError(f"{model}: warm-up 推理没有返回 {out_name}")
            return list(arr.shape)

        shapes["wbc"] = _run(
            MODEL_WBC, "images",
            [self.WBC_MAX_BATCH, 3, self._helper.WBC_INPUT_H, self._helper.WBC_INPUT_W],
            "output0",
        )
        shapes["meg"] = _run(
            MODEL_MEG, "images",
            [self.MEG_MAX_BATCH, 3, self._helper.MEG_INPUT_H, self._helper.MEG_INPUT_W],
            "output0",
        )
        shapes["con"] = _run(
            MODEL_CON, "input",
            [self.CON_MAX_BATCH, 3, self._helper.CONST_INPUT_H, self._helper.CONST_INPUT_W],
            "output",
        )
        if self._enable_ca:
            shapes["ca"] = _run(
                MODEL_CA, "data",
                [self.CA_FIXED_BATCH, 4, self._helper.CA_INPUT_H, self._helper.CA_INPUT_W],
                "222",
            )
        else:
            # CA 关闭时不触发模型 warm-up，保留占位 shape 便于统一 SHM 结构。
            shapes["ca"] = [1]
        if self._verbose:
            for k, v in shapes.items():
                logger.info("[CudaPipeline] probed %s output shape = %s", k, v)
        return shapes

    def _compute_region_bytes(self) -> dict[str, int]:
        # 输入：按客户端预处理好的 shape
        out = self._out_shapes
        wbc_in = [self.WBC_MAX_BATCH, 3, self._helper.WBC_INPUT_H, self._helper.WBC_INPUT_W]
        meg_in = [self.MEG_MAX_BATCH, 3, self._helper.MEG_INPUT_H, self._helper.MEG_INPUT_W]
        con_in = [self.CON_MAX_BATCH, 3, self._helper.CONST_INPUT_H, self._helper.CONST_INPUT_W]
        if self._enable_ca:
            ca_in = [self.CA_FIXED_BATCH, 4, self._helper.CA_INPUT_H, self._helper.CA_INPUT_W]
            ca_out_bytes = _bytes_fp32(out["ca"])
        else:
            ca_in = [1]
            ca_out_bytes = FP32_BYTES
        return {
            "wbc_in": _bytes_fp32(wbc_in),
            "wbc_out": _bytes_fp32(out["wbc"]),
            "meg_in": _bytes_fp32(meg_in),
            "meg_out": _bytes_fp32(out["meg"]),
            "con_in": _bytes_fp32(con_in),
            "con_out": _bytes_fp32(out["con"]),
            "ca_in": _bytes_fp32(ca_in),
            "ca_out": ca_out_bytes,
        }

    def _alloc_slot(self, slot_id: str, slot_ix: int) -> _BatchSlot:
        c = self._client
        dev = self._device_id
        b = self._region_bytes
        mk = lambda tag, size: _ShmRegion(c, f"dpi147246_{tag}_{slot_id}", size, dev)
        try:
            return _BatchSlot(
                slot_id=slot_id,
                slot_ix=slot_ix,
                wbc_in=mk("wbc_in", b["wbc_in"]),
                wbc_out=mk("wbc_out", b["wbc_out"]),
                meg_in=mk("meg_in", b["meg_in"]),
                meg_out=mk("meg_out", b["meg_out"]),
                con_in=mk("con_in", b["con_in"]),
                con_out=mk("con_out", b["con_out"]),
                ca_in=mk("ca_in", b["ca_in"]),
                ca_out=mk("ca_out", b["ca_out"]),
            )
        except Exception:
            # 分配过程中失败，尽量清理已分配的部分（外层捕捉到时这里的
            # 已分配 region 还没被引用到 slot 上，cudashm 在进程退出时会被
            # driver 回收；这里不做精细化补救）
            raise

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def submit(
        self,
        image_bytes: bytes,
        enable_meg: bool | None = None,
        return_timing: bool = False,
        lifecycle_trace: Trace | None = None,
    ) -> dict | tuple[dict, dict]:
        """同步提交一张图，返回与 infer_wbc_meg 一致的 dict。

        `enable_meg` 如果为 None，则使用构造时的全局配置。该批内若任一请求
        要求 MEG，则整批跑 MEG（否则跑 MEG 无效果但不影响结果）。
        """
        if self._closed:
            raise RuntimeError("pipeline 已关闭")
        want_meg = self._enable_meg if enable_meg is None else enable_meg

        submit_t0 = time.perf_counter()
        done = threading.Event()
        holder: list = [None, None, None]  # [result, error, timing]
        stage_start(lifecycle_trace, "pipeline_submit_wait")

        to_flush = None
        fire_timer = False
        my_seq: int = -1
        with self._co_lock:
            self._pending.append((image_bytes, want_meg, done, holder, lifecycle_trace))
            if len(self._pending) == 1:
                fire_timer = True
                my_seq = self._batch_seq
            if len(self._pending) >= self._batch_size:
                to_flush = self._pending
                self._pending = []
                self._batch_seq += 1

        if fire_timer and to_flush is None:
            def _timeout_flush(seq: int = my_seq):
                local = None
                with self._co_lock:
                    if self._batch_seq == seq and self._pending:
                        local = self._pending
                        self._pending = []
                        self._batch_seq += 1
                if local:
                    self._dispatch(local, coalesce_reason="timeout")
            t = threading.Timer(self._timeout_s, _timeout_flush)
            t.daemon = True
            t.start()

        if to_flush is not None:
            self._dispatch(to_flush, coalesce_reason="full")

        done.wait()
        if holder[1] is not None:
            raise holder[1]
        if return_timing:
            timing = dict(holder[2] or {})
            timing["submit_total_ms"] = round((time.perf_counter() - submit_t0) * 1000.0, 3)
            return holder[0], timing
        return holder[0]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # 把剩余 pending 全部用错误 fail 掉（避免调用方永久 wait）
        with self._co_lock:
            leftover = self._pending
            self._pending = []
        for _, _, done, holder, *_ in leftover:
            holder[1] = RuntimeError("pipeline 关闭时尚未处理")
            done.set()
        # 打印性能统计
        try:
            if self._verbose:
                self._print_stats_summary()
            summ = self._format_coalesce_summary()
            if summ is not None:
                if self._verbose:
                    logger.info("%s", summ)
        except Exception as e:
            logger.warning("[CudaPipeline] 记录统计失败: %s", e)
        # 先停掉 CUDA SHM I/O 线程，再 destroy 区（避免与 in-flight memcpy 竞态）
        _shutdown_cuda_shm_io_executor()
        # 释放所有 slot 的 SHM（每块失败不影响其它块的清理）
        n_ok = 0
        n_fail = 0
        for s in self._slots:
            for r in s.regions():
                try:
                    r.close()
                    n_ok += 1
                except Exception:
                    n_fail += 1
        self._slots = []
        if self._verbose or n_fail > 0:
            msg = f"[CudaPipeline] close() 释放 SHM: ok={n_ok}, fail={n_fail}"
            if n_fail > 0:
                logger.warning("%s", msg)
            else:
                logger.info("%s", msg)
        # 关闭后处理 / 预处理 / 解码线程池与进程池
        if getattr(self, "_heavy_post_proc_pool", None) is not None:
            try:
                self._heavy_post_proc_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._heavy_post_proc_pool = None
        for pool in (
            getattr(self, "_heavy_post_thread_pool", None),
            getattr(self, "_decode_pool", None),
            self._pp_pool,
        ):
            if pool is None:
                continue
            try:
                pool.shutdown(wait=False)
            except Exception:
                pass

    def _print_stats_summary(self) -> None:
        """关闭时打印各阶段累计耗时。仅在跑过至少 1 个 batch 时打印。"""
        with self._stats_lock:
            n_batch = self._stats_count
            if n_batch == 0:
                return
            n_imgs = self._stats_imgs
            wbc_total = self._stats_wbc_total
            n_meg = self._stats_meg_batches
            total_ms = self._stats_total_ms
            stage_sum = dict(self._stats_stage_sum)
            stage_cnt = dict(self._stats_stage_count)

        avg_total = total_ms / n_batch if n_batch else 0.0
        # 按 batch 数取平均；某些阶段（如 meg_pre/meg_post）只在跑 MEG 的 batch 计入，需单独平均
        meg_only = {"meg_pre", "meg_post"}
        sum_per_batch = sum(stage_sum.values()) / n_batch if n_batch else 0.0

        lines = [
            "",
            "=" * 78,
            "BatchedCudaWbcMegPipeline 性能统计",
            f"  batches={n_batch}  real_images={n_imgs}  WBC_total={wbc_total}  "
            f"MEG_batches={n_meg}/{n_batch}  num_slots={self._num_slots}",
            f"  avg batch wall-time = {avg_total:.2f} ms  "
            f"(各阶段累加 {sum_per_batch:.2f} ms/batch)",
            "-" * 78,
            f"  {'stage':<14} {'avg ms/batch':>14} {'% of total':>12} "
            f"{'count':>8}  notes",
            "-" * 78,
        ]
        for s in _STAGE_ORDER:
            cnt = stage_cnt.get(s, 0)
            if cnt == 0:
                continue
            divisor = cnt if s in meg_only else n_batch
            avg = stage_sum[s] / divisor if divisor else 0.0
            pct = (stage_sum[s] / total_ms * 100.0) if total_ms > 0 else 0.0
            note = "(only MEG batches)" if s in meg_only else ""
            lines.append(f"  {s:<14} {avg:>14.2f} {pct:>11.1f}% {cnt:>8}  {note}")
        lines.extend(["=" * 78, ""])
        logger.info("%s", "\n".join(lines))

    def _format_coalesce_summary(self) -> str | None:
        """返回合批累计统计多行文本；无 batch 时返回 None。"""
        with self._stats_lock:
            n_batch = self._stats_count
            if n_batch == 0:
                return None
            full = self._coalesce_full
            timeout_batches = self._coalesce_timeout
            n1 = self._coalesce_timeout_n1
            by_n = dict(sorted(self._coalesce_by_n.items()))
        lines = [
            "",
            "=" * 78,
            "BatchedCudaWbcMegPipeline 合批统计",
            f"  总 batch={n_batch}  凑满发出={full}  超时发出={timeout_batches}  "
            f"其中超时且仅 1 张={n1}",
            f"  各实际张数 n 出现次数: {by_n}",
            "=" * 78,
            "",
        ]
        return "\n".join(lines)

    def _accumulate_stats(
        self,
        timer: _StageTimer,
        n_real: int,
        run_meg: bool,
        wbc_count: int,
    ) -> None:
        with self._stats_lock:
            self._stats_count += 1
            self._stats_imgs += n_real
            self._stats_wbc_total += wbc_count
            if run_meg:
                self._stats_meg_batches += 1
            self._stats_total_ms += timer.total_ms()
            for s, ms in timer.stages.items():
                self._stats_stage_sum[s] = self._stats_stage_sum.get(s, 0.0) + ms
                self._stats_stage_count[s] = self._stats_stage_count.get(s, 0) + 1

    # ------------------------------------------------------------------
    # 核心：batch 执行
    # ------------------------------------------------------------------
    def _dispatch(self, items: list, *, coalesce_reason: str) -> None:
        """items: [(image_bytes, want_meg, done, holder, trace), ...] 长度 1..batch_size

        coalesce_reason:
            - "full": 凑满 batch_size 立即发出
            - "timeout": 定时器到期、未凑满即 flush
        """
        n = len(items)
        if coalesce_reason not in ("full", "timeout"):
            raise ValueError(f"coalesce_reason 非法: {coalesce_reason!r}")
        with self._stats_lock:
            self._coalesce_by_n[n] = self._coalesce_by_n.get(n, 0) + 1
            if coalesce_reason == "full":
                self._coalesce_full += 1
            else:
                self._coalesce_timeout += 1
                if n == 1:
                    self._coalesce_timeout_n1 += 1
        _stage_end_items(items, "pipeline_submit_wait")
        timer = _StageTimer()
        try:
            prepared = self._prepare_batch(items, timer)
            if prepared.get("short_circuit_empty", False):
                self._dispatch_short_circuit_empty(
                    items,
                    timer,
                    run_meg=bool(prepared.get("run_meg", False)),
                    coalesce_reason=coalesce_reason,
                )
                return
            _stage_start_items(items, "slot_wait", scope="batch")
            slot = self._slot_q.get()
            _stage_end_items(items, "slot_wait")
            timer.mark("wait_slot")
            try:
                self._run_batch_on_slot(
                    items,
                    slot,
                    timer,
                    prepared,
                    coalesce_reason=coalesce_reason,
                )
            finally:
                self._slot_q.put(slot)
        except Exception as e:
            for _, _, done, holder, *_ in items:
                if not done.is_set():
                    holder[1] = e
                    done.set()

    def _dispatch_short_circuit_empty(
        self,
        items: list,
        timer: _StageTimer,
        *,
        run_meg: bool,
        coalesce_reason: str,
    ) -> None:
        """decode 后短路：返回空结果，跳过 Triton 推理和全部后处理。"""
        n = len(items)
        stage_ms = {k: round(float(v), 3) for k, v in timer.stages.items()}
        preprocess_ms = sum(
            timer.stages.get(k, 0.0)
            for k in (
                "decode",
                "cpu_to_gpu_upload",
                "resize_612",
                "mode2_gpu_preprocess",
                "wbc_pre",
                "meg_pre",
                "con_pre",
                "ca_pre",
            )
        )
        timing_common = {
            "batch_size_real": n,
            "batch_size_config": self._batch_size,
            "coalesce_reason": coalesce_reason,
            "batch_enable_meg": bool(run_meg),
            "batch_total_ms": round(timer.total_ms(), 3),
            "stage_ms": stage_ms,
            "preprocess_total_ms": round(preprocess_ms, 3),
            "infer_total_ms": 0.0,
            "postprocess_total_ms": 0.0,
            "cpu_to_gpu_ms": 0.0,
            "gpu_to_cpu_ms": 0.0,
            "wait_slot_ms": 0.0,
        }
        for item in items:
            _, _, done, holder, *_ = item
            trace = _item_trace(item)
            stage_start(trace, "result_build", scope="image")
            holder[0] = {
                "wbc_detections": np.array([]),
                "wbc_num": 0,
                "meg_detections": np.array([]),
                "meg_num": 0,
                "constituency_regions": np.array([]),
                "constituency_grades": np.array([]),
                "constituency_scores": np.array([]),
                "cell_analysis_wbc_pixel_count": 0,
                "cell_analysis_red_pixel_count": 0,
            }
            set_batch(
                trace,
                batch_size_real=n,
                batch_size_config=self._batch_size,
                batch_enable_meg=bool(run_meg),
            )
            set_result(trace, wbc_num=0, meg_num=0)
            holder[2] = timing_common
            stage_end(trace, "result_build")
            done.set()

    def _run_heavy_post(
        self,
        wbc_out: np.ndarray,
        wbc_meta: list,
        meg_out: np.ndarray | None,
        meg_meta: list | None,
        *,
        run_meg: bool,
        n: int,
    ) -> tuple[list, list]:
        """WBC/MEG 后处理（含 NMS）。模式见 ``heavy_post_mode``。"""
        wbc_post_fn = self._helper._wbc_postprocess_batch
        if self._heavy_post_mode == "process":
            pool = self._heavy_post_proc_pool
            if pool is None:
                raise RuntimeError("heavy_post_mode=process 但进程池未创建")
            fut = pool.submit(
                _heavy_post_proc_task,
                wbc_out,
                wbc_meta,
                meg_out if run_meg else None,
                meg_meta if run_meg else None,
                run_meg,
                n,
            )
            return fut.result()

        if self._heavy_post_mode == "thread":
            pool = self._heavy_post_thread_pool
            if pool is None:
                raise RuntimeError("heavy_post_mode=thread 但线程池未创建")
            f_wbc = pool.submit(wbc_post_fn, wbc_out, wbc_meta)
            if run_meg and meg_out is not None and meg_meta is not None:
                f_meg = pool.submit(
                    self._helper._meg_postprocess_batch, meg_out, meg_meta,
                )
                return f_wbc.result(), f_meg.result()
            return f_wbc.result(), [
                np.zeros((0, 5), dtype=np.float32) for _ in range(n)
            ]

        # serial：全局锁，与旧行为一致
        with _HEAVY_POST_LOCK:
            wbc = wbc_post_fn(wbc_out, wbc_meta)
            if run_meg and meg_out is not None and meg_meta is not None:
                meg = self._helper._meg_postprocess_batch(meg_out, meg_meta)
            else:
                meg = [np.zeros((0, 5), dtype=np.float32) for _ in range(n)]
            return wbc, meg

    def _prepare_batch(self, items: list, timer: _StageTimer) -> dict[str, Any]:
        n = len(items)
        run_meg = self._enable_meg and any(it[1] for it in items)
        helper = self._helper
        pool = self._pp_pool if self._parallel_preprocess else None
        dec_pool = self._decode_pool or pool

        # ── CPU 预处理 ────────────────────────────────────────────
        # decode 4 张图并发（cv2.imdecode 释放 GIL）
        def _decode_item(item: tuple) -> np.ndarray:
            trace = _item_trace(item)
            stage_start(trace, "image_decode", scope="image")
            try:
                return helper._decode(item[0])
            finally:
                stage_end(trace, "image_decode")

        if dec_pool is not None and n > 1:
            bgr_list = list(dec_pool.map(_decode_item, items))
        else:
            bgr_list = [_decode_item(it) for it in items]
        if any(b is None for b in bgr_list):
            raise RuntimeError("cv2.imdecode 失败")
        timer.mark("decode")
        if self._decode_only_return_empty:
            return {
                "n": n,
                "run_meg": False,
                "short_circuit_empty": True,
            }

        if self._use_gpu_preprocess_mode2:
            # 方案2：单次 H2D 后全程 GPU；WBC 与 resize/MEG/CON/CA 尽量双 stream 重叠；
            # 写 SHM 用 D2D（dlpack），避免 D2H+numpy+H2D。
            import torch

            streams = self._wbc_streams
            s_wbc = streams[0]
            s_branch = streams[1] if len(streams) >= 2 else streams[0]
            parallel_branches = s_wbc is not s_branch

            upload_done = torch.cuda.Event()

            _stage_start_items(items, "cpu_to_gpu_upload", scope="image")
            with torch.cuda.stream(s_wbc):
                bgr_u8_gpu_list = [
                    torch.from_numpy(b).to(
                        device=self._torch_device, dtype=torch.uint8, non_blocking=False
                    )
                    for b in bgr_list
                ]
            upload_done.record(s_wbc)
            _stage_end_items(items, "cpu_to_gpu_upload")
            timer.mark("cpu_to_gpu_upload")
            if self._cpu_to_gpu_only_return_empty:
                torch.cuda.synchronize(self._torch_device)
                return {
                    "n": n,
                    "run_meg": False,
                    "short_circuit_empty": True,
                }

            meg_t = None
            meg_meta = None

            _stage_start_items(items, "preprocess", scope="batch")
            if parallel_branches:
                _stage_start_items(items, "mode2_gpu_preprocess", scope="batch")
                with torch.cuda.stream(s_wbc):
                    upload_done.wait(s_wbc)
                    wbc_t, wbc_meta = helper._wbc_preprocess_batch_cuda_from_gpu(
                        bgr_u8_gpu_list, stream=s_wbc
                    )
                with torch.cuda.stream(s_branch):
                    upload_done.wait(s_branch)
                    bgr_612_f32_nchw = helper._resize_612_batch_cuda_from_gpu(
                        bgr_u8_gpu_list, stream=s_branch
                    )
                    if run_meg:
                        meg_t, meg_meta = helper._meg_preprocess_batch_cuda_from_gpu(
                            bgr_612_f32_nchw, stream=s_branch
                        )
                    con_t, con_regions_all = helper._constituency_preprocess_batch_cuda_from_gpu(
                        bgr_612_f32_nchw, stream=s_branch
                    )
                    if self._enable_ca:
                        ca_t = helper._cell_analysis_preprocess_batch_cuda_from_gpu(
                            bgr_612_f32_nchw, stream=s_branch
                        )
                    else:
                        ca_t = torch.zeros((1,), dtype=torch.float32, device=self._torch_device)
                torch.cuda.synchronize(self._torch_device)
                _stage_end_items(items, "mode2_gpu_preprocess")
                timer.mark("mode2_gpu_preprocess")
            else:
                with torch.cuda.stream(s_wbc):
                    upload_done.wait(s_wbc)
                    _stage_start_items(items, "resize_612", scope="batch")
                    bgr_612_f32_nchw = helper._resize_612_batch_cuda_from_gpu(
                        bgr_u8_gpu_list, stream=s_wbc
                    )
                    _stage_end_items(items, "resize_612")
                    timer.mark("resize_612")
                    _stage_start_items(items, "wbc_preprocess", scope="batch")
                    wbc_t, wbc_meta = helper._wbc_preprocess_batch_cuda_from_gpu(
                        bgr_u8_gpu_list, stream=s_wbc
                    )
                    _stage_end_items(items, "wbc_preprocess")
                    if run_meg:
                        _stage_start_items(items, "meg_preprocess", scope="batch")
                        meg_t, meg_meta = helper._meg_preprocess_batch_cuda_from_gpu(
                            bgr_612_f32_nchw, stream=s_wbc
                        )
                        _stage_end_items(items, "meg_preprocess")
                        timer.mark("meg_pre")
                    _stage_start_items(items, "constituency_preprocess", scope="batch")
                    con_t, con_regions_all = helper._constituency_preprocess_batch_cuda_from_gpu(
                        bgr_612_f32_nchw, stream=s_wbc
                    )
                    _stage_end_items(items, "constituency_preprocess")
                    timer.mark("con_pre")
                    if self._enable_ca:
                        _stage_start_items(items, "cell_analysis_preprocess", scope="batch")
                        ca_t = helper._cell_analysis_preprocess_batch_cuda_from_gpu(
                            bgr_612_f32_nchw, stream=s_wbc
                        )
                        _stage_end_items(items, "cell_analysis_preprocess")
                        timer.mark("ca_pre")
                    else:
                        ca_t = torch.zeros((1,), dtype=torch.float32, device=self._torch_device)
                s_wbc.synchronize()
            _stage_end_items(items, "preprocess")

            ca_orig_sizes = [(b.shape[0], b.shape[1]) for b in bgr_list]
            if self._preprocess_only_return_empty:
                return {
                    "n": n,
                    "run_meg": run_meg,
                    "short_circuit_empty": True,
                }

            # 保留 GPU 张量，由 _run_batch_on_slot D2D 写入 SHM（无 D2H）
            wbc_in_np = None
            meg_in_np = None
            con_in_np = None
            ca_in_np = None
        else:
            # resize 到 612×512（Constituency / CA / MEG 共用的中间尺寸）
            _stage_start_items(items, "preprocess", scope="batch")
            _stage_start_items(items, "resize_612", scope="batch")
            if pool is not None and n > 1:
                bgr_612_list = list(pool.map(
                    lambda b: cv2.resize(b, (612, 512), interpolation=cv2.INTER_LINEAR),
                    bgr_list,
                ))
            else:
                bgr_612_list = [
                    cv2.resize(b, (612, 512), interpolation=cv2.INTER_LINEAR)
                    for b in bgr_list
                ]
            _stage_end_items(items, "resize_612")
            timer.mark("resize_612")

            # WBC：可选 GPU letterbox（与 MEG/CON/CA 线程池重叠）；否则与其它任务一样走线程池
            if pool is not None:
                if run_meg:
                    _stage_start_items(items, "meg_preprocess", scope="batch")
                _stage_start_items(items, "constituency_preprocess", scope="batch")
                if self._enable_ca:
                    _stage_start_items(items, "cell_analysis_preprocess", scope="batch")
                f_meg = pool.submit(helper._meg_preprocess_batch, bgr_612_list) if run_meg else None
                f_con = pool.submit(helper._constituency_preprocess_batch, bgr_612_list)
                f_ca = (
                    pool.submit(helper._cell_analysis_preprocess_batch, bgr_612_list)
                    if self._enable_ca else None
                )

                _stage_start_items(items, "wbc_preprocess", scope="batch")
                if self._use_gpu_wbc_pre:
                    wbc_stream = self._next_wbc_stream()
                    wbc_t, wbc_meta = helper._wbc_preprocess_batch_cuda(
                        bgr_list,
                        device=self._torch_device,
                        stream=wbc_stream,
                    )
                    wbc_stream.synchronize()
                    wbc_in_np = wbc_t.detach().cpu().numpy()
                else:
                    f_wbc = pool.submit(helper._wbc_preprocess_batch, bgr_list)
                    wbc_in_np, wbc_meta = f_wbc.result()
                _stage_end_items(items, "wbc_preprocess")
                timer.mark("wbc_pre")
                if run_meg:
                    meg_in_np, meg_meta = f_meg.result()
                    _stage_end_items(items, "meg_preprocess")
                    timer.mark("meg_pre")
                con_in_np, con_regions_all = f_con.result()
                _stage_end_items(items, "constituency_preprocess")
                timer.mark("con_pre")
                if self._enable_ca:
                    ca_in_np = f_ca.result()
                    _stage_end_items(items, "cell_analysis_preprocess")
                    timer.mark("ca_pre")
                else:
                    ca_in_np = np.zeros((1,), dtype=np.float32)
                ca_orig_sizes = [(b.shape[0], b.shape[1]) for b in bgr_list]
            else:
                _stage_start_items(items, "wbc_preprocess", scope="batch")
                if self._use_gpu_wbc_pre:
                    wbc_stream = self._next_wbc_stream()
                    wbc_t, wbc_meta = helper._wbc_preprocess_batch_cuda(
                        bgr_list,
                        device=self._torch_device,
                        stream=wbc_stream,
                    )
                    wbc_stream.synchronize()
                    wbc_in_np = wbc_t.detach().cpu().numpy()
                else:
                    wbc_in_np, wbc_meta = helper._wbc_preprocess_batch(bgr_list)
                _stage_end_items(items, "wbc_preprocess")
                timer.mark("wbc_pre")
                if run_meg:
                    _stage_start_items(items, "meg_preprocess", scope="batch")
                    meg_in_np, meg_meta = helper._meg_preprocess_batch(bgr_612_list)
                    _stage_end_items(items, "meg_preprocess")
                    timer.mark("meg_pre")
                _stage_start_items(items, "constituency_preprocess", scope="batch")
                con_in_np, con_regions_all = helper._constituency_preprocess_batch(bgr_612_list)
                _stage_end_items(items, "constituency_preprocess")
                timer.mark("con_pre")
                if self._enable_ca:
                    _stage_start_items(items, "cell_analysis_preprocess", scope="batch")
                    ca_in_np = helper._cell_analysis_preprocess_batch(bgr_612_list)
                    _stage_end_items(items, "cell_analysis_preprocess")
                    timer.mark("ca_pre")
                else:
                    ca_in_np = np.zeros((1,), dtype=np.float32)
                ca_orig_sizes = [(b.shape[0], b.shape[1]) for b in bgr_list]
            _stage_end_items(items, "preprocess")

        if not self._use_gpu_preprocess_mode2 and self._preprocess_only_return_empty:
            return {
                "n": n,
                "run_meg": run_meg,
                "short_circuit_empty": True,
            }

        if self._use_gpu_preprocess_mode2:
            return {
                "n": n,
                "run_meg": run_meg,
                "use_dlpack_shm": True,
                "wbc_t": wbc_t,
                "meg_t": meg_t if run_meg else None,
                "meg_meta": meg_meta if run_meg else None,
                "con_t": con_t,
                "con_regions_all": con_regions_all,
                "ca_t": ca_t,
                "ca_orig_sizes": ca_orig_sizes,
                "wbc_meta": wbc_meta,
            }

        return {
            "n": n,
            "run_meg": run_meg,
            "use_dlpack_shm": False,
            "wbc_in_np": wbc_in_np,
            "wbc_meta": wbc_meta,
            "meg_in_np": meg_in_np if run_meg else None,
            "meg_meta": meg_meta if run_meg else None,
            "con_in_np": con_in_np,
            "con_regions_all": con_regions_all,
            "ca_in_np": ca_in_np,
            "ca_orig_sizes": ca_orig_sizes,
        }

    def _run_batch_on_slot(
        self,
        items: list,
        slot: _BatchSlot,
        timer: _StageTimer,
        prepared: dict[str, Any],
        *,
        coalesce_reason: str,
    ) -> None:
        n = int(prepared["n"])
        run_meg = bool(prepared["run_meg"])
        use_dlpack_shm = bool(prepared.get("use_dlpack_shm", False))
        wbc_meta = prepared["wbc_meta"]
        meg_meta = prepared["meg_meta"]
        con_regions_all = prepared["con_regions_all"]
        ca_orig_sizes = prepared["ca_orig_sizes"]
        pool = self._pp_pool if self._parallel_preprocess else None

        # ── 写入 CUDA SHM（输入） ─────────────────────────────────
        # 重要：tritonclient.utils.cuda_shared_memory 不是线程安全（cudaMemcpy
        # / cudaSetDevice 多线程并发会触发 driver 段错误），经专用 I/O 线程串行写。
        # mode2：优先 D2D（dlpack），避免预处理张量先 D2H 再 H2D。
        _stage_start_items(items, "cuda_shm_input_write", scope="batch")
        if use_dlpack_shm:
            import torch

            torch.cuda.synchronize(self._torch_device)
            _shm_io_write_slot_batch_dlpack(
                slot,
                prepared["wbc_t"],
                prepared["con_t"],
                prepared["ca_t"],
                prepared["meg_t"] if run_meg else None,
                run_meg,
            )
        else:
            _stage_start_items(items, "cpu_to_gpu_upload", scope="batch")
            wbc_in_np = prepared["wbc_in_np"]
            meg_in_np = prepared["meg_in_np"]
            con_in_np = prepared["con_in_np"]
            ca_in_np = prepared["ca_in_np"]
            meg_for_shm = meg_in_np if run_meg else None
            _shm_io_write_slot_batch(
                slot, wbc_in_np, con_in_np, ca_in_np, meg_for_shm, run_meg,
            )
            _stage_end_items(items, "cpu_to_gpu_upload")
        _stage_end_items(items, "cuda_shm_input_write")
        timer.mark("shm_write")
        if self._shm_write_only_return_empty:
            self._dispatch_short_circuit_empty(
                items,
                timer,
                run_meg=run_meg,
                coalesce_reason=coalesce_reason,
            )
            return

        # ── 构造 InferInput / InferRequestedOutput（引用 SHM） ────
        _stage_start_items(items, "infer_prepare", scope="batch")
        def _shm_input(name: str, shape: list[int], region: _ShmRegion) -> grpcclient.InferInput:
            inp = grpcclient.InferInput(name, shape, "FP32")
            inp.set_shared_memory(region.name, int(np.prod(shape)) * FP32_BYTES)
            return inp

        def _shm_output(name: str, region: _ShmRegion, byte_size: int) -> grpcclient.InferRequestedOutput:
            out = grpcclient.InferRequestedOutput(name)
            out.set_shared_memory(region.name, byte_size)
            return out

        # 各模型的 batched output shape（当前真实 batch）
        # 说明：WBC/MEG/Const 都是 max_batch > 0，所以输出的 batch 维 == 实际 batch
        #       CA 是 max_batch_size: 0，输出 shape 固定 = [4, 3, 512, 640]
        wbc_out_full = self._out_shapes["wbc"]           # 对应 WBC_MAX_BATCH
        meg_out_full = self._out_shapes["meg"]
        con_out_full = self._out_shapes["con"]
        ca_out_full = self._out_shapes["ca"]

        def _out_shape_for(base_shape: list[int], actual_batch: int, max_batch: int) -> list[int]:
            if len(base_shape) == 0:
                return base_shape
            if base_shape[0] == max_batch:
                return [actual_batch] + list(base_shape[1:])
            return list(base_shape)

        wbc_out_shape = _out_shape_for(wbc_out_full, n, self.WBC_MAX_BATCH)
        meg_out_shape = _out_shape_for(meg_out_full, n, self.MEG_MAX_BATCH)
        con_out_shape = _out_shape_for(con_out_full, n * 4, self.CON_MAX_BATCH)
        ca_out_shape = list(ca_out_full) if self._enable_ca else [1]

        # ── 并行发起 async_infer ──────────────────────────────────
        callers: dict[str, _AsyncCall] = {}
        _stage_end_items(items, "infer_prepare")
        _stage_start_items(items, "infer", scope="batch")

        # WBC
        wbc_in_shape = [n, 3, self._helper.WBC_INPUT_H, self._helper.WBC_INPUT_W]
        # SHM 里已经写了完整的 WBC_MAX_BATCH 的数据（前 n 张有效），
        # 但我们只告诉 Triton "n 张"，它按前 n*per_sample_size 字节读取。
        # 预处理时我们实际上只 stack 了 n 张（不是 WBC_MAX_BATCH），所以 SHM 前 n 张就是有效数据。
        call_wbc = _AsyncCall(MODEL_WBC)
        self._client.async_infer(
            model_name=MODEL_WBC,
            inputs=[_shm_input("images", wbc_in_shape, slot.wbc_in)],
            outputs=[_shm_output("output0", slot.wbc_out, _bytes_fp32(wbc_out_shape))],
            callback=call_wbc._callback,
        )
        callers["wbc"] = call_wbc

        # Constituency
        con_in_shape = [n * 4, 3, self._helper.CONST_INPUT_H, self._helper.CONST_INPUT_W]
        call_con = _AsyncCall(MODEL_CON)
        self._client.async_infer(
            model_name=MODEL_CON,
            inputs=[_shm_input("input", con_in_shape, slot.con_in)],
            outputs=[_shm_output("output", slot.con_out, _bytes_fp32(con_out_shape))],
            callback=call_con._callback,
        )
        callers["con"] = call_con

        # Cell Analysis（固定 batch=4）
        if self._enable_ca:
            ca_in_shape = [4, 4, self._helper.CA_INPUT_H, self._helper.CA_INPUT_W]
            call_ca = _AsyncCall(MODEL_CA)
            self._client.async_infer(
                model_name=MODEL_CA,
                inputs=[_shm_input("data", ca_in_shape, slot.ca_in)],
                outputs=[_shm_output("222", slot.ca_out, _bytes_fp32(ca_out_shape))],
                callback=call_ca._callback,
            )
            callers["ca"] = call_ca

        # MEG（可选）
        if run_meg:
            meg_in_shape = [n, 3, self._helper.MEG_INPUT_H, self._helper.MEG_INPUT_W]
            call_meg = _AsyncCall(MODEL_MEG)
            self._client.async_infer(
                model_name=MODEL_MEG,
                inputs=[_shm_input("images", meg_in_shape, slot.meg_in)],
                outputs=[_shm_output("output0", slot.meg_out, _bytes_fp32(meg_out_shape))],
                callback=call_meg._callback,
            )
            callers["meg"] = call_meg

        timer.mark("infer_launch")

        # 等待所有子模型完成
        for c in callers.values():
            c.wait()
        _stage_end_items(items, "infer")
        timer.mark("infer_wait")

        # ── 从 SHM 读回输出 ──────────────────────────────────────
        _stage_start_items(items, "gpu_to_cpu_download", scope="batch")
        wbc_out, con_out, ca_out, meg_out = _shm_io_read_slot_batch(
            slot,
            wbc_out_shape,
            con_out_shape,
            ca_out_shape,
            meg_out_shape if run_meg else None,
            run_meg,
        )
        _stage_end_items(items, "gpu_to_cpu_download")
        timer.mark("shm_read")

        # ── 后处理 ────────────────────────────────────────────────
        # WBC/MEG：按 heavy_post_mode 串行/线程/进程并行；Const/CA 仍可与重后处理重叠。
        _stage_start_items(items, "postprocess", scope="batch")
        if pool is not None:
            _stage_start_items(items, "constituency_postprocess", scope="batch")
            f_con_post = pool.submit(
                self._helper._constituency_postprocess_batch, con_out, n, con_regions_all,
            )
            if self._enable_ca:
                _stage_start_items(items, "cell_analysis_postprocess", scope="batch")
            f_ca_post = (
                pool.submit(
                    self._helper._cell_analysis_postprocess_and_count, ca_out, ca_orig_sizes,
                )
                if self._enable_ca else None
            )
            _stage_start_items(items, "wbc_postprocess", scope="batch")
            if run_meg:
                _stage_start_items(items, "meg_postprocess", scope="batch")
            wbc_res_list, meg_res_list = self._run_heavy_post(
                wbc_out, wbc_meta, meg_out, meg_meta, run_meg=run_meg, n=n,
            )
            _stage_end_items(items, "wbc_postprocess")
            timer.mark("wbc_post")
            if run_meg:
                _stage_end_items(items, "meg_postprocess")
                timer.mark("meg_post")
            con_res_list = f_con_post.result()
            _stage_end_items(items, "constituency_postprocess")
            timer.mark("con_post")
            if self._enable_ca:
                ca_counts_wbc, ca_counts_red = f_ca_post.result()
                _stage_end_items(items, "cell_analysis_postprocess")
                timer.mark("ca_post")
            else:
                ca_counts_wbc = np.zeros((n,), dtype=np.int32)
                ca_counts_red = np.zeros((n,), dtype=np.int32)
        else:
            _stage_start_items(items, "wbc_postprocess", scope="batch")
            if run_meg:
                _stage_start_items(items, "meg_postprocess", scope="batch")
            wbc_res_list, meg_res_list = self._run_heavy_post(
                wbc_out, wbc_meta, meg_out, meg_meta, run_meg=run_meg, n=n,
            )
            _stage_end_items(items, "wbc_postprocess")
            timer.mark("wbc_post")
            if run_meg:
                _stage_end_items(items, "meg_postprocess")
                timer.mark("meg_post")
            _stage_start_items(items, "constituency_postprocess", scope="batch")
            con_res_list = self._helper._constituency_postprocess_batch(
                con_out, n, con_regions_all,
            )
            _stage_end_items(items, "constituency_postprocess")
            timer.mark("con_post")
            if self._enable_ca:
                _stage_start_items(items, "cell_analysis_postprocess", scope="batch")
                ca_counts_wbc, ca_counts_red = self._helper._cell_analysis_postprocess_and_count(
                    ca_out, ca_orig_sizes,
                )
                _stage_end_items(items, "cell_analysis_postprocess")
                timer.mark("ca_post")
            else:
                ca_counts_wbc = np.zeros((n,), dtype=np.int32)
                ca_counts_red = np.zeros((n,), dtype=np.int32)
        _stage_end_items(items, "postprocess")

        # ── 派发每张图的结果到各自的调用方 ─────────────────────────
        stage_ms = {k: round(float(v), 3) for k, v in timer.stages.items()}
        preprocess_ms = sum(
            timer.stages.get(k, 0.0)
            for k in (
                "decode",
                "cpu_to_gpu_upload",
                "resize_612",
                "mode2_gpu_preprocess",
                "wbc_pre",
                "meg_pre",
                "con_pre",
                "ca_pre",
            )
        )
        infer_ms = timer.stages.get("infer_launch", 0.0) + timer.stages.get("infer_wait", 0.0)
        postprocess_ms = sum(
            timer.stages.get(k, 0.0)
            for k in ("wbc_post", "meg_post", "con_post", "ca_post")
        )
        timing_common = {
            "batch_size_real": n,
            "batch_size_config": self._batch_size,
            "coalesce_reason": coalesce_reason,
            "batch_enable_meg": run_meg,
            "batch_total_ms": round(timer.total_ms(), 3),
            "stage_ms": stage_ms,
            "preprocess_total_ms": round(preprocess_ms, 3),
            "infer_total_ms": round(infer_ms, 3),
            "postprocess_total_ms": round(postprocess_ms, 3),
            # CUDA SHM 写入可近似视为 CPU->GPU 拷贝耗时（含必要准备）
            "cpu_to_gpu_ms": round(timer.stages.get("shm_write", 0.0), 3),
            # CUDA SHM 读取可近似视为 GPU->CPU 拷贝耗时
            "gpu_to_cpu_ms": round(timer.stages.get("shm_read", 0.0), 3),
            "wait_slot_ms": round(timer.stages.get("wait_slot", 0.0), 3),
        }
        wbc_count_total = 0
        for i in range(n):
            image_bytes, want_meg, done, holder, *_ = items[i]
            trace = _item_trace(items[i])
            stage_start(trace, "result_build", scope="image")
            wbc_res = wbc_res_list[i]
            wbc_count_total += int(wbc_res.shape[0])
            meg_res = (meg_res_list[i] if want_meg and run_meg
                       else np.zeros((0, 5), dtype=np.float32))
            con_regions_arr, con_grades_arr, con_scores_arr = con_res_list[i]
            result = {
                "wbc_detections": wbc_res if wbc_res.shape[0] > 0 else np.array([]),
                "wbc_num": int(wbc_res.shape[0]),
                "meg_detections": meg_res if meg_res.shape[0] > 0 else np.array([]),
                "meg_num": int(meg_res.shape[0]),
                "constituency_regions": con_regions_arr,
                "constituency_grades": con_grades_arr,
                "constituency_scores": con_scores_arr,
                "cell_analysis_wbc_pixel_count": int(ca_counts_wbc[i]),
                "cell_analysis_red_pixel_count": int(ca_counts_red[i]),
            }
            set_batch(
                trace,
                batch_size_real=n,
                batch_size_config=self._batch_size,
                batch_enable_meg=run_meg,
            )
            set_result(trace, wbc_num=int(wbc_res.shape[0]), meg_num=int(meg_res.shape[0]))
            holder[0] = result
            holder[2] = timing_common
            stage_end(trace, "result_build")
            done.set()
        timer.mark("dispatch")

        # 累加统计
        self._accumulate_stats(timer, n_real=n, run_meg=run_meg, wbc_count=wbc_count_total)


# ─────────────────────────────────────────────────────────────────────────
# 便捷单例
# ─────────────────────────────────────────────────────────────────────────
_SINGLETON_LOCK = threading.Lock()
_SINGLETON: BatchedCudaWbcMegPipeline | None = None


def get_cuda_pipeline(
    url: str,
    *,
    enable_meg: bool = True,
    enable_ca: bool = _PROFILE_DEFAULTS["enable_ca"],
    batch_size: int = _PROFILE_DEFAULTS["batch_size"],
    timeout_ms: int = _PROFILE_DEFAULTS["timeout_ms"],
    num_slots: int = _PROFILE_DEFAULTS["num_slots"],
    device_id: int = 0,
    verbose: bool = False,
    parallel_preprocess: bool = True,
    preprocess_workers: int = _PROFILE_DEFAULTS["preprocess_workers"],
    decode_workers: int | None = None,
    heavy_post_mode: str | None = None,
    heavy_post_workers: int | None = None,
    use_gpu_wbc_preprocess: bool | None = None,
    use_gpu_preprocess_mode2: bool = True,
    purge_stale_cuda_shm: bool = False,
    decode_only_return_empty: bool = False,
    cpu_to_gpu_only_return_empty: bool = False,
    preprocess_only_return_empty: bool = False,
    shm_write_only_return_empty: bool = False,
    coalesce_log_path: str | None = None,
) -> BatchedCudaWbcMegPipeline:
    """拿到进程级单例。首次调用时构造，后续复用。"""
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = BatchedCudaWbcMegPipeline(
                url,
                enable_meg=enable_meg,
                enable_ca=enable_ca,
                batch_size=batch_size,
                timeout_ms=timeout_ms,
                num_slots=num_slots,
                device_id=device_id,
                verbose=verbose,
                parallel_preprocess=parallel_preprocess,
                preprocess_workers=preprocess_workers,
                decode_workers=decode_workers,
                heavy_post_mode=heavy_post_mode,
                heavy_post_workers=heavy_post_workers,
                use_gpu_wbc_preprocess=use_gpu_wbc_preprocess,
                use_gpu_preprocess_mode2=use_gpu_preprocess_mode2,
                purge_stale_cuda_shm=purge_stale_cuda_shm,
                decode_only_return_empty=decode_only_return_empty,
                cpu_to_gpu_only_return_empty=cpu_to_gpu_only_return_empty,
                preprocess_only_return_empty=preprocess_only_return_empty,
                shm_write_only_return_empty=shm_write_only_return_empty,
                coalesce_log_path=coalesce_log_path,
            )
        return _SINGLETON
