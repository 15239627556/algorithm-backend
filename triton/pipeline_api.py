#!/usr/bin/env python3
"""进程内算法管线接口（替代 multi_pipeline_server 的 HTTP 服务）。

后端可直接 ``import pipeline_api``，在进程内以 ``image_bytes`` 调用三套管线，
不再需要单独启动 multi_pipeline_server（FastAPI）。真正的深度学习子模型仍由
Triton server 提供（gRPC），本模块只负责在客户端进程内做解码 / 预处理 / 编排。

三套管线（147246 / 357378 / 714756）相互独立，可按需单独初始化，只用到哪套就构建
哪套，避免一次性把三套的管线对象 / 预处理线程池 / 信号量全部建好造成资源浪费。

对外接口:
    configure(**overrides)           # 仅更新运行参数（如 triton_grpc_url），不构建管线
    init_147246() / init_357378() / init_714756()   # 单独构建某一套，幂等
    init_target(target)              # 按 target/别名/DPI 单独构建某一套
    close(target=None)               # 释放资源：None 全部，否则仅指定套
    is_ready(target=None) -> bool    # None 表示三套都就绪
    infer_147246(image_bytes, enable_meg=None) -> dict   # 需先 init_147246()，未初始化将报错
    infer_357378(image_bytes) -> dict                    # 需先 init_357378()
    infer_714756(image_bytes, tasks=None, task_mode=None) -> dict  # 需先 init_714756()

返回的 dict 结构与原 multi_pipeline_server 的 JSON 完全一致，可直接被
backend.tools.triton_client 的解析函数复用。
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from dpi147246_cpu_profile import ACTIVE_PROFILE, DEFAULT_KWARGS
from dpi147246_cuda_pipeline import BatchedCudaWbcMegPipeline, get_cuda_pipeline
from dpi714756_bm_pb_local_pipeline import LocalBmpbPipeline714756
from dpi357378_local_pipeline import LocalMegPipeline357378

try:
    import tritonclient.grpc as grpcclient
except ImportError:  # pragma: no cover - 友好报错
    raise ImportError(
        "缺少依赖 tritonclient[grpc]，请安装: pip install 'tritonclient[grpc]'"
    )

logger = logging.getLogger("pipeline_api")

_P147246 = DEFAULT_KWARGS


class PipelineConfig:
    """与 multi_pipeline_server._Config 对齐的运行参数（可被 init() 覆盖）。"""

    # Triton 推理地址（gRPC 用于推理，HTTP 仅用于可选的模型状态查询）
    triton_grpc_url: str = "localhost:8001"
    triton_http_url: str = "http://localhost:8000"

    # 147246（默认随 dpi147246_cpu_profile 预设）
    p147246_cpu_profile: str = ACTIVE_PROFILE
    p147246_enable_meg: bool = True
    p147246_enable_ca: bool = _P147246["enable_ca"]
    p147246_batch_size: int = _P147246["batch_size"]
    p147246_timeout_ms: int = _P147246["timeout_ms"]
    p147246_num_slots: int = _P147246["num_slots"]
    p147246_device_id: int = 0
    p147246_verbose: bool = False
    p147246_parallel_preprocess: bool = True
    p147246_preprocess_workers: int = _P147246["preprocess_workers"]
    p147246_decode_workers: int = _P147246["decode_workers"]
    p147246_heavy_post_mode: str = _P147246.get("heavy_post_mode", "serial")
    p147246_heavy_post_workers: int = _P147246.get("heavy_post_workers", _P147246["num_slots"])

    # 714756
    p714756_batch_size: int = 8
    p714756_timeout_ms: int = 10000
    p714756_num_slots: int = 1
    p714756_device_id: int = 0
    p714756_verbose: bool = False
    p714756_bgr_npy_dump_dir: str = ""
    p714756_parallel_preprocess: bool = True
    p714756_preprocess_workers: int = 8

    # 357378
    p357378_batch_size: int = 1
    p357378_timeout_ms: int = 10000
    p357378_num_slots: int = 1
    p357378_device_id: int = 0
    p357378_verbose: bool = False
    p357378_parallel_preprocess: bool = True
    p357378_preprocess_workers: int = 8


_CFG = PipelineConfig()

_PIPE_147246: BatchedCudaWbcMegPipeline | None = None
_PIPE_714756: LocalBmpbPipeline714756 | None = None
_PIPE_357378: LocalMegPipeline357378 | None = None

_thread_local_grpc = threading.local()
_PREP_EXEC_714756: ThreadPoolExecutor | None = None
_PREP_EXEC_357378: ThreadPoolExecutor | None = None
# 限制 714756 / 357378 同时在跑的请求数，避免 GPU 过载（与原服务 num_slots 一致）
_SEM_714756: threading.BoundedSemaphore | None = None
_SEM_357378: threading.BoundedSemaphore | None = None

VALID_TARGETS = ("147246", "357378", "714756")
# 兼容别名：模型名 / DPI 整数 等映射到内部 target
_TARGET_ALIASES = {
    "144750": "147246",
    "DPI147246_BM_PB_pipeline": "147246",
    "DPI357378_BM_MEG_pipeline": "357378",
    "DPI714756_BM_PB_pipeline": "714756",
}

# 每个 target 独立初始化，互不影响（只用到哪套就构建哪套，避免浪费资源）。
_INIT_LOCK = threading.RLock()
_INIT_147246 = False
_INIT_357378 = False
_INIT_714756 = False


def _grpc_client() -> "grpcclient.InferenceServerClient":
    """线程内复用的 Triton gRPC 客户端（714756 / 357378 使用）。"""
    c = getattr(_thread_local_grpc, "client", None)
    if c is not None and getattr(_thread_local_grpc, "url", None) == _CFG.triton_grpc_url:
        return c
    c = grpcclient.InferenceServerClient(url=_CFG.triton_grpc_url)
    _thread_local_grpc.client = c
    _thread_local_grpc.url = _CFG.triton_grpc_url
    return c


def _to_jsonable(x: Any) -> Any:
    """numpy -> 原生 python，保持与原 HTTP JSON 完全一致的返回结构。"""
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return []
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def normalize_target(target: Any) -> str:
    """把 target（如 "714756" / DPI 144750 / 模型名）规整为内部 target 字符串。"""
    t = str(target).strip()
    t = _TARGET_ALIASES.get(t, t)
    if t not in VALID_TARGETS:
        raise ValueError(f"未知 target: {target!r}（可选 {VALID_TARGETS} 或其别名）")
    return t


def init_target(target: Any) -> None:
    """按 target（支持别名/DPI）单独构建对应那一套管线。幂等。"""
    {"147246": init_147246, "357378": init_357378, "714756": init_714756}[normalize_target(target)]()


def configure(**overrides: Any) -> None:
    """更新运行参数（PipelineConfig 字段），不构建任何管线。

    注意：仅影响“之后才初始化”的管线；已构建的管线不会因此重建。
    常用: triton_grpc_url、triton_http_url、p*_num_slots、p*_batch_size 等。
    """
    with _INIT_LOCK:
        for key, value in overrides.items():
            if value is None:
                continue
            if not hasattr(_CFG, key):
                logger.warning("pipeline_api.configure: 未知配置项 %s 已忽略", key)
                continue
            setattr(_CFG, key, value)


def init_147246() -> None:
    """仅构建 147246（BM/PB WBC+MEG）管线。幂等。"""
    global _INIT_147246, _PIPE_147246
    with _INIT_LOCK:
        if _INIT_147246:
            return
        _PIPE_147246 = get_cuda_pipeline(
            url=_CFG.triton_grpc_url,
            enable_meg=_CFG.p147246_enable_meg,
            enable_ca=_CFG.p147246_enable_ca,
            batch_size=_CFG.p147246_batch_size,
            timeout_ms=_CFG.p147246_timeout_ms,
            num_slots=_CFG.p147246_num_slots,
            device_id=_CFG.p147246_device_id,
            verbose=_CFG.p147246_verbose,
            parallel_preprocess=_CFG.p147246_parallel_preprocess,
            preprocess_workers=_CFG.p147246_preprocess_workers,
            decode_workers=_CFG.p147246_decode_workers,
            heavy_post_mode=_CFG.p147246_heavy_post_mode,
            heavy_post_workers=_CFG.p147246_heavy_post_workers,
        )
        _INIT_147246 = True
        logger.info(
            "pipeline_api 147246 ready | grpc=%s profile=%s slots=%s batch=%s timeout_ms=%s meg=%s ca=%s",
            _CFG.triton_grpc_url, _CFG.p147246_cpu_profile, _CFG.p147246_num_slots,
            _CFG.p147246_batch_size, _CFG.p147246_timeout_ms,
            _CFG.p147246_enable_meg, _CFG.p147246_enable_ca,
        )


def init_714756() -> None:
    """仅构建 714756（BM/PB WBC/RED/PLAT）管线 + 预处理线程池 + 信号量。幂等。"""
    global _INIT_714756, _PIPE_714756, _PREP_EXEC_714756, _SEM_714756
    with _INIT_LOCK:
        if _INIT_714756:
            return
        w = max(1, _CFG.p714756_preprocess_workers if _CFG.p714756_parallel_preprocess else 1)
        _PREP_EXEC_714756 = ThreadPoolExecutor(max_workers=w) if _CFG.p714756_parallel_preprocess else None
        _SEM_714756 = threading.BoundedSemaphore(value=max(1, _CFG.p714756_num_slots))
        _PIPE_714756 = LocalBmpbPipeline714756(
            batch_size=_CFG.p714756_batch_size,
            timeout_ms=_CFG.p714756_timeout_ms,
            device_id=_CFG.p714756_device_id,
            parallel_preprocess=_CFG.p714756_parallel_preprocess,
            preprocess_workers=_CFG.p714756_preprocess_workers,
            verbose=_CFG.p714756_verbose,
            bgr_npy_dump_dir=_CFG.p714756_bgr_npy_dump_dir or None,
            preprocess_executor=_PREP_EXEC_714756,
        )
        _INIT_714756 = True
        logger.info(
            "pipeline_api 714756 ready | grpc=%s slots=%s batch=%s timeout_ms=%s",
            _CFG.triton_grpc_url, _CFG.p714756_num_slots,
            _CFG.p714756_batch_size, _CFG.p714756_timeout_ms,
        )


def init_357378() -> None:
    """仅构建 357378（BM 巨核细胞定位分类）管线 + 预处理线程池 + 信号量。幂等。"""
    global _INIT_357378, _PIPE_357378, _PREP_EXEC_357378, _SEM_357378
    with _INIT_LOCK:
        if _INIT_357378:
            return
        w = max(1, _CFG.p357378_preprocess_workers if _CFG.p357378_parallel_preprocess else 1)
        _PREP_EXEC_357378 = ThreadPoolExecutor(max_workers=w) if _CFG.p357378_parallel_preprocess else None
        _SEM_357378 = threading.BoundedSemaphore(value=max(1, _CFG.p357378_num_slots))
        _PIPE_357378 = LocalMegPipeline357378(
            batch_size=_CFG.p357378_batch_size,
            timeout_ms=_CFG.p357378_timeout_ms,
            device_id=_CFG.p357378_device_id,
            parallel_preprocess=_CFG.p357378_parallel_preprocess,
            preprocess_workers=_CFG.p357378_preprocess_workers,
            verbose=_CFG.p357378_verbose,
            preprocess_executor=_PREP_EXEC_357378,
        )
        _INIT_357378 = True
        logger.info(
            "pipeline_api 357378 ready | grpc=%s slots=%s batch=%s timeout_ms=%s",
            _CFG.triton_grpc_url, _CFG.p357378_num_slots,
            _CFG.p357378_batch_size, _CFG.p357378_timeout_ms,
        )


def is_ready(target: Any = None) -> bool:
    """查询初始化状态：target=None 时所有三套都就绪才为 True；否则查询指定 target。"""
    flags = {"147246": _INIT_147246, "357378": _INIT_357378, "714756": _INIT_714756}
    if target is None:
        return all(flags.values())
    return flags[normalize_target(target)]


def close(target: Any = None) -> None:
    """释放资源：target=None 关闭全部；否则仅关闭指定 target。"""
    global _INIT_147246, _INIT_357378, _INIT_714756
    global _PIPE_147246, _PIPE_714756, _PIPE_357378
    global _PREP_EXEC_714756, _PREP_EXEC_357378, _SEM_714756, _SEM_357378

    sel = list(VALID_TARGETS) if target is None else [normalize_target(target)]
    with _INIT_LOCK:
        if "147246" in sel and _INIT_147246:
            if _PIPE_147246 is not None:
                try:
                    _PIPE_147246.close()
                except Exception:
                    logger.exception("147246 pipeline close 失败")
            _PIPE_147246 = None
            _INIT_147246 = False
            logger.info("pipeline_api 147246 stopped")
        if "714756" in sel and _INIT_714756:
            if _PREP_EXEC_714756 is not None:
                _PREP_EXEC_714756.shutdown(wait=False, cancel_futures=True)
            _PIPE_714756 = None
            _PREP_EXEC_714756 = None
            _SEM_714756 = None
            _INIT_714756 = False
            logger.info("pipeline_api 714756 stopped")
        if "357378" in sel and _INIT_357378:
            if _PREP_EXEC_357378 is not None:
                _PREP_EXEC_357378.shutdown(wait=False, cancel_futures=True)
            _PIPE_357378 = None
            _PREP_EXEC_357378 = None
            _SEM_357378 = None
            _INIT_357378 = False
            logger.info("pipeline_api 357378 stopped")


def infer_147246(image_bytes: bytes, enable_meg: bool | None = None, trace: Any = None) -> dict:
    """147246（BM/PB WBC+MEG）推理。需先调用 init_147246()。返回结构同原 /147246/infer。"""
    if not _INIT_147246:
        raise RuntimeError("147246 管线未初始化，请先调用 pipeline_api.init_147246()")
    t0 = time.perf_counter()
    res, timing = _PIPE_147246.submit(
        image_bytes,
        enable_meg=enable_meg,
        return_timing=True,
        lifecycle_trace=trace,
    )
    out = _to_jsonable(res)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    out["timing"] = timing
    return out


def infer_714756(image_bytes: bytes, tasks: str | None = None, task_mode: int | None = None) -> dict:
    """714756（BM/PB WBC/RED/PLAT）推理。需先调用 init_714756()。返回结构同原 /714756/infer。"""
    if not _INIT_714756:
        raise RuntimeError("714756 管线未初始化，请先调用 pipeline_api.init_714756()")
    t0 = time.perf_counter()
    with _SEM_714756:
        res = _PIPE_714756.run(
            _grpc_client(),
            image_bytes,
            tasks=tasks,
            task_mode=task_mode,
        )
    out = _to_jsonable(res)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    return out


def infer_357378(image_bytes: bytes) -> dict:
    """357378（BM 巨核细胞定位分类）推理。需先调用 init_357378()。返回结构同原 /357378/infer。"""
    if not _INIT_357378:
        raise RuntimeError("357378 管线未初始化，请先调用 pipeline_api.init_357378()")
    t0 = time.perf_counter()
    with _SEM_357378:
        res = _PIPE_357378.run(_grpc_client(), image_bytes)
    out = _to_jsonable(res)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    return out
