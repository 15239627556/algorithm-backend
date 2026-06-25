#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

import numpy as np

from model_control import get_loaded_models, load_model, unload_model
from dpi147246_cpu_profile import ACTIVE_PROFILE, DEFAULT_KWARGS, get_profile_kwargs
from dpi147246_cuda_pipeline import BatchedCudaWbcMegPipeline, get_cuda_pipeline
from service_logging import (
    SERVICE_LOG_BACKUP_COUNT,
    SERVICE_LOG_FILE,
    SERVICE_LOG_LEVEL,
    SERVICE_LOG_MAX_MB,
    configure_dedicated_file_logger,
    configure_service_logging,
)
from p147246_lifecycle import (
    finish_error,
    finish_ok,
    new_trace,
    set_field,
    set_result,
    stage_end,
    stage_instant,
    stage_start,
    to_json_line,
    Trace,
)

_P147246 = DEFAULT_KWARGS
from dpi714756_bm_pb_local_pipeline import LocalBmpbPipeline714756
from dpi357378_local_pipeline import LocalMegPipeline357378

# 手动修改区：147246 每张图生命周期日志开关，所有内容都会写入独立 request 日志文件。
# P147246_LOG_REQUESTS：记录每张图从请求进入、payload 读取、decode/preprocess/infer 到结果返回的生命周期。
# P147246_REQUEST_LOG_FILE：P147246_LOG_REQUESTS=True 时写入的独立日志文件。
# P147246_REQUEST_LOG_MAX_MB：单个 request 日志文件最大大小，达到后自动轮转并 gzip 压缩旧文件。
# P147246_REQUEST_LOG_BACKUP_COUNT：最多保留多少个 request 日志压缩备份。
# P147246_REQUEST_LOG_LEVEL：request 日志级别，保持 INFO 即可。
P147246_LOG_REQUESTS = False
P147246_REQUEST_LOG_FILE = "logs/p147246_requests.log"
P147246_REQUEST_LOG_MAX_MB = 50
P147246_REQUEST_LOG_BACKUP_COUNT = 10
P147246_REQUEST_LOG_LEVEL = "INFO"

try:
    import tritonclient.grpc as grpcclient
    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    sys.stderr.write(
        "缺少依赖，请安装: pip install fastapi 'uvicorn[standard]' python-multipart tritonclient[grpc]\n"
    )
    raise

try:
    import uvicorn
except ImportError:  # pragma: no cover
    uvicorn = None  # type: ignore


MODEL_GROUPS = {
    "147246": [
        "DPI147246_BM_PB_WBC_cell_detection",
        "DPI147246_BM_PB_MEG_cell_detection",
        "DPI147246_BM_PB_constituency_score",
        "DPI147246_BM_PB_cell_analysis",
    ],
    "714756": [
        "DPI714756_BM_PB_WBC_detector",
        "DPI714756_BM_PB_WBC_classifier",
        "DPI714756_BM_PB_RED_cell_detection",
        "DPI714756_BM_PB_RED_cell_classifier",
        "DPI714756_BM_PB_PLAT_detection",
    ],
    "357378": [
        "DPI357378_BM_MEG_cell_detection",
        "DPI357378_BM_MEG_cell_classifier",
    ],
}


class _Config:
    triton_grpc_url: str = "localhost:8001"
    triton_http_url: str = "http://localhost:8000"
    host: str = "0.0.0.0"
    port: int = 9000
    # 147246（默认 3970x 预设，见 dpi147246_cpu_profile.py）
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
    p147246_log_requests: bool = False
    # 714756
    p714756_enable_meg: bool = False  # 保留字段，仅用于参数对齐
    p714756_batch_size: int = 8       # WBC 分类子模型推理 batch
    p714756_timeout_ms: int = 10000      # Triton gRPC client_timeout
    p714756_num_slots: int = 1        # 同时运行请求槽位
    p714756_device_id: int = 0        # OpenCV CUDA 设备号（可用时生效）
    p714756_verbose: bool = False
    p714756_bgr_npy_dump_dir: str = ""  # 非空则每张图解码后保存 BGR 为 .npy
    p714756_parallel_preprocess: bool = True  # 预处理并行开关
    p714756_preprocess_workers: int = 8  # 全局 shared 分类 patch 池大小（同 147246 思路，与 num_slots 无关）
    # 357378
    p357378_enable_meg: bool = False  # 保留字段，仅用于参数对齐
    p357378_batch_size: int = 1       # 分类子模型推理 batch
    p357378_timeout_ms: int = 10000      # Triton gRPC client_timeout
    p357378_num_slots: int = 1        # 同时运行请求槽位
    p357378_device_id: int = 0        # OpenCV CUDA 设备号（可用时生效）
    p357378_verbose: bool = False
    p357378_parallel_preprocess: bool = True  # 预处理并行开关
    p357378_preprocess_workers: int = 8  # 全局 shared 分类 patch 池（同 147246 思路，与 num_slots 无关）


_CFG = _Config()
_PIPE_147246: BatchedCudaWbcMegPipeline | None = None
_PIPE_714756: LocalBmpbPipeline714756 | None = None
_PIPE_357378: LocalMegPipeline357378 | None = None
_thread_local_grpc = threading.local()
_PREP_EXEC_714756: ThreadPoolExecutor | None = None
_PREP_EXEC_357378: ThreadPoolExecutor | None = None
_EXEC_714756: ThreadPoolExecutor | None = None
_EXEC_357378: ThreadPoolExecutor | None = None
_SEM_714756: threading.BoundedSemaphore | None = None
_SEM_357378: threading.BoundedSemaphore | None = None
_STAT_LOCK = threading.Lock()
_ACTIVE_714756 = 0
_ACTIVE_357378 = 0
_PEAK_714756 = 0
_PEAK_357378 = 0
_TOTAL_714756 = 0
_TOTAL_357378 = 0


logger = logging.getLogger("multi_pipeline_server")
request_logger = logging.getLogger("multi_pipeline_server.p147246_requests")
_147246_IMAGE_SEQ = 0
_147246_IMAGE_SEQ_LOCK = threading.Lock()


def _next_147246_image_seq() -> int:
    global _147246_IMAGE_SEQ
    with _147246_IMAGE_SEQ_LOCK:
        _147246_IMAGE_SEQ += 1
        return _147246_IMAGE_SEQ


def _err_response(target: str, route: str, e: BaseException) -> JSONResponse:
    """统一 500 响应：服务端记录 traceback；body 同时回带最后几行 trace，便于客户端日志排错。"""
    tb = traceback.format_exc()
    logger.error(
        "[%s][%s] 500 %s: %s\n%s",
        target,
        route,
        type(e).__name__,
        e,
        tb,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": str(e),
            "type": type(e).__name__,
            "trace_tail": tb.splitlines()[-8:],
        },
    )


def _to_jsonable(x: Any) -> Any:
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


def _grpc_client() -> grpcclient.InferenceServerClient:
    c = getattr(_thread_local_grpc, "client", None)
    if c is not None and getattr(_thread_local_grpc, "url", None) == _CFG.triton_grpc_url:
        return c
    c = grpcclient.InferenceServerClient(url=_CFG.triton_grpc_url)
    _thread_local_grpc.client = c
    _thread_local_grpc.url = _CFG.triton_grpc_url
    return c


def _decode_b64(s: str) -> bytes:
    try:
        return base64.b64decode(s)
    except Exception as e:
        raise HTTPException(400, f"base64 decode failed: {e}")


def _slot_stats(total: int, active: int, peak: int, total_requests: int) -> dict:
    used = max(0, active)
    cap = max(1, total)
    return {
        "slots_total": cap,
        "slots_in_use": min(used, cap),
        "slots_available": max(0, cap - used),
        "peak_in_use": min(max(0, peak), cap),
        "total_requests": max(0, total_requests),
    }


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _PIPE_147246, _PIPE_714756, _PIPE_357378
    global _PREP_EXEC_714756, _PREP_EXEC_357378, _EXEC_714756, _EXEC_357378, _SEM_714756, _SEM_357378
    global _ACTIVE_714756, _ACTIVE_357378, _PEAK_714756, _PEAK_357378, _TOTAL_714756, _TOTAL_357378
    with _STAT_LOCK:
        _ACTIVE_714756 = 0
        _ACTIVE_357378 = 0
        _PEAK_714756 = 0
        _PEAK_357378 = 0
        _TOTAL_714756 = 0
        _TOTAL_357378 = 0
    w714prep = max(1, _CFG.p714756_preprocess_workers if _CFG.p714756_parallel_preprocess else 1)
    w357prep = max(1, _CFG.p357378_preprocess_workers if _CFG.p357378_parallel_preprocess else 1)
    # 与 147246 同思路：全进程共享一个「分类 patch 预处理」池；与 run_in_executor 分离，避免与池内再 submit 到同一池死锁。
    _PREP_EXEC_714756 = ThreadPoolExecutor(max_workers=w714prep) if _CFG.p714756_parallel_preprocess else None
    _PREP_EXEC_357378 = ThreadPoolExecutor(max_workers=w357prep) if _CFG.p357378_parallel_preprocess else None
    # async 端每条推理占一线程，按 num_slots 与 sem 一致即可
    _EXEC_714756 = ThreadPoolExecutor(max_workers=max(1, _CFG.p714756_num_slots))
    _EXEC_357378 = ThreadPoolExecutor(max_workers=max(1, _CFG.p357378_num_slots))
    _SEM_714756 = threading.BoundedSemaphore(value=max(1, _CFG.p714756_num_slots))
    _SEM_357378 = threading.BoundedSemaphore(value=max(1, _CFG.p357378_num_slots))
    logger.info(
        "server starting | triton_grpc=%s triton_http=%s p147246=(profile=%s slots=%s batch=%s timeout_ms=%s meg=%s ca=%s) "
        "p714756=(slots=%s batch=%s timeout_ms=%s) p357378=(slots=%s batch=%s timeout_ms=%s)",
        _CFG.triton_grpc_url,
        _CFG.triton_http_url,
        _CFG.p147246_cpu_profile,
        _CFG.p147246_num_slots,
        _CFG.p147246_batch_size,
        _CFG.p147246_timeout_ms,
        _CFG.p147246_enable_meg,
        _CFG.p147246_enable_ca,
        _CFG.p714756_num_slots,
        _CFG.p714756_batch_size,
        _CFG.p714756_timeout_ms,
        _CFG.p357378_num_slots,
        _CFG.p357378_batch_size,
        _CFG.p357378_timeout_ms,
    )
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
    _PIPE_357378 = LocalMegPipeline357378(
        batch_size=_CFG.p357378_batch_size,
        timeout_ms=_CFG.p357378_timeout_ms,
        device_id=_CFG.p357378_device_id,
        parallel_preprocess=_CFG.p357378_parallel_preprocess,
        preprocess_workers=_CFG.p357378_preprocess_workers,
        verbose=_CFG.p357378_verbose,
        preprocess_executor=_PREP_EXEC_357378,
    )
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
    logger.info("server ready")
    try:
        yield
    finally:
        logger.info("server shutting down")
        if _PREP_EXEC_714756 is not None:
            _PREP_EXEC_714756.shutdown(wait=False, cancel_futures=True)
        if _PREP_EXEC_357378 is not None:
            _PREP_EXEC_357378.shutdown(wait=False, cancel_futures=True)
        if _EXEC_714756 is not None:
            _EXEC_714756.shutdown(wait=False, cancel_futures=True)
        if _EXEC_357378 is not None:
            _EXEC_357378.shutdown(wait=False, cancel_futures=True)
        if _PIPE_147246 is not None:
            _PIPE_147246.close()
        _PIPE_714756 = None
        _PIPE_357378 = None
        _SEM_714756 = None
        _SEM_357378 = None
        logger.info("server stopped")


app = FastAPI(
    title="Multi Pipeline Server",
    version="1.0",
    lifespan=_lifespan,
)


class _Req147246(BaseModel):
    image_b64: str
    enable_meg: bool | None = None


class _Req714756(BaseModel):
    image_b64: str
    tasks: str = "wbc"
    task_mode: int | None = None  # 已弃用，兼容旧客户端


class _Req357378(BaseModel):
    image_b64: str


class _ModelActionReq(BaseModel):
    target: str
    timeout: int = 300


@app.get("/health")
def health():
    return {
        "status": "ok" if _PIPE_147246 is not None else "not_ready",
        "triton_grpc_url": _CFG.triton_grpc_url,
        "triton_http_url": _CFG.triton_http_url,
    }


@app.get("/models/status")
def models_status():
    with _STAT_LOCK:
        s714756 = _slot_stats(
            total=_CFG.p714756_num_slots,
            active=_ACTIVE_714756,
            peak=_PEAK_714756,
            total_requests=_TOTAL_714756,
        )
        s357378 = _slot_stats(
            total=_CFG.p357378_num_slots,
            active=_ACTIVE_357378,
            peak=_PEAK_357378,
            total_requests=_TOTAL_357378,
        )
    loaded = set(get_loaded_models(_CFG.triton_http_url))
    return {
        "runtime_config": {
            "147246": {
                "cpu_profile": _CFG.p147246_cpu_profile,
                "enable_meg": _CFG.p147246_enable_meg,
                "enable_ca": _CFG.p147246_enable_ca,
                "batch_size": _CFG.p147246_batch_size,
                "timeout_ms": _CFG.p147246_timeout_ms,
                "num_slots": _CFG.p147246_num_slots,
                "device_id": _CFG.p147246_device_id,
                "verbose": _CFG.p147246_verbose,
                "parallel_preprocess": _CFG.p147246_parallel_preprocess,
                "preprocess_workers": _CFG.p147246_preprocess_workers,
                "decode_workers": _CFG.p147246_decode_workers,
                "heavy_post_mode": _CFG.p147246_heavy_post_mode,
                "heavy_post_workers": _CFG.p147246_heavy_post_workers,
                "log_requests": _CFG.p147246_log_requests,
            },
            "714756": {
                "enable_meg": _CFG.p714756_enable_meg,
                "batch_size": _CFG.p714756_batch_size,
                "timeout_ms": _CFG.p714756_timeout_ms,
                "num_slots": _CFG.p714756_num_slots,
                "device_id": _CFG.p714756_device_id,
                "verbose": _CFG.p714756_verbose,
                "parallel_preprocess": _CFG.p714756_parallel_preprocess,
                "preprocess_workers": _CFG.p714756_preprocess_workers,
                "runtime_slots": s714756,
            },
            "357378": {
                "enable_meg": _CFG.p357378_enable_meg,
                "batch_size": _CFG.p357378_batch_size,
                "timeout_ms": _CFG.p357378_timeout_ms,
                "num_slots": _CFG.p357378_num_slots,
                "device_id": _CFG.p357378_device_id,
                "verbose": _CFG.p357378_verbose,
                "parallel_preprocess": _CFG.p357378_parallel_preprocess,
                "preprocess_workers": _CFG.p357378_preprocess_workers,
                "runtime_slots": s357378,
            },
        },
        "loaded_models": sorted(loaded),
        "targets": {
            t: {
                "required": names,
                "ready": all(n in loaded for n in names),
                "loaded_count": sum(1 for n in names if n in loaded),
            }
            for t, names in MODEL_GROUPS.items()
        },
    }


@app.get("/stats")
def stats_compat():
    """与 pipeline_server 的 GET /stats 兼容（147246 合批统计）。"""
    return stats_147246()


@app.get("/147246/stats")
def stats_147246():
    """暴露 147246 pipeline 内部累计的合批 / 分阶段统计，供 check_coalesce.py 差分使用。

    返回结构与单 target 版 pipeline_server.py 的 /stats 一致。
    """
    p = _PIPE_147246
    if p is None:
        raise HTTPException(503, "147246 pipeline not ready")
    with p._stats_lock:
        n_batch = p._stats_count
        data: dict[str, Any] = {
            "batches": n_batch,
            "images": p._stats_imgs,
            "wbc_total": p._stats_wbc_total,
            "meg_batches": p._stats_meg_batches,
            "avg_batch_wall_time_ms": (
                p._stats_total_ms / n_batch if n_batch else 0.0
            ),
            "num_slots": p._num_slots,
            "batch_size": p._batch_size,
            "timeout_ms": int(p._timeout_s * 1000),
            "enable_ca": p._enable_ca,
            "stages_avg_ms_per_batch": {
                s: (p._stats_stage_sum[s] / n_batch if n_batch else 0.0)
                for s in p._stats_stage_sum
            },
            "coalesce": {
                "full_batches": p._coalesce_full,
                "timeout_batches": p._coalesce_timeout,
                "timeout_batches_only_1": p._coalesce_timeout_n1,
                "by_real_image_count": {
                    str(k): v for k, v in sorted(p._coalesce_by_n.items())
                },
            },
        }
    return data


@app.post("/models/load")
def models_load(req: _ModelActionReq):
    target = str(req.target).strip()
    if target not in MODEL_GROUPS:
        raise HTTPException(400, f"unsupported target: {target}")
    results = []
    for m in MODEL_GROUPS[target]:
        ok, msg = load_model(_CFG.triton_http_url, m, timeout=req.timeout)
        results.append({"model": m, "ok": ok, "msg": msg})
    return {"target": target, "results": results}


@app.post("/models/unload")
def models_unload(req: _ModelActionReq):
    target = str(req.target).strip()
    if target not in MODEL_GROUPS:
        raise HTTPException(400, f"unsupported target: {target}")
    results = []
    for m in MODEL_GROUPS[target]:
        ok, msg = unload_model(_CFG.triton_http_url, m, timeout=req.timeout)
        results.append({"model": m, "ok": ok, "msg": msg})
    return {"target": target, "results": results}


def _infer_147246(image_bytes: bytes, enable_meg: bool | None, trace: Trace | None = None) -> dict:
    if _PIPE_147246 is None:
        raise RuntimeError("147246 pipeline not ready")
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


def _log_147246_infer_error(
    route: str,
    *,
    image_seq: int,
    request_enter_wall: str,
    error: BaseException,
    client_seq: str | None = None,
) -> None:
    error_wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    parts = [
        "147246 infer_error",
        f"image_seq={image_seq}",
        f"route={route}",
        f"request_enter_wall={request_enter_wall}",
        f"error_wall={error_wall}",
        f"error_type={type(error).__name__}",
        f"error={str(error)}",
    ]
    if client_seq:
        parts.append(f"client_seq={client_seq}")
    line = " | ".join(parts)
    logger.warning(line)


def _infer_714756(image_bytes: bytes, tasks: str | None = None, task_mode: int | None = None) -> dict:
    global _ACTIVE_714756, _PEAK_714756, _TOTAL_714756
    if _PIPE_714756 is None:
        raise RuntimeError("714756 pipeline not ready")
    if _SEM_714756 is None:
        raise RuntimeError("714756 slot limiter not ready")
    t0 = time.perf_counter()
    with _SEM_714756:
        with _STAT_LOCK:
            _ACTIVE_714756 += 1
            _TOTAL_714756 += 1
            if _ACTIVE_714756 > _PEAK_714756:
                _PEAK_714756 = _ACTIVE_714756
        try:
            res = _PIPE_714756.run(
                _grpc_client(),
                image_bytes,
                tasks=tasks,
                task_mode=task_mode,
            )
        finally:
            with _STAT_LOCK:
                _ACTIVE_714756 = max(0, _ACTIVE_714756 - 1)
    out = _to_jsonable(res)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    return out


def _infer_357378(image_bytes: bytes) -> dict:
    global _ACTIVE_357378, _PEAK_357378, _TOTAL_357378
    if _PIPE_357378 is None:
        raise RuntimeError("357378 pipeline not ready")
    if _SEM_357378 is None:
        raise RuntimeError("357378 slot limiter not ready")
    t0 = time.perf_counter()
    with _SEM_357378:
        with _STAT_LOCK:
            _ACTIVE_357378 += 1
            _TOTAL_357378 += 1
            if _ACTIVE_357378 > _PEAK_357378:
                _PEAK_357378 = _ACTIVE_357378
        try:
            res = _PIPE_357378.run(_grpc_client(), image_bytes)
        finally:
            with _STAT_LOCK:
                _ACTIVE_357378 = max(0, _ACTIVE_357378 - 1)
    out = _to_jsonable(res)
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    return out


@app.post("/147246/infer_b64")
def infer_147246_b64(req: _Req147246):
    image_seq = _next_147246_image_seq()
    t_enter_ns = time.perf_counter_ns()
    wall_enter = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    trace = (
        new_trace(
            image_seq=image_seq,
            route="infer_b64",
            enable_meg=req.enable_meg,
        )
        if _CFG.p147246_log_requests
        else None
    )
    stage_instant(trace, "request_received")
    try:
        stage_start(trace, "payload_read")
        decode_t0_ns = time.perf_counter_ns()
        data = _decode_b64(req.image_b64)
        decode_ms = round((time.perf_counter_ns() - decode_t0_ns) / 1_000_000.0, 3)
        stage_end(trace, "payload_read")
        set_field(trace, "payload_bytes", len(data))
        stage_start(trace, "pipeline_call")
        out = _infer_147246(data, req.enable_meg, trace)
        stage_end(trace, "pipeline_call")
        compute_done_wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        request_total_ms = round((time.perf_counter_ns() - t_enter_ns) / 1_000_000.0, 3)
        out["payload_bytes"] = len(data)
        out["payload_read_ms"] = decode_ms
        out["request_total_ms"] = request_total_ms
        out["image_seq"] = image_seq
        out["request_enter_wall"] = wall_enter
        out["compute_done_wall"] = compute_done_wall
        set_result(trace, wbc_num=out.get("wbc_num"), meg_num=out.get("meg_num"))
        stage_instant(trace, "response_ready")
        finish_ok(trace, request_total_ms=request_total_ms)
        if trace is not None:
            request_logger.info(to_json_line(trace))
        return out
    except Exception as e:
        request_total_ms = round((time.perf_counter_ns() - t_enter_ns) / 1_000_000.0, 3)
        stage_instant(trace, "response_ready")
        finish_error(trace, e, request_total_ms=request_total_ms)
        if trace is not None:
            request_logger.info(to_json_line(trace))
        _log_147246_infer_error(
            "infer_b64",
            image_seq=image_seq,
            request_enter_wall=wall_enter,
            error=e,
        )
        return _err_response("147246", "infer_b64", e)


@app.post("/147246/infer")
async def infer_147246(
    request: Request,
    image: UploadFile = File(...),
    enable_meg: int | None = Form(None),
):
    image_seq = _next_147246_image_seq()
    t_enter_ns = time.perf_counter_ns()
    wall_enter = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    client_seq = request.headers.get("x-client-seq")
    want_meg = None if enable_meg is None else bool(enable_meg)
    trace = (
        new_trace(
            image_seq=image_seq,
            route="infer",
            client_seq=client_seq,
            filename=image.filename,
            enable_meg=want_meg,
        )
        if _CFG.p147246_log_requests
        else None
    )
    stage_instant(trace, "request_received")
    try:
        stage_start(trace, "payload_read")
        read_t0_ns = time.perf_counter_ns()
        data = await image.read()
        t_recv_done_ns = time.perf_counter_ns()
        payload_read_ms = round((t_recv_done_ns - read_t0_ns) / 1_000_000.0, 3)
        stage_end(trace, "payload_read")
        if not data:
            raise HTTPException(400, "empty image payload")
        set_field(trace, "payload_bytes", len(data))
        t_infer_start_ns = time.perf_counter_ns()
        stage_start(trace, "pipeline_call")
        out = await asyncio.to_thread(_infer_147246, data, want_meg, trace)
        stage_end(trace, "pipeline_call")
        t_done_ns = time.perf_counter_ns()
        compute_done_wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        request_total_ms = round((t_done_ns - t_enter_ns) / 1_000_000.0, 3)
        out["payload_bytes"] = len(data)
        out["request_total_ms"] = request_total_ms
        out["image_seq"] = image_seq
        out["request_enter_wall"] = wall_enter
        out["compute_done_wall"] = compute_done_wall
        # 近似表示图片上传/读取阶段耗时（受客户端网络+服务端multipart解析影响）
        out["payload_read_ms"] = payload_read_ms
        out["net_timing"] = {
            "client_seq": client_seq,
            "server_enter_wall": wall_enter,
            "compute_done_wall": compute_done_wall,
            "server_enter_to_recv_done_ms": round((t_recv_done_ns - t_enter_ns) / 1_000_000.0, 3),
            "body_read_await_ms": payload_read_ms,
            "recv_done_to_infer_start_ms": round((t_infer_start_ns - t_recv_done_ns) / 1_000_000.0, 3),
            "infer_and_response_build_ms": round((t_done_ns - t_infer_start_ns) / 1_000_000.0, 3),
        }
        set_result(trace, wbc_num=out.get("wbc_num"), meg_num=out.get("meg_num"))
        stage_instant(trace, "response_ready")
        finish_ok(trace, request_total_ms=request_total_ms)
        if trace is not None:
            request_logger.info(to_json_line(trace))
        return out
    except Exception as e:
        request_total_ms = round((time.perf_counter_ns() - t_enter_ns) / 1_000_000.0, 3)
        stage_instant(trace, "response_ready")
        finish_error(trace, e, request_total_ms=request_total_ms)
        if trace is not None:
            request_logger.info(to_json_line(trace))
        _log_147246_infer_error(
            "infer",
            image_seq=image_seq,
            request_enter_wall=wall_enter,
            error=e,
            client_seq=client_seq,
        )
        return _err_response("147246", "infer", e)


@app.post("/714756/infer_b64")
def infer_714756_b64(req: _Req714756):
    data = _decode_b64(req.image_b64)
    try:
        sel_tasks = None if req.task_mode is not None else req.tasks
        return _infer_714756(data, tasks=sel_tasks, task_mode=req.task_mode)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        return _err_response("714756", "infer_b64", e)


@app.post("/714756/infer")
async def infer_714756(
    image: UploadFile = File(...),
    tasks: str = Form("wbc"),
    task_mode: int | None = Form(None),
):
    data = await image.read()
    if not data:
        raise HTTPException(400, "empty image payload")
    try:
        loop = asyncio.get_running_loop()
        if _EXEC_714756 is None:
            raise RuntimeError("714756 executor not ready")
        sel_tasks = None if task_mode is not None else tasks
        return await loop.run_in_executor(
            _EXEC_714756,
            _infer_714756,
            data,
            sel_tasks,
            task_mode,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        return _err_response("714756", "infer", e)


@app.post("/357378/infer_b64")
def infer_357378_b64(req: _Req357378):
    data = _decode_b64(req.image_b64)
    try:
        return _infer_357378(data)
    except Exception as e:
        return _err_response("357378", "infer_b64", e)


@app.post("/357378/infer")
async def infer_357378(image: UploadFile = File(...)):
    data = await image.read()
    if not data:
        raise HTTPException(400, "empty image payload")
    try:
        loop = asyncio.get_running_loop()
        if _EXEC_357378 is None:
            raise RuntimeError("357378 executor not ready")
        return await loop.run_in_executor(_EXEC_357378, _infer_357378, data)
    except Exception as e:
        return _err_response("357378", "infer", e)


def _parse_args() -> argparse.Namespace:
    from dpi147246_cpu_profile import resolve_cpu_profile

    prof_name = resolve_cpu_profile(os.environ.get("DPI147246_CPU_PROFILE"))
    prof = get_profile_kwargs(prof_name)

    p = argparse.ArgumentParser()
    p.add_argument(
        "--p147246-cpu-profile",
        default=prof_name,
        choices=["3970x", "10900x"],
        help="147246 CPU 预设（或环境变量 DPI147246_CPU_PROFILE）",
    )
    p.add_argument("--triton-grpc-url", default=os.environ.get("TRITON_GRPC_URL", "localhost:8001"))
    p.add_argument("--triton-http-url", default=os.environ.get("TRITON_HTTP_URL", "http://localhost:8000"))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("MULTI_PIPELINE_PORT", 9000)))
    p.add_argument("--p147246-enable-meg", action="store_true", default=True)
    p.add_argument("--p147246-enable-ca", action="store_true", default=prof["enable_ca"])
    p.add_argument("--p147246-disable-ca", dest="p147246_enable_ca", action="store_false")
    p.add_argument("--p147246-disable-meg", dest="p147246_enable_meg", action="store_false")
    p.add_argument("--p147246-num-slots", type=int, default=prof["num_slots"])
    p.add_argument("--p147246-batch-size", type=int, default=prof["batch_size"])
    p.add_argument("--p147246-timeout-ms", type=int, default=prof["timeout_ms"])
    p.add_argument("--p147246-device-id", type=int, default=0)
    p.add_argument("--p147246-verbose", action="store_true")
    p.add_argument("--p147246-parallel-preprocess", action="store_true", default=True)
    p.add_argument("--p147246-disable-parallel-preprocess", dest="p147246_parallel_preprocess", action="store_false")
    p.add_argument("--p147246-preprocess-workers", type=int, default=prof["preprocess_workers"])
    p.add_argument("--p147246-decode-workers", type=int, default=prof["decode_workers"])
    p.add_argument(
        "--p147246-heavy-post-mode",
        choices=["serial", "thread", "process"],
        default=prof.get("heavy_post_mode", "serial"),
    )
    p.add_argument(
        "--p147246-heavy-post-workers",
        type=int,
        default=prof.get("heavy_post_workers", prof["num_slots"]),
    )
    p.add_argument("--p714756-enable-meg", action="store_true", default=False)
    p.add_argument("--p714756-batch-size", type=int, default=8)
    p.add_argument("--p714756-timeout-ms", type=int, default=10000)
    p.add_argument("--p714756-num-slots", type=int, default=1)
    p.add_argument("--p714756-device-id", type=int, default=0)
    p.add_argument("--p714756-verbose", action="store_true")
    p.add_argument(
        "--p714756-bgr-npy-dir",
        default="",
        help="解码后 BGR 保存目录（.npy）；空且 --p714756-verbose 时默认 debug_bgr_npy_714756",
    )
    p.add_argument("--p714756-parallel-preprocess", action="store_true", default=True)
    p.add_argument("--p714756-disable-parallel-preprocess", dest="p714756_parallel_preprocess", action="store_false")
    p.add_argument("--p714756-preprocess-workers", type=int, default=8)

    p.add_argument("--p357378-enable-meg", action="store_true", default=False)
    p.add_argument("--p357378-batch-size", type=int, default=1)
    p.add_argument("--p357378-timeout-ms", type=int, default=10000)
    p.add_argument("--p357378-num-slots", type=int, default=1)
    p.add_argument("--p357378-device-id", type=int, default=0)
    p.add_argument("--p357378-verbose", action="store_true")
    p.add_argument("--p357378-parallel-preprocess", action="store_true", default=True)
    p.add_argument("--p357378-disable-parallel-preprocess", dest="p357378_parallel_preprocess", action="store_false")
    p.add_argument("--p357378-preprocess-workers", type=int, default=8)
    return p.parse_args()


def main():
    args = _parse_args()
    log_path = configure_service_logging(
        log_file=SERVICE_LOG_FILE,
        log_max_mb=SERVICE_LOG_MAX_MB,
        log_backup_count=SERVICE_LOG_BACKUP_COUNT,
        log_level=SERVICE_LOG_LEVEL,
    )
    logger.info(
        "logging initialized | file=%s max_mb=%s backups=%s level=%s",
        log_path,
        SERVICE_LOG_MAX_MB,
        SERVICE_LOG_BACKUP_COUNT,
        SERVICE_LOG_LEVEL,
    )
    if P147246_LOG_REQUESTS:
        request_log_path = configure_dedicated_file_logger(
            "multi_pipeline_server.p147246_requests",
            log_file=P147246_REQUEST_LOG_FILE,
            log_max_mb=P147246_REQUEST_LOG_MAX_MB,
            log_backup_count=P147246_REQUEST_LOG_BACKUP_COUNT,
            log_level=P147246_REQUEST_LOG_LEVEL,
        )
        logger.info(
            "p147246 request logging initialized | file=%s max_mb=%s backups=%s level=%s",
            request_log_path,
            P147246_REQUEST_LOG_MAX_MB,
            P147246_REQUEST_LOG_BACKUP_COUNT,
            P147246_REQUEST_LOG_LEVEL,
        )
    _CFG.triton_grpc_url = args.triton_grpc_url
    _CFG.triton_http_url = args.triton_http_url.rstrip("/")
    _CFG.host = args.host
    _CFG.port = args.port
    from dpi147246_cpu_profile import resolve_cpu_profile as _resolve_cpu_profile

    _CFG.p147246_cpu_profile = _resolve_cpu_profile(args.p147246_cpu_profile)
    _CFG.p147246_enable_meg = args.p147246_enable_meg
    _CFG.p147246_enable_ca = args.p147246_enable_ca
    _CFG.p147246_num_slots = args.p147246_num_slots
    _CFG.p147246_batch_size = args.p147246_batch_size
    _CFG.p147246_timeout_ms = args.p147246_timeout_ms
    _CFG.p147246_device_id = args.p147246_device_id
    _CFG.p147246_verbose = args.p147246_verbose
    _CFG.p147246_parallel_preprocess = args.p147246_parallel_preprocess
    _CFG.p147246_preprocess_workers = max(1, args.p147246_preprocess_workers)
    _CFG.p147246_decode_workers = max(1, args.p147246_decode_workers)
    _CFG.p147246_heavy_post_mode = args.p147246_heavy_post_mode
    _CFG.p147246_heavy_post_workers = max(1, args.p147246_heavy_post_workers)
    _CFG.p147246_log_requests = bool(P147246_LOG_REQUESTS)

    _CFG.p714756_enable_meg = args.p714756_enable_meg
    _CFG.p714756_batch_size = args.p714756_batch_size
    _CFG.p714756_timeout_ms = args.p714756_timeout_ms
    _CFG.p714756_num_slots = args.p714756_num_slots
    _CFG.p714756_device_id = args.p714756_device_id
    _CFG.p714756_verbose = args.p714756_verbose
    _CFG.p714756_bgr_npy_dump_dir = (args.p714756_bgr_npy_dir or "").strip()
    _CFG.p714756_parallel_preprocess = args.p714756_parallel_preprocess
    _CFG.p714756_preprocess_workers = max(1, args.p714756_preprocess_workers)

    _CFG.p357378_enable_meg = args.p357378_enable_meg
    _CFG.p357378_batch_size = args.p357378_batch_size
    _CFG.p357378_timeout_ms = args.p357378_timeout_ms
    _CFG.p357378_num_slots = args.p357378_num_slots
    _CFG.p357378_device_id = args.p357378_device_id
    _CFG.p357378_verbose = args.p357378_verbose
    _CFG.p357378_parallel_preprocess = args.p357378_parallel_preprocess
    _CFG.p357378_preprocess_workers = max(1, args.p357378_preprocess_workers)

    if uvicorn is None:
        sys.stderr.write("uvicorn 未安装：pip install 'uvicorn[standard]'\n")
        sys.exit(1)
    uvicorn.run(
        app,
        host=_CFG.host,
        port=_CFG.port,
        workers=1,
        loop="asyncio",
        http="h11",
        log_level="info" if (_CFG.p147246_verbose or _CFG.p714756_verbose or _CFG.p357378_verbose) else "warning",
        access_log=False,
        log_config=None,
    )


if __name__ == "__main__":
    main()
