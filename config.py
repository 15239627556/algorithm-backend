# -*- coding: utf-8 -*-
"""
应用配置文件。Triton 推理服务地址、端口等。

TRITON_IPS：端点主机列表，顺序即 gpu_id；定点模式用 TRITON_GPU_ID 取下标。
默认关闭轮询（TRITON_ROUND_ROBIN=0）；设为 1 时在列表内交替发请求。
"""
from __future__ import annotations

import os
import threading


def _endpoint(
    *,
    name: str,
    ip: str,
    http_port: int,
    grpc_port: int,
    metrics_port: int,
    multi_pipeline_port: int,
) -> dict:
    return {
        "name": name,
        "ip": ip,
        "http_port": http_port,
        "grpc_port": grpc_port,
        "metrics_port": metrics_port,
        "multi_pipeline_port": multi_pipeline_port,
        "url": f"{ip}:{grpc_port}",  # gRPC
        "http_url": f"http://{ip}:{http_port}",
        "pipeline_base_url": f"http://{ip}:{multi_pipeline_port}",
    }


# ========== Triton ==========
# 端点主机列表（改这里即可；顺序即 gpu_id，定点默认用 [0]）
TRITON_IPS = [
    "triton_gpu0",
    # "triton_gpu1",
]

# 各端点共用端口：8000(HTTP)、8001(gRPC)、8002(Metrics)、9000(multi_pipeline)
TRITON_HTTP_PORT = 8000
TRITON_GRPC_PORT = 8001
TRITON_METRICS_PORT = 8002
MULTI_PIPELINE_PORT = 9000

TRITON_ENDPOINTS = [
    _endpoint(
        name=f"triton_gpu{i}",
        ip=ip,
        http_port=TRITON_HTTP_PORT,
        grpc_port=TRITON_GRPC_PORT,
        metrics_port=TRITON_METRICS_PORT,
        multi_pipeline_port=MULTI_PIPELINE_PORT,
    )
    for i, ip in enumerate(TRITON_IPS)
]

# 兼容旧引用（列表第一项）
TRITON_IP = TRITON_IPS[0]
TRITON_URL = f"{TRITON_IP}:{TRITON_GRPC_PORT}"
TRITON_HTTP_URL = f"http://{TRITON_IP}:{TRITON_HTTP_PORT}"
PIPELINE_SERVER_BASE_URL = f"http://{TRITON_IP}:{MULTI_PIPELINE_PORT}"

# 轮询开关：默认关闭（定点 TRITON_GPU_ID）。设为 1/true 时多端点交替发请求。
_TRITON_RR_RAW = os.environ.get("TRITON_ROUND_ROBIN", "0").strip().lower()
TRITON_ROUND_ROBIN = _TRITON_RR_RAW not in ("0", "false", "no", "off")

# 关闭轮询时的定点 GPU：对应 TRITON_IPS 下标，0=列表第一项
TRITON_GPU_ID = int(os.environ.get("TRITON_GPU_ID", "0"))

_rr_lock = threading.Lock()
_rr_index = 0


def get_triton_endpoint(gpu_id: int | None = None) -> dict:
    """按 gpu_id 取端点；None 时用 TRITON_GPU_ID。越界则回退到 gpu0。"""
    idx = TRITON_GPU_ID if gpu_id is None else int(gpu_id)
    if idx < 0 or idx >= len(TRITON_ENDPOINTS):
        idx = 0
    return TRITON_ENDPOINTS[idx]


def next_triton_endpoint() -> tuple[int, dict]:
    """
    线程安全轮询下一个端点。
    返回 (gpu_id, endpoint)。仅一个端点或关闭轮询时固定返回 TRITON_GPU_ID。
    """
    global _rr_index
    n = len(TRITON_ENDPOINTS)
    if n <= 0:
        raise RuntimeError("TRITON_ENDPOINTS is empty")
    if not TRITON_ROUND_ROBIN or n == 1:
        idx = TRITON_GPU_ID if 0 <= TRITON_GPU_ID < n else 0
        return idx, TRITON_ENDPOINTS[idx]
    with _rr_lock:
        idx = _rr_index % n
        _rr_index = (idx + 1) % n
    return idx, TRITON_ENDPOINTS[idx]


# ========== 应用配置（FastAPI / uvicorn）==========
APP_HOST = os.environ.get("APP_HOST", os.environ.get("FLASK_HOST", "0.0.0.0"))
APP_PORT = int(os.environ.get("APP_PORT", os.environ.get("FLASK_PORT", "3090")))
FLASK_HOST = APP_HOST
FLASK_PORT = APP_PORT

THREAD_POOL_SIZE = int(os.environ.get("THREAD_POOL_SIZE", "256"))

sufa_version = "v42"
is_doc = True
