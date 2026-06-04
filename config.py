# -*- coding: utf-8 -*-
"""
应用配置文件。Triton 推理服务地址、端口、常驻 pipeline 名称等，可通过环境变量覆盖以适配不同部署环境。
"""
import os

# ========== Triton 推理服务配置 ==========
# 默认：本机 127.0.0.1
# 端口：8000(HTTP 模型加载/卸载)、8001(gRPC 推理)、8002(Metrics)
TRITON_IP = os.environ.get("TRITON_IP", "127.0.0.1")
TRITON_HTTP_PORT = int(os.environ.get("TRITON_HTTP_PORT", "8000"))
TRITON_GRPC_PORT = int(os.environ.get("TRITON_GRPC_PORT", "8001"))
TRITON_METRICS_PORT = int(os.environ.get("TRITON_METRICS_PORT", "8002"))

# 若设置了完整 URL，则优先使用（覆盖上述 IP/端口）
# TRITON_URL: gRPC 地址，如 "127.0.0.1:8001" 或 "192.168.1.100:18001"
# TRITON_HTTP_URL: HTTP 地址，如 "http://127.0.0.1:8000" 或 "http://192.168.1.100:18000"
_TRITON_URL_ENV = os.environ.get("TRITON_URL")
_TRITON_HTTP_URL_ENV = os.environ.get("TRITON_HTTP_URL")

if _TRITON_URL_ENV:
    TRITON_URL = _TRITON_URL_ENV
else:
    TRITON_URL = f"{TRITON_IP}:{TRITON_GRPC_PORT}"

if _TRITON_HTTP_URL_ENV:
    TRITON_HTTP_URL = _TRITON_HTTP_URL_ENV.rstrip("/")
else:
    TRITON_HTTP_URL = f"http://{TRITON_IP}:{TRITON_HTTP_PORT}"

# 额外常驻 pipeline（不参与 LRU）；启动时预载列表见 backend.tools.model_control.STARTUP_WARMUP_PIPELINES，也可通过 TRITON_PINNED_PIPELINE_NAME 追加一组
TRITON_PINNED_PIPELINE_NAME = os.environ.get(
    "TRITON_PINNED_PIPELINE_NAME", "DPI147246_BM_PB_pipeline"
)

# ========== Flask 应用配置 ==========
FLASK_HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.environ.get("FLASK_PORT", "3090"))

# 配置算法服务版本以及doc的开关
sufa_version = "2026.06.04"
is_doc = True # 是否开启doc