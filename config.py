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
TRITON_URL = f"{TRITON_IP}:{TRITON_GRPC_PORT}"  # GRPC URL
TRITON_HTTP_URL = f"http://{TRITON_IP}:{TRITON_HTTP_PORT}"  # HTTP URL

# ========== Flask 应用配置 ==========
FLASK_HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.environ.get("FLASK_PORT", "3090"))

# 配置算法服务版本以及doc的开关
sufa_version = "2026.06.30"
is_doc = True # 是否开启doc