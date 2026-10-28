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

# 骨髓增生原图路径前缀
cellularity_file_path_prefix = os.environ.get("CELLULARITY_FILE_PATH_PREFIX", "/mnt/d/data/result")

# flir相机   "camera": "flir"
# 其它相机 "camera": "default"
camera = os.environ.get("CAMERA", "flir")

# 红细胞开关（环境变量 RBC_SWITCH: 1/true/on/yes 开启，0/false/no/off 关闭，默认 True）
_rbc_switch_raw = os.environ.get("RBC_SWITCH")
if _rbc_switch_raw is None:
    rbc_switch = True
else:
    rbc_switch = _rbc_switch_raw.strip().lower() not in ("0", "false", "no", "off", "")

# 血小板开关（环境变量 PLT_SWITCH: 同上，默认 True）
_plt_switch_raw = os.environ.get("PLT_SWITCH")
if _plt_switch_raw is None:
    plt_switch = True
else:
    plt_switch = _plt_switch_raw.strip().lower() not in ("0", "false", "no", "off", "")

# 最大显存（GB）；可用预算 = max_memory - reserved_memory
max_memory = int(os.environ.get("MAX_MEMORY", "11"))

# 预留显存（GB）
reserved_memory = int(os.environ.get("RESERVED_MEMORY", "1"))

# 是否保存热力图，默认关闭
save_heatmap = os.environ.get("SAVE_HEATMAP", "false").strip().lower() not in ("0", "false", "no", "off", "")

# 各端点共用端口：8000(HTTP)、8001(gRPC)、8002(Metrics)、9000(multi_pipeline)
TRITON_HTTP_PORT = int(os.environ.get("TRITON_HTTP_PORT", "8000"))
TRITON_GRPC_PORT = int(os.environ.get("TRITON_GRPC_PORT", "8001"))
TRITON_METRICS_PORT = int(os.environ.get("TRITON_METRICS_PORT", "8002"))
MULTI_PIPELINE_PORT = int(os.environ.get("MULTI_PIPELINE_PORT", "9000"))

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

# 同时驻留内存的平扫图数量。2448×2048 解码约 15MB，默认 16 大约三百 MB。
# 槽位在读 body 之前占用，多出来的上传在连接上排队，不会先把整图收进进程。
TILE_UPLOAD_CONCURRENCY = max(1, int(os.environ.get("TILE_UPLOAD_CONCURRENCY", "16")))

# 单个 worker 内同时进行的选区次数。每次会映射整份 roi.npz。
ROI_SELECTION_CONCURRENCY = max(1, int(os.environ.get("ROI_SELECTION_CONCURRENCY", "1")))

# ========== 运维平台（Eureka + 统一配置）==========
# 注册中心在安装时配置。研发环境默认下面这个地址。
# 算法名确定后不要改：Eureka 的 app 用这里的大写名，vipAddress 和配置路径用它的小写。
yunwei_plat = os.environ.get("YUNWEI_PLAT", "http://192.168.31.80:9000/").strip()
sufa_name = "SUANFA_AI"
# manifest.json 没有 version 时才回退到这里
sufa_version = "v1.0.3"
# 是否显示文档
is_doc = True

# 统一配置环境 / 分支。平台默认 dev、master，生产用环境变量覆盖。
yunwei_profile = os.environ.get("YUNWEI_PROFILE", "dev").strip() or "dev"
yunwei_label = os.environ.get("YUNWEI_LABEL", "master").strip() or "master"
# 拉取配置的应用名，与算法名相同：/{app}/{profile}/{label}
yunwei_config_app = sufa_name
# 后续鉴权。为空则不发送；有值后注册、心跳、查实例、拉配置都带 X-Micro-Token。
yunwei_micro_token = os.environ.get("X_MICRO_TOKEN", "").strip()
yunwei_user = os.environ.get("YUNWEI_USER", "").strip()
yunwei_password = os.environ.get("YUNWEI_PASSWORD", "")
# metadata.zone。不填则写 default。
yunwei_zone = os.environ.get("YUNWEI_ZONE", "").strip()
# 注册用的可达 IP。不填则在 WSL 里取宿主机 Windows 的局域网 IP。
yunwei_ip = os.environ.get("SUFA_IP", "").strip()
# 注册中心上公布的端口。进程仍听 APP_PORT，对外健康检查固定 9703。
yunwei_advertise_port = 9703
# 退出时先标 OUT_OF_SERVICE，等待这么多秒再注销，给网关缓存过期。
yunwei_drain_seconds = max(0, int(os.environ.get("YUNWEI_DRAIN_SECONDS", "60")))
