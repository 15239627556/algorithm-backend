"""DPI147246 客户端编排的 CPU 主机预设。

本仓库默认使用 ``10900x`` 预设；Threadripper 3970X 可设环境变量::

    DPI147246_CPU_PROFILE=3970x

或在启动参数中指定 ``--cpu-profile 3970x``。
"""
from __future__ import annotations

import os
from typing import Any

PROFILE_10900X = "10900x"
PROFILE_3970X = "3970x"

_PROFILES: dict[str, dict[str, Any]] = {
    PROFILE_10900X: {
        "timeout_ms": 50,
        "num_slots": 2,
        "preprocess_workers": 8,
        "decode_workers": 8,
        "enable_ca": False,
        "batch_size": 4,
        # 多 slot 下线程并行 NMS 易 GIL 互抢，保持全局串行
        "heavy_post_mode": "serial",
        "heavy_post_workers": 2,
    },
    PROFILE_3970X: {
        # 合批：尽量凑满 4 张，避免 2s 超时导致长期 batch=1
        "timeout_ms": 50,
        # CA 关闭后腾出显存，可多 1 个 slot 叠 batch（仍受 8GB 卡约束，可按需改回 2）
        "num_slots": 3,
        # 32 核：提高 JPEG 解码 / OpenCV 预处理并行（受内存带宽限制，不宜过大）
        "preprocess_workers": 16,
        "decode_workers": 16,
        # 用户要求：CA 先不开；返回字段仍为 0
        "enable_ca": False,
        "batch_size": 4,
        # 跨 slot 用进程池并行 WBC/MEG 后处理（绕过 GIL，与 num_slots 对齐）
        "heavy_post_mode": "process",
        "heavy_post_workers": 3,
    },
}

_ALIASES: dict[str, str] = {
    "default": PROFILE_10900X,
    "3970": PROFILE_3970X,
    "tr": PROFILE_3970X,
    "threadripper": PROFILE_3970X,
    "legacy": PROFILE_10900X,
    "10900": PROFILE_10900X,
    "old": PROFILE_10900X,
}


def resolve_cpu_profile(name: str | None = None) -> str:
    raw = (name or os.environ.get("DPI147246_CPU_PROFILE", PROFILE_10900X)).strip().lower()
    if raw in _PROFILES:
        return raw
    return _ALIASES.get(raw, PROFILE_10900X)


def get_profile_kwargs(profile: str | None = None) -> dict[str, Any]:
    return dict(_PROFILES[resolve_cpu_profile(profile)])


# 模块加载时解析一次，供各模块默认参数引用
ACTIVE_PROFILE = resolve_cpu_profile()
DEFAULT_KWARGS: dict[str, Any] = get_profile_kwargs(ACTIVE_PROFILE)
