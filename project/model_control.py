# model_control.py
"""
Triton 模型动态加载/卸载。需 Triton 以 --model-control-mode=explicit 启动。
按模型组管理：DPI147246_BM_PB（144750 pipeline）常驻、预加载、不参与 LRU、不会被卸载。
其余组在「预估显存占用 + 常驻组」超过 GPU 预算（见 config.TRITON_GPU_VRAM_GB − 预留）时，
按 LRU（最久未访问）淘汰整组，直至能装入目标组（槽位数不再固定为 3）。
每次 ensure_model_loaded 调用都更新该组最后访问时间；淘汰仅从非常驻组中选择。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

from config import (
    TRITON_GPU_VRAM_GB,
    TRITON_HTTP_URL,
    TRITON_PINNED_PIPELINE_NAME,
    TRITON_VRAM_RESERVE_GB,
)

logger = logging.getLogger(__name__)

_http_base_url_logged = False

# 常驻 pipeline 名称见 config.TRITON_PINNED_PIPELINE_NAME；MODEL_GROUPS 中键须与 Triton 仓库内名称一致
PINNED_PIPELINE_NAME = TRITON_PINNED_PIPELINE_NAME

# 模型组：pipeline 名称 -> (组标识, [子模型..., pipeline])
# 子模型先加载，pipeline 最后加载
MODEL_GROUPS: Dict[str, Tuple[str, List[str]]] = {
    PINNED_PIPELINE_NAME: (
        "DPI147246_BM_PB",
        [
            "DPI147246_BM_PB_WBC_cell_detection",
            "DPI147246_BM_PB_MEG_cell_detection",
            "DPI147246_BM_PB_constituency_score",
            PINNED_PIPELINE_NAME,
            "DPI147246_BM_PB_cell_analysis",
        ],
    ),
    "DPI357378_BM_MEG_pipeline": (
        "DPI357378_BM_MEG",
        [
            "DPI357378_BM_MEG_cell_detection",
            "DPI357378_BM_MEG_cell_classifier",
            "DPI357378_BM_MEG_pipeline",
        ],
    ),
    "DPI714756_CF_WBC_pipeline": (
        "DPI714756_CF",
        [
            "DPI714756_CF_WBC_cell_detector",
            "DPI714756_CF_WBC_cell_classifier",
            "DPI714756_CF_WBC_pipeline",
        ],
    ),
    "DPI714756_BM_PB_pipeline": (
        "DPI714756_BM_PB",
        [
            "DPI714756_BM_PB_WBC_detector",
            "DPI714756_BM_PB_WBC_classifier",
            "DPI714756_BM_PB_RED_cell_detection",
            "DPI714756_BM_PB_pipeline",
        ],
    ),
    "Image_enhance_pipeline": (
        "Image_enhance",
        [
            "Image_enhance",
            "Image_enhance_pipeline",
        ],
    ),
}

# 每组预估显存占用（GB），与 triton_client 注释一致；用于加载前测算是否需 LRU 淘汰
GROUP_VRAM_GB: Dict[str, float] = {
    "DPI147246_BM_PB": 6.0,
    "DPI357378_BM_MEG": 3.5,
    "DPI714756_CF": 7.5,
    "DPI714756_BM_PB": 3.0,
    "Image_enhance": 3.0,
}

# 常驻组 group_key（与 MODEL_GROUPS 中常驻项的元组第一项一致）；换 pipeline 时若组名变化需同步改 GROUP_VRAM_GB 键
PINNED_GROUP_KEY = "DPI147246_BM_PB"

# LRU: group_key -> 最后访问时间戳（常驻组也会更新，仅用于观测，不参与淘汰）
_group_last_used: Dict[str, float] = {}
_model_lock = threading.Lock()
LOAD_TIMEOUT = int(os.environ.get("TRITON_LOAD_TIMEOUT", "600"))


def _effective_vram_budget_gb() -> float:
    """可用显存上限（总显存 − 预留，防碎片/峰值）"""
    return max(0.5, TRITON_GPU_VRAM_GB - TRITON_VRAM_RESERVE_GB)


def _group_vram_gb(group_key: str) -> float:
    return float(GROUP_VRAM_GB.get(group_key, 2.0))


def _estimated_vram_for_groups(loaded_groups: set) -> float:
    return sum(_group_vram_gb(g) for g in loaded_groups)


def _get_http_base_url() -> str:
    global _http_base_url_logged
    url = os.environ.get("TRITON_HTTP_URL")
    base = url.rstrip("/") if url else TRITON_HTTP_URL
    if not _http_base_url_logged:
        _http_base_url_logged = True
        logger.info("Triton HTTP API base: %s", base)
    return base


def _http_post(url: str, body: bytes = None, timeout: int = LOAD_TIMEOUT) -> Tuple[bool, str]:
    try:
        req = urllib.request.Request(url, data=body or b"{}", method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, resp.read().decode() if resp.length else ""
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode()
        except Exception:
            err_body = str(e)
        return False, f"HTTP {e.code}: {err_body}"
    except Exception as e:
        return False, str(e)


def get_loaded_models() -> List[str]:
    """获取当前已加载的模型列表（Triton repository index，state=READY）"""
    base = _get_http_base_url()
    url = f"{base}/v2/repository/index"
    body = json.dumps({"ready": True}).encode()
    ok, msg = _http_post(url, body, timeout=30)
    if not ok:
        logger.warning("get_loaded_models failed: %s", msg)
        return []
    try:
        data = json.loads(msg)
        return [m["name"] for m in data if m.get("state") == "READY"]
    except Exception as e:
        logger.warning("parse get_loaded_models response: %s", e)
        return []


def load_model(model_name: str, timeout: int = LOAD_TIMEOUT) -> Tuple[bool, str]:
    """加载单个模型"""
    base = _get_http_base_url()
    url = f"{base}/v2/repository/models/{model_name}/load"
    ok, msg = _http_post(url, b"{}", timeout=timeout)
    if ok:
        logger.info("Model %s loaded.", model_name)
    else:
        logger.warning("Load model %s failed: %s", model_name, msg)
    return ok, msg


def unload_model(model_name: str, timeout: int = 60) -> Tuple[bool, str]:
    """卸载单个模型"""
    base = _get_http_base_url()
    url = f"{base}/v2/repository/models/{model_name}/unload"
    ok, msg = _http_post(url, b"{}", timeout=timeout)
    if ok:
        logger.info("Model %s unloaded.", model_name)
    else:
        logger.warning("Unload model %s failed: %s", model_name, msg)
    return ok, msg


def _get_group_for_pipeline(pipeline_name: str) -> Tuple[str, List[str]] | None:
    """根据 pipeline 名称获取所属组。若不在组内则返回 None。"""
    return MODEL_GROUPS.get(pipeline_name)


def _unload_group(group_key: str, models: List[str]) -> None:
    """卸载整组模型（pipeline 先卸，子模型后卸）"""
    for m in reversed(models):
        unload_model(m)


def _get_loaded_group_keys(actually_loaded: set) -> set:
    """根据已加载的模型列表，反推出当前加载的组 key 集合"""
    loaded_groups = set()
    for _pn, (gk, ms) in MODEL_GROUPS.items():
        if all(m in actually_loaded for m in ms):
            loaded_groups.add(gk)
    return loaded_groups


def _evictable_group_keys(loaded_groups: set) -> set:
    """可作为 LRU 淘汰候选的组。若常驻组已加载，则将其排除；否则全部可淘汰。"""
    if PINNED_GROUP_KEY in loaded_groups:
        return loaded_groups - {PINNED_GROUP_KEY}
    return set(loaded_groups)


def ensure_model_loaded(
    model_name: str,
    max_models: Optional[int] = None,
    max_groups: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    确保模型组已加载。model_name 为 pipeline 名称（见 config.TRITON_PINNED_PIPELINE_NAME 等）。
    容量规则：常驻组（PINNED_GROUP_KEY）永不卸载；其余组在
    「当前已加载组预估显存之和 + 待加载组预估显存」超过 TRITON_GPU_VRAM_GB − 预留时，
    在非常驻组中按 LRU 整组卸载，直至预算足够（不再使用固定「最多 3 组」）。
    max_groups / max_models 若传入则额外限制「已加载组数量」上限（兼容旧调用），默认不限制仅按显存。
    每次调用都会更新该组的最后访问时间戳。
    返回 (成功, 错误信息)
    """
    global _group_last_used
    use_count_cap = max_groups is not None or max_models is not None
    count_cap = max_groups if max_groups is not None else max_models

    group_info = _get_group_for_pipeline(model_name)
    if not group_info:
        # 不在预定义组内，退化为单模型加载
        with _model_lock:
            actually_loaded = set(get_loaded_models())
            if model_name in actually_loaded:
                return True, ""
            ok, msg = load_model(model_name)
            return ok, msg

    group_key, models = group_info
    now = time.time()
    with _model_lock:
        actually_loaded = set(get_loaded_models())

        if all(m in actually_loaded for m in models):
            _group_last_used[group_key] = now
            return True, ""

        loaded_groups = _get_loaded_group_keys(actually_loaded)
        budget = _effective_vram_budget_gb()
        new_gb = _group_vram_gb(group_key)

        def _over_vram() -> bool:
            return _estimated_vram_for_groups(loaded_groups) + new_gb > budget + 1e-6

        def _over_count() -> bool:
            if not use_count_cap or count_cap is None:
                return False
            return len(loaded_groups) >= count_cap and loaded_groups

        while (_over_vram() or _over_count()) and loaded_groups:
            evictable = _evictable_group_keys(loaded_groups)
            if not evictable:
                need = _estimated_vram_for_groups(loaded_groups) + new_gb
                logger.warning(
                    "Cannot load group %s: need ~%.1fGB, budget %.1fGB, no evictable non-pinned group",
                    group_key,
                    need,
                    budget,
                )
                return (
                    False,
                    f"VRAM insufficient: need ~{need:.1f}GB, budget {budget:.1f}GB (pinned group not evictable)",
                )
            lru_group = min(evictable, key=lambda g: _group_last_used.get(g, 0))
            lru_info = None
            for _pn, (_gk, _ms) in MODEL_GROUPS.items():
                if _gk == lru_group:
                    lru_info = (_gk, _ms)
                    break
            if lru_info:
                logger.info(
                    "Evicting LRU model group %s (~%.1fGB) to load %s (~%.1fGB); budget=%.1fGB",
                    lru_group,
                    _group_vram_gb(lru_group),
                    group_key,
                    new_gb,
                    budget,
                )
                _unload_group(lru_info[0], lru_info[1])
                loaded_groups.discard(lru_group)
                if lru_group in _group_last_used:
                    del _group_last_used[lru_group]
            actually_loaded = set(get_loaded_models())
            loaded_groups = _get_loaded_group_keys(actually_loaded)

        if _over_vram() or _over_count():
            need = _estimated_vram_for_groups(loaded_groups) + new_gb
            return False, f"model group capacity full: need ~{need:.1f}GB, budget {budget:.1f}GB"

        for m in models:
            if m in actually_loaded:
                continue
            ok, msg = load_model(m)
            if not ok:
                return False, msg
            actually_loaded.add(m)

        _group_last_used[group_key] = now
        return True, ""


def warmup_pinned_models_at_startup() -> None:
    """Web 服务启动时预加载常驻 pipeline（config.TRITON_PINNED_PIPELINE_NAME），失败打 ERROR 日志。"""
    logger.info(
        "Triton VRAM policy: effective budget %.1fGB (GPU %.1fGB − reserve %.1fGB); pinned=%s",
        _effective_vram_budget_gb(),
        TRITON_GPU_VRAM_GB,
        TRITON_VRAM_RESERVE_GB,
        PINNED_PIPELINE_NAME,
    )
    ok, msg = ensure_model_loaded(PINNED_PIPELINE_NAME)
    if not ok:
        logger.error(
            "Pinned model group %s failed to load at startup: %s",
            PINNED_PIPELINE_NAME,
            msg,
        )
    else:
        logger.info("Pinned model group %s loaded at startup.", PINNED_PIPELINE_NAME)


def warmup_model(model_name: str) -> None:
    """模型预热：创建任务或单张识别前调用，确保模型组已加载。失败时记录日志，不抛异常。"""
    ok, msg = ensure_model_loaded(model_name)
    if not ok:
        logger.warning("Model warmup failed for %s: %s", model_name, msg)


if __name__ == "__main__":
    print("Loaded models:", get_loaded_models())
    # print("ensure_model_loaded pinned:", ensure_model_loaded(PINNED_PIPELINE_NAME))
