# model_control.py
"""
Triton 模型动态加载/卸载。需 Triton 以 --model-control-mode=explicit 启动。
按模型组管理，显存最多保留 max_groups 组；其中 DPI147246_BM_PB_pipeline 对应组常驻、不参与 LRU、不会被卸载。
其余组在 (max_groups - 1) 个槽位内按 LRU（最少最近使用）淘汰整组。
每次 ensure_model_loaded 调用都记录时间；淘汰时只从非常驻组中选最久未用者。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Dict, List, Tuple

from config import TRITON_HTTP_URL

logger = logging.getLogger(__name__)

_http_base_url_logged = False

# 模型组：pipeline 名称 -> (组标识, [子模型..., pipeline])
# 子模型先加载，pipeline 最后加载
MODEL_GROUPS: Dict[str, Tuple[str, List[str]]] = {
    "DPI147246_BM_PB_pipeline": (
        "DPI147246_BM_PB",
        [
            "DPI147246_BM_PB_WBC_cell_detection",
            "DPI147246_BM_PB_MEG_cell_detection",
            "DPI147246_BM_PB_constituency_score",
            "DPI147246_BM_PB_pipeline",
            "DPI147246_BM_PB_cell_analysis"
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

# 常驻组（与 MODEL_GROUPS 中 DPI147246_BM_PB_pipeline 的 group_key 一致）：启动预加载，不参与 LRU，永不淘汰
PINNED_GROUP_KEY = "DPI147246_BM_PB"
PINNED_PIPELINE_NAME = "DPI147246_BM_PB_pipeline"

# LRU: group_key -> 最后访问时间戳（常驻组也会更新，仅用于观测，不参与淘汰）
_group_last_used: Dict[str, float] = {}
_max_groups = int(os.environ.get("TRITON_MAX_GROUPS", "3"))
_model_lock = threading.Lock()
LOAD_TIMEOUT = int(os.environ.get("TRITON_LOAD_TIMEOUT", "600"))


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


def ensure_model_loaded(model_name: str, max_models: int = None, max_groups: int = None) -> Tuple[bool, str]:
    """
    确保模型组已加载。model_name 为 pipeline 名称（如 DPI147246_BM_PB_pipeline）。
    显存最多保留 max_groups 组（默认 3）：其中 DPI147246_BM_PB 组常驻、不淘汰；
    其余组最多占 (max_groups - 1) 个槽位，超限时在非常驻组中按 LRU 淘汰。
    每次调用都会更新该组的最后访问时间戳。
    返回 (成功, 错误信息)
    """
    global _group_last_used
    max_groups = max_groups if max_groups is not None else _max_groups
    if max_models is not None:
        max_groups = max_models  # 兼容旧参数

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
        while len(loaded_groups) >= max_groups and loaded_groups:
            evictable = _evictable_group_keys(loaded_groups)
            if not evictable:
                logger.warning(
                    "Model groups at limit (%s) but no evictable group (pinned=%s holds slot); cannot load %s",
                    len(loaded_groups),
                    PINNED_GROUP_KEY,
                    group_key,
                )
                return False, "model group capacity full and pinned group cannot be evicted"
            lru_group = min(evictable, key=lambda g: _group_last_used.get(g, 0))
            lru_info = None
            for _pn, (_gk, _ms) in MODEL_GROUPS.items():
                if _gk == lru_group:
                    lru_info = (_gk, _ms)
                    break
            if lru_info:
                _unload_group(lru_info[0], lru_info[1])
                loaded_groups.discard(lru_group)
                if lru_group in _group_last_used:
                    del _group_last_used[lru_group]
            actually_loaded = set(get_loaded_models())
            loaded_groups = _get_loaded_group_keys(actually_loaded)

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
    """Web 服务启动时预加载常驻组（DPI147246_BM_PB_pipeline），失败打 ERROR 日志。"""
    ok, msg = ensure_model_loaded(PINNED_PIPELINE_NAME, max_groups=_max_groups)
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
    ok, msg = ensure_model_loaded(model_name, max_groups=_max_groups)
    if not ok:
        logger.warning("Model warmup failed for %s: %s", model_name, msg)


if __name__ == "__main__":
    print("Loaded models:", get_loaded_models())
    # print("ensure_model_loaded DPI147246_BM_PB_pipeline:", ensure_model_loaded("DPI147246_BM_PB_pipeline"))
