# model_control.py
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Dict, FrozenSet, List, Optional, Tuple

from config import (
    TRITON_ENDPOINTS,
    TRITON_HTTP_URL,
    get_triton_endpoint,
    next_triton_endpoint,
)

logger = logging.getLogger(__name__)

_logged_http_bases: set[str] = set()

# 模型组：pipeline 名称 -> (组标识, [子模型..., pipeline])
# 子模型先加载，pipeline 最后加载
MODEL_GROUPS: Dict[str, Tuple[str, List[str]]] = {
    "DPI147246_BM_PB_pipeline": (
        "DPI147246_BM_PB",
        [
            "DPI147246_BM_PB_WBC_cell_detection",
            "DPI147246_BM_PB_MEG_cell_detection",
            "DPI147246_BM_PB_constituency_score",
            "DPI147246_BM_PB_cell_analysis",
        ],
    ),
    "DPI357378_BM_MEG_pipeline": (
        "DPI357378_BM_MEG",
        [
            "DPI357378_BM_MEG_cell_detection",
            "DPI357378_BM_MEG_cell_classifier",
        ],
    ),
    # "DPI714756_CF_WBC_pipeline": (
    #     "DPI714756_CF",
    #     [
            # "DPI714756_CF_WBC_cell_detector",
            # "DPI714756_CF_WBC_cell_classifier",
    #     ],
    # ),
    "DPI714756_BM_PB_pipeline": (
        "DPI714756_BM_PB",
        [
            "DPI714756_BM_PB_WBC_detector",
            "DPI714756_BM_PB_WBC_classifier",
            "DPI714756_BM_PB_RED_cell_detection",
            "DPI714756_BM_PB_RED_cell_classifier",
            "DPI714756_BM_PB_PLAT_detection"
        ],
    ),
    "Image_enhance_pipeline": (
        "Image_enhance",
        [
            "Image_enhance",
        ],
    ),
}

# 常驻组（加载后不参与 LRU 淘汰）；亦为启动预热列表（当前预热已关闭，见 warmup_pinned_models_at_startup）
STARTUP_WARMUP_PIPELINES: Tuple[str, ...] = (
    "DPI357378_BM_MEG_pipeline",
)


def _pinned_group_keys() -> FrozenSet[str]:
    keys: set[str] = set()
    for pname in STARTUP_WARMUP_PIPELINES:
        entry = MODEL_GROUPS.get(pname)
        if entry:
            keys.add(entry[0])
    return frozenset(keys)


PINNED_GROUP_KEYS: FrozenSet[str] = _pinned_group_keys()

# 每组预估显存占用（GB），与 backend.tools.triton_client 注释一致；用于加载前测算是否需 LRU 淘汰
GROUP_VRAM_GB: Dict[str, float] = {
    "DPI147246_BM_PB": 3.3,
    "DPI357378_BM_MEG": 0.2,
    # "DPI714756_CF": 7.5,
    "DPI714756_BM_PB": 3.1,
    "Image_enhance": 1.6,
}

# 按 GPU 隔离的 LRU：gpu_id -> {group_key -> 最后访问时间戳}
_group_last_used_by_gpu: Dict[int, Dict[str, float]] = {}
# 本进程已确认 READY 的组/单模型（启动预热或成功 load 后写入；unload 时清除）
# 热路径只查此缓存，避免每次请求在全局锁内打 Triton HTTP
_ready_groups_by_gpu: Dict[int, set] = {}
_ready_models_by_gpu: Dict[int, set] = {}
_model_lock = threading.Lock()
LOAD_TIMEOUT = int(os.environ.get("TRITON_LOAD_TIMEOUT", "600"))
TRITON_GPU_VRAM_GB = float(os.environ.get("TRITON_GPU_VRAM_GB", "11"))
TRITON_VRAM_RESERVE_GB = float(os.environ.get("TRITON_VRAM_RESERVE_GB", "3"))
# 每张 GPU 显存中最多同时保留的模型组数（含常驻组；超限时 LRU 淘汰最久未用非常驻组）
MAX_LOADED_MODEL_GROUPS = int(os.environ.get("TRITON_MAX_LOADED_MODEL_GROUPS", "2"))
# 互斥组：同 GPU 上不可共存，加载一侧时强制卸载另一侧
MUTEX_GROUP_KEYS: Tuple[FrozenSet[str], ...] = (
    frozenset({"DPI147246_BM_PB", "DPI714756_BM_PB"}),
)


def _effective_vram_budget_gb() -> float:
    """可用显存上限（总显存 − 预留，防碎片/峰值）"""
    return max(0.5, TRITON_GPU_VRAM_GB - TRITON_VRAM_RESERVE_GB)


def _group_vram_gb(group_key: str) -> float:
    return float(GROUP_VRAM_GB.get(group_key, 2.0))


def _estimated_vram_for_groups(loaded_groups: set) -> float:
    return sum(_group_vram_gb(g) for g in loaded_groups)


def _normalize_gpu_id(gpu_id: Optional[int]) -> int:
    if gpu_id is None:
        return 0
    gid = int(gpu_id)
    if gid < 0 or gid >= len(TRITON_ENDPOINTS):
        return 0
    return gid


def _warmup_gpu_targets(gpu_id: Optional[int] = None, *, all_gpus: bool = False) -> List[int]:
    """显式 gpu_id 时仅用该卡；all_gpus 时预热 TRITON_ENDPOINTS 全部端点；否则轮询单卡（与 infer 一致）。"""
    if gpu_id is not None:
        return [_normalize_gpu_id(gpu_id)]
    n = len(TRITON_ENDPOINTS)
    if all_gpus and n > 0:
        return list(range(n))
    gid, _ = next_triton_endpoint()
    return [gid]


def _get_http_base_url(gpu_id: Optional[int] = None) -> str:
    """取指定 GPU 的 Triton HTTP 基址。显式 gpu_id 时不使用单点 TRITON_HTTP_URL 覆盖。"""
    if gpu_id is not None:
        base = (get_triton_endpoint(gpu_id).get("http_url") or TRITON_HTTP_URL).rstrip("/")
    else:
        url = os.environ.get("TRITON_HTTP_URL")
        if url:
            base = url.rstrip("/")
        else:
            base = (get_triton_endpoint().get("http_url") or TRITON_HTTP_URL).rstrip("/")
    if base not in _logged_http_bases:
        _logged_http_bases.add(base)
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


def get_loaded_models(gpu_id: Optional[int] = None) -> List[str]:
    """获取指定 GPU 上已加载的模型列表（Triton repository index，state=READY）"""
    base = _get_http_base_url(gpu_id)
    url = f"{base}/v2/repository/index"
    body = json.dumps({"ready": True}).encode()
    ok, msg = _http_post(url, body, timeout=30)
    if not ok:
        logger.warning("get_loaded_models failed (gpu=%s): %s", gpu_id, msg)
        return []
    try:
        data = json.loads(msg)
        return [m["name"] for m in data if m.get("state") == "READY"]
    except Exception as e:
        logger.warning("parse get_loaded_models response: %s", e)
        return []


def load_model(
    model_name: str,
    timeout: int = LOAD_TIMEOUT,
    gpu_id: Optional[int] = None,
) -> Tuple[bool, str]:
    """在指定 GPU 上加载单个模型"""
    base = _get_http_base_url(gpu_id)
    url = f"{base}/v2/repository/models/{model_name}/load"
    ok, msg = _http_post(url, b"{}", timeout=timeout)
    if ok:
        logger.info("Model %s loaded on gpu=%s (%s).", model_name, gpu_id, base)
    else:
        logger.warning("Load model %s failed on gpu=%s: %s", model_name, gpu_id, msg)
    return ok, msg


def unload_model(
    model_name: str,
    timeout: int = 60,
    gpu_id: Optional[int] = None,
) -> Tuple[bool, str]:
    """在指定 GPU 上卸载单个模型"""
    base = _get_http_base_url(gpu_id)
    url = f"{base}/v2/repository/models/{model_name}/unload"
    ok, msg = _http_post(url, b"{}", timeout=timeout)
    if ok:
        logger.info("Model %s unloaded on gpu=%s (%s).", model_name, gpu_id, base)
    else:
        logger.warning("Unload model %s failed on gpu=%s: %s", model_name, gpu_id, msg)
    return ok, msg


def _get_group_for_pipeline(pipeline_name: str) -> Tuple[str, List[str]] | None:
    """根据 pipeline 名称获取所属组。若不在组内则返回 None。"""
    return MODEL_GROUPS.get(pipeline_name)


def _unload_group(group_key: str, models: List[str], gpu_id: Optional[int] = None) -> None:
    """卸载整组模型（pipeline 先卸，子模型后卸）"""
    for m in reversed(models):
        unload_model(m, gpu_id=gpu_id)


def _get_loaded_group_keys(actually_loaded: set) -> set:
    """根据已加载的模型列表，反推出当前加载的组 key 集合"""
    loaded_groups = set()
    for _pn, (gk, ms) in MODEL_GROUPS.items():
        if all(m in actually_loaded for m in ms):
            loaded_groups.add(gk)
    return loaded_groups


def _evictable_group_keys(loaded_groups: set) -> set:
    """可作为 LRU 淘汰候选的组。常驻组集合 PINNED_GROUP_KEYS 中的已加载项排除；无常驻集合时全部可淘汰。"""
    if PINNED_GROUP_KEYS:
        return loaded_groups - PINNED_GROUP_KEYS
    return set(loaded_groups)


def _mutex_groups_to_evict(group_key: str, loaded_groups: set) -> set:
    """加载 group_key 时需强制卸载的互斥组（已在 loaded_groups 中）。"""
    evict: set = set()
    for pair in MUTEX_GROUP_KEYS:
        if group_key in pair:
            evict |= pair & loaded_groups - {group_key}
    return evict


def _models_for_group_key(group_key: str) -> List[str] | None:
    for _pn, (gk, ms) in MODEL_GROUPS.items():
        if gk == group_key:
            return ms
    return None


def _evict_group_on_gpu(
    group_key: str,
    gid: int,
    *,
    ready_groups: set,
    last_used: Dict[str, float],
    loaded_groups: set,
    reason: str,
) -> None:
    models = _models_for_group_key(group_key)
    if not models:
        return
    logger.info(
        "Evicting model group %s (~%.1fGB) on gpu=%s; reason=%s",
        group_key,
        _group_vram_gb(group_key),
        gid,
        reason,
    )
    _unload_group(group_key, models, gpu_id=gid)
    loaded_groups.discard(group_key)
    ready_groups.discard(group_key)
    last_used.pop(group_key, None)


def ensure_model_loaded(
    model_name: str,
    max_models: Optional[int] = None,
    max_groups: Optional[int] = None,
    gpu_id: Optional[int] = None,
) -> Tuple[bool, str]:
    gid = _normalize_gpu_id(gpu_id)
    if max_groups is None and max_models is None:
        max_groups = MAX_LOADED_MODEL_GROUPS
    use_count_cap = max_groups is not None or max_models is not None
    count_cap = max_groups if max_groups is not None else max_models

    group_info = _get_group_for_pipeline(model_name)
    if not group_info:
        # 不在预定义组内，退化为单模型加载
        with _model_lock:
            if model_name in _ready_models_by_gpu.get(gid, ()):
                return True, ""
            actually_loaded = set(get_loaded_models(gpu_id=gid))
            if model_name in actually_loaded:
                _ready_models_by_gpu.setdefault(gid, set()).add(model_name)
                return True, ""
            ok, msg = load_model(model_name, gpu_id=gid)
            if ok:
                _ready_models_by_gpu.setdefault(gid, set()).add(model_name)
            return ok, msg

    group_key, models = group_info
    now = time.time()

    # 热路径：启动预热/先前 load 已确认 READY → 不打 Triton HTTP、几乎无串行
    with _model_lock:
        if group_key in _ready_groups_by_gpu.get(gid, ()):
            last_used = _group_last_used_by_gpu.setdefault(gid, {})
            last_used[group_key] = now
            return True, ""

    with _model_lock:
        last_used = _group_last_used_by_gpu.setdefault(gid, {})
        ready_groups = _ready_groups_by_gpu.setdefault(gid, set())
        if group_key in ready_groups:
            last_used[group_key] = now
            return True, ""

        actually_loaded = set(get_loaded_models(gpu_id=gid))

        if all(m in actually_loaded for m in models):
            last_used[group_key] = now
            ready_groups.add(group_key)
            return True, ""

        loaded_groups = _get_loaded_group_keys(actually_loaded)
        budget = _effective_vram_budget_gb()
        new_gb = _group_vram_gb(group_key)

        for mutex_gk in sorted(_mutex_groups_to_evict(group_key, loaded_groups)):
            _evict_group_on_gpu(
                mutex_gk,
                gid,
                ready_groups=ready_groups,
                last_used=last_used,
                loaded_groups=loaded_groups,
                reason=f"mutex with {group_key}",
            )
        actually_loaded = set(get_loaded_models(gpu_id=gid))
        loaded_groups = _get_loaded_group_keys(actually_loaded)

        def _over_vram() -> bool:
            return _estimated_vram_for_groups(loaded_groups) + new_gb > budget + 1e-6

        def _over_count() -> bool:
            if not use_count_cap or count_cap is None:
                return False
            if group_key in loaded_groups:
                return False
            return len(loaded_groups) >= int(count_cap) and bool(loaded_groups)

        while (_over_vram() or _over_count()) and loaded_groups:
            evictable = _evictable_group_keys(loaded_groups)
            if not evictable:
                logger.warning(
                    "Cannot load group %s on gpu=%s: loaded=%d cap=%d, "
                    "need ~%.1fGB budget %.1fGB, no evictable non-pinned group",
                    group_key,
                    gid,
                    len(loaded_groups),
                    count_cap,
                    _estimated_vram_for_groups(loaded_groups) + new_gb,
                    budget,
                )
                return (
                    False,
                    f"Model group capacity full on gpu={gid}: "
                    f"loaded {len(loaded_groups)} groups, cap {count_cap} "
                    f"(pinned group not evictable)",
                )
            lru_group = min(evictable, key=lambda g: last_used.get(g, 0))
            lru_info = None
            for _pn, (_gk, _ms) in MODEL_GROUPS.items():
                if _gk == lru_group:
                    lru_info = (_gk, _ms)
                    break
            if lru_info:
                logger.info(
                    "Evicting LRU model group %s (~%.1fGB) on gpu=%s to load %s (~%.1fGB); "
                    "loaded=%d cap=%d budget=%.1fGB",
                    lru_group,
                    _group_vram_gb(lru_group),
                    gid,
                    group_key,
                    new_gb,
                    len(loaded_groups),
                    count_cap,
                    budget,
                )
                _evict_group_on_gpu(
                    lru_group,
                    gid,
                    ready_groups=ready_groups,
                    last_used=last_used,
                    loaded_groups=loaded_groups,
                    reason=f"LRU to load {group_key}",
                )
            actually_loaded = set(get_loaded_models(gpu_id=gid))
            loaded_groups = _get_loaded_group_keys(actually_loaded)

        if _over_vram() or _over_count():
            return (
                False,
                f"Model group capacity full on gpu={gid}: "
                f"loaded {len(loaded_groups)} groups, cap {count_cap}, "
                f"need ~{_estimated_vram_for_groups(loaded_groups) + new_gb:.1f}GB, budget {budget:.1f}GB",
            )

        for m in models:
            if m in actually_loaded:
                continue
            ok, msg = load_model(m, gpu_id=gid)
            if not ok:
                return False, msg
            actually_loaded.add(m)

        last_used[group_key] = now
        ready_groups.add(group_key)
        return True, ""


def warmup_pinned_models_at_startup() -> None:
    """预热常驻 pipeline；轮询时双容器均预热，定点时仅 TRITON_GPU_ID。（暂时关闭）"""
    logger.info("Startup warmup disabled; models load on demand via warmup_model/LRU.")
    return
    # loaded_any = False
    # for gpu_id in _warmup_gpu_targets():
    #     ep = get_triton_endpoint(gpu_id)
    #     logger.info("Startup warmup on %s (gpu_id=%s)", ep.get("name"), gpu_id)
    #     for pname in STARTUP_WARMUP_PIPELINES:
    #         if pname not in MODEL_GROUPS:
    #             logger.warning("Startup warmup: pipeline %r 不在 MODEL_GROUPS 中，已跳过。", pname)
    #             continue
    #         ok, msg = ensure_model_loaded(pname, gpu_id=gpu_id)
    #         if not ok:
    #             logger.error(
    #                 "Startup warmup: pipeline %s failed on gpu=%s: %s",
    #                 pname,
    #                 gpu_id,
    #                 msg,
    #             )
    #         else:
    #             loaded_any = True
    #             logger.info("Startup warmup: pipeline %s loaded on gpu=%s.", pname, gpu_id)
    # if not loaded_any:
    #     logger.warning(
    #         "启动预加载未成功加载任何 pipeline（请检查 MODEL_GROUPS 与 Triton）；请求仍将走动态加载/LRU。"
    #     )


def warmup_model(
    model_name: str,
    gpu_id: Optional[int] = None,
    *,
    all_gpus: bool = False,
) -> None:
    """
    预热模型。
    - all_gpus=True：预热 config.TRITON_ENDPOINTS 全部端点（平扫 create_task 串行场景）。
    - 否则：未指定 gpu_id 时经 next_triton_endpoint 轮询单卡（与 infer 一致）。
    """
    targets = _warmup_gpu_targets(gpu_id, all_gpus=all_gpus)
    for gid in targets:
        ok, msg = ensure_model_loaded(model_name, gpu_id=gid)
        if not ok:
            logger.warning("Model warmup failed for %s on gpu=%s: %s", model_name, gid, msg)
        else:
            ep = get_triton_endpoint(gid)
            logger.info(
                "Model warmup ok: %s on gpu=%s (%s)",
                model_name,
                gid,
                ep.get("name"),
            )
