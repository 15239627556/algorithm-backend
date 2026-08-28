# model_control.py
from __future__ import annotations

import json
import logging
import os
import threading
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from config import (
    TRITON_ENDPOINTS,
    TRITON_HTTP_URL,
    camera,
    get_triton_endpoint,
    max_memory,
    next_triton_endpoint,
    reserved_memory,
)
from backend.tools.MESSAGE_DICT import (
    DEFAULT_MODEL_VRAM_GB,
    MODEL_TABLE,
    _OUTPUT_KIND,
)
from backend.tools.combo_validator import (
    LEGACY_DPI_MAP,
    _parse_cell_types,
    normalize_smear_type,
)

logger = logging.getLogger(__name__)

_logged_http_bases: set[str] = set()
_model_lock = threading.Lock()

KIND_DETECTION = "detection"
KIND_CLASSIFICATION = "classification"
KIND_SCORE = "score"

_IMAGE_ENHANCE_MODELS: Tuple[str, ...] = ("Image_enhance",)

LOAD_TIMEOUT = int(os.environ.get("TRITON_LOAD_TIMEOUT", "600"))

_FLIR_WBC_CLASSIFIER = "DPIALL_FLIR_BM_PB_WBC_classifier"
_DEFAULT_WBC_CLASSIFIER = "DPIALL_BM_PB_WBC_classifier"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    actual_dpi: int
    smear_types: frozenset[str]
    targets: frozenset[str]
    output: str
    dpi_min: int | None = None
    dpi_max: int | None = None
    vram_gb: float = DEFAULT_MODEL_VRAM_GB

    def dpi_matches(self, dpi: int) -> bool:
        if self.dpi_min is None or self.dpi_max is None:
            return True
        return self.dpi_min <= dpi <= self.dpi_max

    def smear_matches(self, smear_type: str) -> bool:
        return smear_type in self.smear_types

    def target_matches(self, types: set[str]) -> bool:
        if not self.targets:
            return True
        if not types:
            return False
        return bool(self.targets & types)


@dataclass
class ResolvedModels:
    detection: List[ModelSpec] = field(default_factory=list)
    classification: List[ModelSpec] = field(default_factory=list)
    score: List[ModelSpec] = field(default_factory=list)
    warning: str | None = None
    dpi_unsuitable: bool = False

    @property
    def names(self) -> List[str]:
        ordered: dict[str, None] = {}
        for spec in (*self.detection, *self.classification, *self.score):
            ordered.setdefault(spec.name, None)
        return list(ordered.keys())

    @property
    def specs(self) -> List[ModelSpec]:
        ordered: dict[str, ModelSpec] = {}
        for spec in (*self.detection, *self.classification, *self.score):
            ordered.setdefault(spec.name, spec)
        return list(ordered.values())

    def has_classification(self) -> bool:
        return bool(self.classification)

    def has_classifier_for(self, target: str) -> bool:
        key = (target or "").strip().upper()
        return any(key in spec.targets for spec in self.classification)


def _row_to_spec(row: dict) -> ModelSpec:
    kind = _OUTPUT_KIND.get(str(row.get("output") or "").strip().lower(), KIND_DETECTION)
    smear_raw = str(row.get("smear_types") or "")
    smear_types = frozenset(s.strip().upper() for s in smear_raw.split("/") if s.strip())
    target_raw = row.get("targets")
    if target_raw in (None, "", "不限制"):
        targets: frozenset[str] = frozenset()
    else:
        targets = frozenset(t.strip().upper() for t in str(target_raw).split("/") if t.strip())
    dr = row.get("dpi_range")
    dpi_min = dpi_max = None
    if dr and dr != "不限制":
        dpi_min, dpi_max = int(dr[0]), int(dr[1])
    vram = float(row.get("vram_gb") if row.get("vram_gb") is not None else DEFAULT_MODEL_VRAM_GB)
    return ModelSpec(
        name=str(row["name"]),
        kind=kind,
        actual_dpi=int(row["actual_dpi"]),
        smear_types=smear_types,
        targets=targets,
        output=str(row.get("output") or ""),
        dpi_min=dpi_min,
        dpi_max=dpi_max,
        vram_gb=vram,
    )


def _apply_camera_override(spec: ModelSpec) -> ModelSpec:
    if camera == "flir" and spec.name == _DEFAULT_WBC_CLASSIFIER:
        return ModelSpec(
            name=_FLIR_WBC_CLASSIFIER,
            kind=spec.kind,
            actual_dpi=spec.actual_dpi,
            smear_types=spec.smear_types,
            targets=spec.targets,
            output=spec.output,
            dpi_min=spec.dpi_min,
            dpi_max=spec.dpi_max,
            vram_gb=spec.vram_gb,
        )
    return spec


def get_model_catalog() -> Tuple[ModelSpec, ...]:
    """从 MESSAGE_DICT.MODEL_TABLE 构建目录，并应用 camera 替换。"""
    return tuple(_apply_camera_override(_row_to_spec(row)) for row in MODEL_TABLE)


def _specs_by_name() -> dict[str, ModelSpec]:
    return {spec.name: spec for spec in get_model_catalog()}


def _effective_dpi(dpi: int) -> int:
    return LEGACY_DPI_MAP.get(int(dpi), int(dpi))


def resolve_models(
    dpi: int,
    smear_type: str,
    target_cell_types: str,
) -> ResolvedModels:
    """
    三步解析需加载的模型：
    1. 按 DPI 区间 + 玻片类型 + 目标类型匹配定位模型
    2. 按玻片类型 + 目标类型匹配分类模型（DPI 不限制，须与已命中定位模型的实际 DPI 对齐）
    3. 按 DPI 区间 + 玻片类型匹配评分模型（目标不限制）
    DPI 不在任何适用区间内时 dpi_unsuitable=True，不回退到最近模型。
    """
    catalog = get_model_catalog()
    st = normalize_smear_type(smear_type)
    types = _parse_cell_types(target_cell_types)
    req_dpi = _effective_dpi(dpi)

    smear_target_detections = [
        spec
        for spec in catalog
        if spec.kind == KIND_DETECTION
        and spec.smear_matches(st)
        and spec.target_matches(types)
    ]
    detection = [spec for spec in smear_target_detections if spec.dpi_matches(req_dpi)]
    if not detection and smear_target_detections:
        return ResolvedModels(dpi_unsuitable=True)

    detection_dpis = {spec.actual_dpi for spec in detection}
    classification: List[ModelSpec] = []
    for spec in catalog:
        if spec.kind != KIND_CLASSIFICATION:
            continue
        if not spec.smear_matches(st) or not spec.target_matches(types):
            continue
        if spec.actual_dpi not in detection_dpis:
            continue
        if not any(spec.targets & det.targets and det.actual_dpi == spec.actual_dpi for det in detection):
            continue
        classification.append(spec)

    score = [
        spec
        for spec in catalog
        if spec.kind == KIND_SCORE
        and spec.dpi_matches(req_dpi)
        and spec.smear_matches(st)
        and spec.target_matches(types)
    ]

    return ResolvedModels(
        detection=detection,
        classification=classification,
        score=score,
    )


def resolve_required_models(
    dpi: int,
    smear_type: str,
    target_cell_types: str,
) -> List[str]:
    """按玻片类型、DPI、目标检测类型解析需加载的 Triton 子模型列表。"""
    return resolve_models(dpi, smear_type, target_cell_types).names


def _normalize_gpu_id(gpu_id: Optional[int]) -> int:
    if gpu_id is None:
        return 0
    gid = int(gpu_id)
    if gid < 0 or gid >= len(TRITON_ENDPOINTS):
        return 0
    return gid


def _warmup_gpu_targets(gpu_id: Optional[int] = None, *, all_gpus: bool = False) -> List[int]:
    """显式 gpu_id 时仅用该卡；all_gpus 时预热全部端点；否则轮询单卡。"""
    if gpu_id is not None:
        return [_normalize_gpu_id(gpu_id)]
    n = len(TRITON_ENDPOINTS)
    if all_gpus and n > 0:
        return list(range(n))
    gid, _ = next_triton_endpoint()
    return [gid]


def _get_http_base_url(gpu_id: Optional[int] = None) -> str:
    """取指定 GPU 的 Triton HTTP 基址。"""
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
    """获取指定 GPU 上已加载的模型列表（Triton repository index，state=READY）。"""
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
    """在指定 GPU 上加载单个模型。"""
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
    """在指定 GPU 上卸载单个模型。"""
    base = _get_http_base_url(gpu_id)
    url = f"{base}/v2/repository/models/{model_name}/unload"
    ok, msg = _http_post(url, b"{}", timeout=timeout)
    if ok:
        logger.info("Model %s unloaded on gpu=%s (%s).", model_name, gpu_id, base)
    else:
        logger.warning("Unload model %s failed on gpu=%s: %s", model_name, gpu_id, msg)
    return ok, msg


def _vram_budget_gb() -> float:
    return max(0.5, float(max_memory) - float(reserved_memory))


def _vram_of(name: str, by_name: dict[str, ModelSpec]) -> float:
    spec = by_name.get(name)
    return float(spec.vram_gb) if spec else DEFAULT_MODEL_VRAM_GB


def _actual_dpi_of(name: str, by_name: dict[str, ModelSpec]) -> int | None:
    spec = by_name.get(name)
    return spec.actual_dpi if spec else None


def _estimated_vram(names: set[str], by_name: dict[str, ModelSpec]) -> float:
    return sum(_vram_of(n, by_name) for n in names)


def _evict_other_dpi_models(
    needed: List[ModelSpec],
    gpu_id: int,
) -> None:
    """显存不够时卸载其它 actual_dpi 层级的已加载模型。"""
    by_name = _specs_by_name()
    needed_names = {spec.name for spec in needed}
    needed_dpis = {spec.actual_dpi for spec in needed}
    budget = _vram_budget_gb()
    loaded = set(get_loaded_models(gpu_id=gpu_id))
    future = loaded | needed_names
    if _estimated_vram(future, by_name) <= budget + 1e-6:
        return

    other = [
        name
        for name in loaded
        if name not in needed_names
        and _actual_dpi_of(name, by_name) not in needed_dpis
    ]
    if other:
        logger.info(
            "VRAM not enough (need ~%.1fGB, budget %.1fGB); "
            "unloading other DPI-level models on gpu=%s: %s",
            _estimated_vram(future, by_name),
            budget,
            gpu_id,
            other,
        )
        for name in other:
            unload_model(name, gpu_id=gpu_id)

    loaded = set(get_loaded_models(gpu_id=gpu_id))
    future = loaded | needed_names
    used = _estimated_vram(future, by_name)
    if used <= budget + 1e-6:
        return

    extra_same_dpi = [
        name
        for name in loaded
        if name not in needed_names
        and _actual_dpi_of(name, by_name) in needed_dpis
    ]
    if extra_same_dpi:
        logger.info(
            "VRAM still not enough (~%.1fGB / %.1fGB); "
            "unloading extra same-DPI models on gpu=%s: %s",
            used,
            budget,
            gpu_id,
            extra_same_dpi,
        )
        for name in extra_same_dpi:
            unload_model(name, gpu_id=gpu_id)


def load_models(
    dpi: int,
    smear_type: str,
    target_cell_types: str,
    *,
    gpu_id: Optional[int] = None,
    all_gpus: bool = False,
) -> Tuple[bool, str, List[str]]:
    """
    按需加载模型：参数为玻片类型、DPI、目标检测类型。
    加载前按 max_memory-reserved_memory 判断显存；不够则卸载其它 DPI 层级模型。
    """
    resolved = resolve_models(dpi, smear_type, target_cell_types)
    if resolved.dpi_unsuitable:
        return False, "DPI不合适", []
    needed = resolved.specs
    models = [spec.name for spec in needed]
    if not models:
        return True, "", []

    last_msg = ""
    with _model_lock:
        for gid in _warmup_gpu_targets(gpu_id, all_gpus=all_gpus):
            _evict_other_dpi_models(needed, gid)
            by_name = _specs_by_name()
            loaded = set(get_loaded_models(gpu_id=gid))
            future = loaded | set(models)
            used = _estimated_vram(future, by_name)
            budget = _vram_budget_gb()
            if used > budget + 1e-6:
                msg = (
                    f"VRAM not enough on gpu={gid}: need ~{used:.1f}GB, "
                    f"budget {budget:.1f}GB (max_memory={max_memory}, "
                    f"reserved_memory={reserved_memory})"
                )
                logger.warning(msg)
                return False, msg, models
            for model_name in models:
                if model_name in loaded:
                    continue
                ok, msg = load_model(model_name, gpu_id=gid)
                if not ok:
                    return False, msg, models
                loaded.add(model_name)
                last_msg = msg
    return True, last_msg, models


def unload_models(
    dpi: int,
    smear_type: str,
    target_cell_types: str,
    *,
    gpu_id: Optional[int] = None,
    all_gpus: bool = False,
) -> Tuple[bool, str, List[str]]:
    """
    按需卸载模型：参数为玻片类型、DPI、目标检测类型。
    不检查是否已加载，直接调用 Triton unload。
    """
    models = resolve_required_models(dpi, smear_type, target_cell_types)
    if not models:
        return True, "", []

    last_msg = ""
    for gid in _warmup_gpu_targets(gpu_id, all_gpus=all_gpus):
        for model_name in reversed(models):
            ok, msg = unload_model(model_name, gpu_id=gid)
            if not ok:
                return False, msg, models
            last_msg = msg
    return True, last_msg, models


def ensure_model_loaded(
    model_name: str,
    max_models: Optional[int] = None,
    max_groups: Optional[int] = None,
    gpu_id: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    兼容滤镜等单模型/pipeline 加载；细胞检测请用 load_models。
    max_models / max_groups 保留签名兼容，已不再使用。
    """
    del max_models, max_groups
    if model_name == "Image_enhance_pipeline":
        models = list(_IMAGE_ENHANCE_MODELS)
    else:
        models = [model_name]

    gid = _normalize_gpu_id(gpu_id)
    enhance_specs = [
        ModelSpec(
            name=name,
            kind="filter",
            actual_dpi=0,
            smear_types=frozenset(),
            targets=frozenset(),
            output="",
            vram_gb=DEFAULT_MODEL_VRAM_GB,
        )
        for name in models
    ]
    with _model_lock:
        _evict_other_dpi_models(enhance_specs, gid)
        for name in models:
            ok, msg = load_model(name, gpu_id=gid)
            if not ok:
                return False, msg
    return True, ""


def warmup_pinned_models_at_startup() -> None:
    """启动预热（暂时关闭，模型按需经 load_models 加载）。"""
    logger.info("Startup warmup disabled; models load on demand via load_models.")


def warmup_model(
    dpi: int,
    smear_type: str = "BM",
    algorithm_types: str = "",
    *,
    gpu_id: Optional[int] = None,
    all_gpus: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    预热模型：三步解析 + load_models。
    返回 (MODEL_TABLE 命中的模型名逗号串, dpi_warning)。
    """
    resolved = resolve_models(int(dpi), smear_type, algorithm_types)
    model_names = ",".join(resolved.names)
    ok, msg, loaded = load_models(
        int(dpi),
        smear_type,
        algorithm_types,
        gpu_id=gpu_id,
        all_gpus=all_gpus,
    )
    if not ok:
        logger.warning(
            "Model warmup failed for %s models=%s: %s",
            model_names,
            loaded,
            msg,
        )
        if msg == "DPI不合适":
            return model_names, msg
    else:
        targets = _warmup_gpu_targets(gpu_id, all_gpus=all_gpus)
        logger.info(
            "Model warmup ok: detection=%s classification=%s score=%s gpus=%s",
            [s.name for s in resolved.detection],
            [s.name for s in resolved.classification],
            [s.name for s in resolved.score],
            targets,
        )
    return model_names, resolved.warning
