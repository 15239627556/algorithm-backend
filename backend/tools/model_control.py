# model_control.py
from __future__ import annotations

import json
import logging
import os
import re
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
# Triton 仓库名与 MODEL_TABLE.name 不一致时的映射（驱逐 / 显存估算用 catalog 元数据）
_TRITON_MODEL_ALIASES: dict[str, str] = {
    "DPI147246_BM_PB_cell_analysis": "LOWRES-CELLULARITY",  # 遗留 Triton 名
}

LOAD_TIMEOUT = int(os.environ.get("TRITON_LOAD_TIMEOUT", "600"))
_GPU_METRICS_TIMEOUT_S = float(os.environ.get("GPU_METRICS_TIMEOUT_S", "3"))
# 卸载后指标可能还停在旧值。新读数比「卸载前剩余 + 刚卸掉的模型」少这么多，就认为指标没跟上。
_GPU_METRICS_LAG_GB = 0.25
_NV_GPU_MEMORY_LINE = re.compile(
    r"^(nv_gpu_memory_(?:total|used)_bytes)(?:\{([^}]*)\})?\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)

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
    optional: bool = False
    single_only: bool = False

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

    def without_names(self, names: set[str]) -> ResolvedModels:
        skip = set(names)
        return ResolvedModels(
            detection=[spec for spec in self.detection if spec.name not in skip],
            classification=[spec for spec in self.classification if spec.name not in skip],
            score=[spec for spec in self.score if spec.name not in skip],
            warning=self.warning,
            dpi_unsuitable=self.dpi_unsuitable,
        )

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
    actual_raw = row.get("actual_dpi")
    actual_dpi = int(actual_raw) if actual_raw is not None else 0
    return ModelSpec(
        name=str(row["name"]),
        kind=kind,
        actual_dpi=actual_dpi,
        smear_types=smear_types,
        targets=targets,
        output=str(row.get("output") or ""),
        dpi_min=dpi_min,
        dpi_max=dpi_max,
        vram_gb=vram,
        optional=bool(row.get("optional")),
        single_only=bool(row.get("single_only")),
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
            optional=spec.optional,
            single_only=spec.single_only,
        )
    return spec


def get_model_catalog() -> Tuple[ModelSpec, ...]:
    """从 MESSAGE_DICT.MODEL_TABLE 构建目录，并应用 camera 替换。"""
    return tuple(_apply_camera_override(_row_to_spec(row)) for row in MODEL_TABLE)


def _specs_by_name() -> dict[str, ModelSpec]:
    return {spec.name: spec for spec in get_model_catalog()}


def _effective_dpi(dpi: int) -> int:
    return LEGACY_DPI_MAP.get(int(dpi), int(dpi))


def _include_classifier(spec: ModelSpec, include_single_only: bool) -> bool:
    """single_only 分类器默认不进平扫；单张识别显式打开。"""
    return include_single_only or not spec.single_only


def resolve_models(
    dpi: int,
    smear_type: str,
    target_cell_types: str,
    *,
    include_single_only: bool = False,
) -> ResolvedModels:
    """
    三步解析需加载的模型：
    1. 按 DPI 区间 + 玻片类型 + 目标类型匹配定位模型
    2. 按玻片类型 + 目标类型匹配分类模型（DPI 不限制，须与已命中定位模型的实际 DPI 对齐）
    3. 按 DPI 区间 + 玻片类型匹配评分模型（目标不限制）
    DPI 不在任何适用区间内时 dpi_unsuitable=True，不回退到最近模型。
    include_single_only: 单张识别才纳入 single_only 分类器；平扫保持 False。
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
        if not _include_classifier(spec, include_single_only):
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


def resolve_models_by_actual_dpi(
    actual_dpi: int,
    smear_type: str,
    target_cell_types: str,
    *,
    include_single_only: bool = False,
) -> ResolvedModels:
    """
    只加载指定 actual_dpi 的模型，不看 dpi_range。
    用于小程序猜 DPI：图像按比值强制缩放到该档，推理始终走这一档模型。
    """
    catalog = get_model_catalog()
    st = normalize_smear_type(smear_type)
    types = _parse_cell_types(target_cell_types)
    actual = int(actual_dpi)

    detection = [
        spec
        for spec in catalog
        if spec.kind == KIND_DETECTION
        and spec.actual_dpi == actual
        and spec.smear_matches(st)
        and spec.target_matches(types)
    ]
    classification: List[ModelSpec] = []
    for spec in catalog:
        if spec.kind != KIND_CLASSIFICATION or spec.actual_dpi != actual:
            continue
        if not _include_classifier(spec, include_single_only):
            continue
        if not spec.smear_matches(st) or not spec.target_matches(types):
            continue
        if not any(spec.targets & det.targets for det in detection):
            continue
        classification.append(spec)

    score = [
        spec
        for spec in catalog
        if spec.kind == KIND_SCORE
        and spec.actual_dpi == actual
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
    if not ok:
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
    if not ok:
        logger.warning("Unload model %s failed on gpu=%s: %s", model_name, gpu_id, msg)
    return ok, msg


def _vram_budget_gb() -> float:
    """查不到实际剩余时的上限，用总显存。实际剩余里已经含了当前占用。"""
    return max(0.5, float(max_memory))


def _endpoint_gpu_uuid(gpu_id: int) -> str:
    return str(get_triton_endpoint(gpu_id).get("gpu_uuid") or "").strip()


def _metrics_url(gpu_id: int) -> str:
    endpoint = get_triton_endpoint(gpu_id)
    ip = str(endpoint.get("ip") or "127.0.0.1")
    port = int(endpoint.get("metrics_port") or 8002)
    return f"http://{ip}:{port}/metrics"


def _gpu_series_key(labels: str) -> str:
    """同一张卡的 total/used 行用同一个键对齐。优先 gpu_uuid。"""
    uuid = re.search(r'gpu_uuid="([^"]*)"', labels or "")
    if uuid:
        return uuid.group(1)
    gpu = re.search(r'\bgpu="([^"]*)"', labels or "")
    if gpu:
        return gpu.group(1)
    return labels or ""


def _parse_gpu_free_vram_gb(metrics_text: str, gpu_uuid: str) -> float | None:
    """从 Triton Prometheus 文本取出指定 gpu_uuid 的剩余显存（GB）。"""
    totals: dict[str, float] = {}
    useds: dict[str, float] = {}
    for raw in metrics_text.splitlines():
        match = _NV_GPU_MEMORY_LINE.match(raw.strip())
        if not match:
            continue
        name, labels, value = match.group(1), match.group(2) or "", float(match.group(3))
        key = _gpu_series_key(labels)
        if name.endswith("total_bytes"):
            totals[key] = value
        else:
            useds[key] = value
    if not totals:
        return None
    want = (gpu_uuid or "").strip()
    if want:
        if want not in totals:
            return None
        return max(0.0, totals[want] - useds.get(want, 0.0)) / (1024 ** 3)
    if len(totals) == 1:
        key = next(iter(totals))
        return max(0.0, totals[key] - useds.get(key, 0.0)) / (1024 ** 3)
    return None


def _query_gpu_free_vram_gb(gpu_id: int) -> float | None:
    """读指定 gpu_id 对应 Triton 的 /metrics，只取该端点 gpu_uuid 这张卡的剩余显存。"""
    gpu_uuid = _endpoint_gpu_uuid(gpu_id)
    if not gpu_uuid:
        return None
    url = _metrics_url(gpu_id)
    try:
        with urllib.request.urlopen(url, timeout=_GPU_METRICS_TIMEOUT_S) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("query GPU free VRAM failed gpu=%s url=%s: %s", gpu_id, url, e)
        return None
    free = _parse_gpu_free_vram_gb(text, gpu_uuid)
    if free is None:
        logger.warning(
            "GPU metrics have no memory for gpu=%s uuid=%s url=%s",
            gpu_id,
            gpu_uuid or "-",
            url,
        )
        return None
    return free


def _vram_allowance(
    gpu_id: int,
    names: set[str],
    loaded: set[str],
    by_name: dict[str, ModelSpec],
) -> tuple[float, float, str]:
    """
    返回 (需要的 GB, 上限 GB, 来源)。
    该卡 GPU_UUID 为空：需要量是已加载加待加载的估算，上限是 max_memory。
    GPU_UUID 有值：需要量只算还没加载的模型，上限是这张卡的实际剩余显存。
    """
    if not _endpoint_gpu_uuid(gpu_id):
        need = _estimated_vram(set(loaded) | set(names), by_name)
        limit = _vram_budget_gb()
        return need, limit, "configured"
    free = _query_gpu_free_vram_gb(gpu_id)
    if free is not None:
        return _pending_vram(names, loaded, by_name), free, "actual"
    used = _estimated_vram(set(loaded), by_name)
    estimated = max(0.0, _vram_budget_gb() - used)
    logger.warning(
        "gpu=%s actual free VRAM unavailable, fallback %.2fGB "
        "(max_memory=%s loaded~%.2fGB)",
        gpu_id,
        estimated,
        max_memory,
        used,
    )
    return _pending_vram(names, loaded, by_name), estimated, "estimated"


def _pending_vram(
    names: set[str],
    loaded: set[str],
    by_name: dict[str, ModelSpec],
) -> float:
    """还没在这张卡上的模型才占用即将发生的显存。已加载的已经反映在剩余显存里。"""
    return _estimated_vram(set(names) - set(loaded), by_name)


def _vram_of(name: str, by_name: dict[str, ModelSpec]) -> float:
    spec = by_name.get(name)
    return float(spec.vram_gb) if spec else DEFAULT_MODEL_VRAM_GB


def _actual_dpi_of(name: str, by_name: dict[str, ModelSpec]) -> int | None:
    spec = by_name.get(name)
    return spec.actual_dpi if spec else None


def _estimated_vram(names: set[str], by_name: dict[str, ModelSpec]) -> float:
    return sum(_vram_of(n, by_name) for n in names)


def _resolve_load_spec(model_name: str) -> ModelSpec:
    """解析单模型加载时的 ModelSpec（含 Triton 名 → MODEL_TABLE 别名）。"""
    by_name = _specs_by_name()
    if model_name in by_name:
        return by_name[model_name]
    alias = _TRITON_MODEL_ALIASES.get(model_name)
    if alias and alias in by_name:
        base = by_name[alias]
        return ModelSpec(
            name=model_name,
            kind=base.kind,
            actual_dpi=base.actual_dpi,
            smear_types=base.smear_types,
            targets=base.targets,
            output=base.output,
            dpi_min=base.dpi_min,
            dpi_max=base.dpi_max,
            vram_gb=base.vram_gb,
        )
    actual_dpi = 0
    for token in ("147246", "357378", "714756", "35000", "71000", "144750"):
        if token in model_name:
            actual_dpi = int(token)
            break
    return ModelSpec(
        name=model_name,
        kind="single",
        actual_dpi=actual_dpi,
        smear_types=frozenset(),
        targets=frozenset(),
        output="",
        vram_gb=DEFAULT_MODEL_VRAM_GB,
    )


def _load_specs_by_triton_name(specs: List[ModelSpec]) -> dict[str, ModelSpec]:
    """Triton 模型名 → spec，供显存估算时查 vram_gb / actual_dpi。"""
    by_name = _specs_by_name()
    out = dict(by_name)
    for spec in specs:
        out[spec.name] = spec
    return out


def _allowance_after_unload(
    gpu_id: int,
    names: set[str],
    loaded: set[str],
    by_name: dict[str, ModelSpec],
    free_before: float,
    unloaded: list[str],
) -> tuple[float, float, str]:
    """卸载后再算一次。UUID 为空仍用总显存；有 UUID 时指标没跟上就加上刚卸掉的模型。"""
    need, limit, source = _vram_allowance(gpu_id, names, loaded, by_name)
    if not unloaded or source != "actual":
        return need, limit, source
    expected = free_before + _estimated_vram(set(unloaded), by_name)
    if limit < expected - _GPU_METRICS_LAG_GB:
        return need, expected, "estimated-after-unload"
    return need, limit, source


def _evict_other_dpi_models(
    needed: List[ModelSpec],
    gpu_id: int,
) -> tuple[float, float, str]:
    """放不下时先卸载用不到的模型，再按卸载后的剩余显存判断。返回 (需要, 上限, 来源)。"""
    by_name = _load_specs_by_triton_name(needed)
    needed_names = {spec.name for spec in needed}
    needed_dpis = {spec.actual_dpi for spec in needed}
    loaded = set(get_loaded_models(gpu_id=gpu_id))
    need, limit, source = _vram_allowance(gpu_id, needed_names, loaded, by_name)
    remaining = "实际剩余" if source == "actual" else "剩余"
    # 待加载为 0 说明模型已在卡上，不打日志，避免每个请求一行。
    if need <= limit + 1e-6:
        if need > 1e-6:
            logger.info(
                "gpu=%s 加载新模型需要 %.1fGB，%s %.1fGB",
                gpu_id,
                need,
                remaining,
                limit,
            )
        return need, limit, source
    has_other = any(name not in needed_names for name in loaded)
    logger.info(
        "gpu=%s 加载新模型需要 %.1fGB，%s %.1fGB%s",
        gpu_id,
        need,
        remaining,
        limit,
        "，不够，卸载其他模型" if has_other else "，不够",
    )

    other = [
        name
        for name in loaded
        if name not in needed_names
        and _actual_dpi_of(name, by_name) not in needed_dpis
    ]
    if other:
        unloaded: list[str] = []
        for name in other:
            ok, _msg = unload_model(name, gpu_id=gpu_id)
            if ok:
                unloaded.append(name)
        loaded = set(get_loaded_models(gpu_id=gpu_id))
        need, limit, source = _allowance_after_unload(
            gpu_id, needed_names, loaded, by_name, limit, unloaded,
        )
        if unloaded:
            logger.info("gpu=%s 卸载后剩余 %.1fGB", gpu_id, limit)
        if need <= limit + 1e-6:
            return need, limit, source

    extra_same_dpi = [
        name
        for name in loaded
        if name not in needed_names
        and _actual_dpi_of(name, by_name) in needed_dpis
    ]
    if extra_same_dpi:
        unloaded = []
        for name in extra_same_dpi:
            ok, _msg = unload_model(name, gpu_id=gpu_id)
            if ok:
                unloaded.append(name)
        loaded = set(get_loaded_models(gpu_id=gpu_id))
        need, limit, source = _allowance_after_unload(
            gpu_id, needed_names, loaded, by_name, limit, unloaded,
        )
        if unloaded:
            logger.info("gpu=%s 卸载后剩余 %.1fGB", gpu_id, limit)
    return need, limit, source


def _drop_single_only_not_needed(
    needed: List[ModelSpec],
    gpu_id: int,
    loaded: set[str],
) -> set[str]:
    """本次不需要的 single_only 模型卸掉，避免平扫仍走分类。"""
    needed_names = {spec.name for spec in needed}
    by_name = _specs_by_name()
    for name in list(loaded):
        spec = by_name.get(name)
        if spec is None or not spec.single_only or name in needed_names:
            continue
        ok, msg = unload_model(name, gpu_id=gpu_id)
        if ok:
            loaded.discard(name)
        else:
            logger.warning(
                "Unload single-only model %s failed on gpu=%s: %s",
                name,
                gpu_id,
                msg,
            )
    return loaded


def _vram_error(gpu_id: int, need: float, limit: float, source: str) -> str:
    if source == "actual":
        return (
            f"VRAM not enough on gpu={gpu_id}: unloaded models need ~{need:.1f}GB, "
            f"free {limit:.1f}GB (gpu metrics)"
        )
    if source == "configured":
        return (
            f"VRAM not enough on gpu={gpu_id}: need ~{need:.1f}GB, "
            f"budget {limit:.1f}GB (max_memory={max_memory})"
        )
    return (
        f"VRAM not enough on gpu={gpu_id}: unloaded models need ~{need:.1f}GB, "
        f"free {limit:.1f}GB ({source}, max_memory={max_memory})"
    )


def load_resolved_models(
    resolved: ResolvedModels,
    *,
    gpu_id: Optional[int] = None,
    all_gpus: bool = False,
) -> Tuple[bool, str, List[str]]:
    """
    按已解析的模型列表加载。dpi_unsuitable 时直接拒绝。
    optional 模型加载失败或显存不够时跳过，不让整次请求失败。
    返回值里的模型名是实际就绪的；调用方据此丢掉没加载上的可选分类器。
    """
    if resolved.dpi_unsuitable:
        return False, "DPI不合适", []
    needed = resolved.specs
    if not needed:
        return True, "", []

    required = [spec for spec in needed if not spec.optional]
    optional = [spec for spec in needed if spec.optional]
    required_names = [spec.name for spec in required]
    last_msg = ""
    ready_sets: list[set[str]] = []

    with _model_lock:
        for gid in _warmup_gpu_targets(gpu_id, all_gpus=all_gpus):
            loaded = _drop_single_only_not_needed(
                needed,
                gid,
                set(get_loaded_models(gpu_id=gid)),
            )
            need, limit, source = _evict_other_dpi_models(required, gid)
            if need > limit + 1e-6:
                msg = _vram_error(gid, need, limit, source)
                logger.warning(msg)
                return False, msg, required_names
            by_name = _specs_by_name()
            loaded = set(get_loaded_models(gpu_id=gid))

            ready: set[str] = set()
            for model_name in required_names:
                if model_name not in loaded:
                    ok, msg = load_model(model_name, gpu_id=gid)
                    if not ok:
                        return False, msg, required_names
                    loaded.add(model_name)
                    last_msg = msg
                ready.add(model_name)

            for spec in optional:
                name = spec.name
                if name in loaded:
                    ready.add(name)
                    continue
                need, limit, _source = _vram_allowance(gid, {name}, loaded, by_name)
                if need > limit + 1e-6:
                    need, limit, _source = _evict_other_dpi_models(required + [spec], gid)
                    loaded = set(get_loaded_models(gpu_id=gid))
                    reloaded = False
                    for req_name in required_names:
                        if req_name in loaded:
                            continue
                        ok, msg = load_model(req_name, gpu_id=gid)
                        if not ok:
                            return False, msg, required_names
                        loaded.add(req_name)
                        reloaded = True
                        last_msg = msg
                    if reloaded:
                        need, limit, _source = _vram_allowance(gid, {name}, loaded, by_name)
                if need > limit + 1e-6:
                    logger.warning(
                        "Skip optional model %s on gpu=%s: VRAM not enough",
                        name,
                        gid,
                    )
                    continue
                ok, msg = load_model(name, gpu_id=gid)
                if not ok:
                    logger.warning(
                        "Optional model %s unavailable on gpu=%s, detection only: %s",
                        name,
                        gid,
                        msg,
                    )
                    continue
                loaded.add(name)
                ready.add(name)
                last_msg = msg
            ready_sets.append(ready)

    if not ready_sets:
        return True, last_msg, []
    common = set.intersection(*ready_sets)
    ordered = [spec.name for spec in needed if spec.name in common]
    return True, last_msg, ordered


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
    GPU_UUID 有值时查该卡实际剩余显存；为空时用 max_memory。不够则卸载其它 DPI 层级模型。
    """
    resolved = resolve_models(dpi, smear_type, target_cell_types)
    return load_resolved_models(resolved, gpu_id=gpu_id, all_gpus=all_gpus)


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
    单模型按需加载：与 load_models 相同，先按该卡实际剩余显存驱逐其它 DPI 层级，再 load。
    滤镜 / 增生分析 / 全局图等单模型场景统一走此路径。
    max_models / max_groups 保留签名兼容，已不再使用。
    """
    del max_models, max_groups
    if model_name == "Image_enhance_pipeline":
        model_names = list(_IMAGE_ENHANCE_MODELS)
    else:
        model_names = [model_name]

    gid = _normalize_gpu_id(gpu_id)
    needed = [_resolve_load_spec(name) for name in model_names]
    with _model_lock:
        need, limit, source = _evict_other_dpi_models(needed, gid)
        if need > limit + 1e-6:
            msg = _vram_error(gid, need, limit, source)
            logger.warning(msg)
            return False, msg
        by_name = _load_specs_by_triton_name(needed)
        loaded = set(get_loaded_models(gpu_id=gid))
        for name in model_names:
            if name in loaded:
                continue
            ok, msg = load_model(name, gpu_id=gid)
            if not ok:
                return False, msg
            loaded.add(name)
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
