# -*- coding: utf-8 -*-
"""DPI + smear_type + target_cell_types 有效组合校验"""
from __future__ import annotations

from backend.tools.MESSAGE_DICT import DPI_NOT_SUITABLE, model_dpi_ranges

DPI_35000 = 35000
DPI_71000 = 71000
DPI_147246 = 147246
DPI_357378 = 357378
DPI_714756 = 714756
DPI_144750 = 144750
BM_PB_DPI_BASES = (147246, 357378, 714756)
CF_DPI_BASES = (35000, 71000)
VALID_DPI_BASES = BM_PB_DPI_BASES
DPI_OUT_OF_RANGE_WARNING_BM_PB = DPI_NOT_SUITABLE
DPI_OUT_OF_RANGE_WARNING_CF = DPI_NOT_SUITABLE
TOLERANCE = 0.1

# 遗留倍率 -> 实际模型 DPI
LEGACY_DPI_MAP = {40: 147246, 50: 357378, 100: 714756}

_SMEAR_ALIASES = {"CF": "CSF"}
_TARGET_ALIASES = {"RED": "RBC", "PLT": "PLAT"}


def normalize_smear_type(smear_type: str | None) -> str:
    st = (smear_type or "BM").strip().upper()
    return _SMEAR_ALIASES.get(st, st)


def _parse_cell_types(s: str) -> set[str]:
    """解析 target_cell_types / algorithm_types，如 'WBC,MEG' 或 'BM_WBC,BM_MEG' -> {'WBC','MEG'}"""
    if not s or not s.strip():
        return set()
    result = set()
    for part in s.replace(" ", "").split(","):
        part = part.strip().upper()
        if not part:
            continue
        if "_" in part:
            part = part.split("_")[-1]
        part = _TARGET_ALIASES.get(part, part)
        result.add(part)
    return result


def _get_dpi_bucket(dpi: int, smear_type: str | None = None) -> tuple[int | None, str | None]:
    """
    DPI 落在 MODEL_TABLE 区间内则返回对应 actual_dpi；
    否则返回 (None, 'DPI不合适')，不允许回退到最近模型。
    """
    del smear_type
    req = LEGACY_DPI_MAP.get(int(dpi), int(dpi))
    ranges = model_dpi_ranges()
    for actual, (low, high) in ranges.items():
        if low <= req <= high:
            return actual, None
    return None, DPI_NOT_SUITABLE


def validate_combo(
    dpi: int,
    smear_type: str,
    target_cell_types: str,
    *,
    allow_empty_types: bool = False,
) -> tuple[bool, str | None]:
    """
    校验 (dpi, smear_type, target_cell_types) 是否能命中定位模型。
    DPI 不在适用区间内时返回 (False, 'DPI不合适')。
    """
    from backend.tools.model_control import resolve_models

    st = normalize_smear_type(smear_type)
    if st not in {"BM", "PB", "CSF"}:
        return False, f"Invalid smear_type: must be BM/PB/CSF, got {st}"

    requested = _parse_cell_types(target_cell_types)
    resolved = resolve_models(int(dpi), st, target_cell_types)
    if resolved.dpi_unsuitable:
        return False, DPI_NOT_SUITABLE

    if allow_empty_types and not requested:
        return True, None

    if not requested:
        return False, "target_cell_types cannot be empty"

    covered: set[str] = set()
    for spec in resolved.detection:
        covered |= spec.targets
    missing = requested - covered
    if missing:
        return False, (
            f"Invalid combo: DPI={dpi} smear_type={st} has no detection model for "
            f"{sorted(missing)}"
        )
    if not resolved.detection:
        return False, DPI_NOT_SUITABLE
    return True, None
