# -*- coding: utf-8 -*-
"""DPI + smear_type + target_cell_types 有效组合校验"""
from __future__ import annotations

# 有效组合: (dpi_base, smear_type) -> 允许的 target_cell_types 集合
# dpi 匹配 ±10%: 144750(130275-159225), 357378(321640-393116), 714756(643280-786232)
VALID_COMBINATIONS = {
    (144750, "BM"): {"WBC", "MEG"},
    (144750, "PB"): {"WBC", "RBC", "PLAT"},
    (357378, "BM"): {"MEG", "WBC"},
    (357378, "PB"): {"WBC", "RBC"},
    (714756, "BM"): {"WBC", "RBC", "MEG"},
    (714756, "PB"): {"WBC", "RBC", "MEG"},
    (714756, "CF"): {"WBC"},
}
TOLERANCE = 0.1

# 遗留倍率 -> DPI
LEGACY_DPI_MAP = {40: 144750, 50: 357378, 100: 714756}


def _get_dpi_bucket(dpi: int) -> int | None:
    """根据 DPI 返回所属 bucket，不在任一范围内返回 None"""
    if dpi in LEGACY_DPI_MAP:
        return LEGACY_DPI_MAP[dpi]
    for base in (144750, 357378, 714756):
        low = int(base * (1 - TOLERANCE))
        high = int(base * (1 + TOLERANCE))
        if low <= dpi <= high:
            return base
    return None


def _parse_cell_types(s: str) -> set[str]:
    """解析 target_cell_types / algorithm_types，如 'WBC,MEG' 或 'BM_WBC,BM_MEG' -> {'WBC','MEG'}"""
    if not s or not s.strip():
        return set()
    result = set()
    for part in s.replace(" ", "").split(","):
        part = part.strip().upper()
        if not part:
            continue
        # BM_WBC -> WBC, RBC/PLAT 保持
        if "_" in part:
            part = part.split("_")[-1]
        # RED -> RBC
        if part == "RED":
            part = "RBC"
        result.add(part)
    return result


def validate_combo(
    dpi: int,
    smear_type: str,
    target_cell_types: str,
    *,
    allow_empty_types: bool = False,
) -> tuple[bool, str | None]:
    """
    校验 (dpi, smear_type, target_cell_types) 是否为有效组合。
    返回 (True, None) 或 (False, "错误描述")。
    allow_empty_types: create_task 时 target_cell_types 可为空，后续 upload 再校验。
    """
    bucket = _get_dpi_bucket(dpi)
    if bucket is None:
        return False, f"DPI {dpi} out of valid range (144750/357378/714756 ±10%)"

    st = (smear_type or "BM").strip().upper()
    key = (bucket, st)
    if key not in VALID_COMBINATIONS:
        valid_st = sorted({k[1] for k in VALID_COMBINATIONS if k[0] == bucket})
        return False, f"Invalid combo: DPI={bucket} smear_type must be one of {valid_st}, got {st}"

    requested = _parse_cell_types(target_cell_types)
    if allow_empty_types and not requested:
        return True, None

    allowed = VALID_COMBINATIONS[key]
    invalid = requested - allowed
    if invalid:
        return False, f"Invalid combo: DPI={bucket} smear_type={st} target_cell_types must be subset of {sorted(allowed)}, got invalid {sorted(invalid)}"

    return True, None