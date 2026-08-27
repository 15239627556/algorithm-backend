# -*- coding: utf-8 -*-
"""DPI + smear_type + target_cell_types 有效组合校验"""
from __future__ import annotations

# BM/PB DPI ±10%: 144750(130275-159225), 357378(321640-393116), 714756(643280-786232)
# CF DPI ±10%: 35000(31500-38500), 71000(63900-78100)
BM_PB_DPI_BASES = (144750, 357378, 714756)
CF_DPI_BASES = (35000, 71000)
VALID_DPI_BASES = BM_PB_DPI_BASES  # 兼容旧引用
DPI_35000 = 35000
DPI_71000 = 71000
DPI_OUT_OF_RANGE_WARNING_BM_PB = "DPI out of valid range (144750/357378/714756 ±10%)"
DPI_OUT_OF_RANGE_WARNING_CF = "DPI out of valid range (35000/71000 ±10%)"
VALID_COMBINATIONS = {
    (144750, "BM"): {"WBC", "MEG"},
    (144750, "PB"): {"WBC", "RBC"},
    # 357378 允许 PLAT，推理走 714756 血小板模型（见 get_model_by_dpi）
    (357378, "BM"): {"WBC", "RBC", "MEG", "PLAT"},
    (357378, "PB"): {"WBC", "RBC", "MEG", "PLAT"},
    (714756, "BM"): {"WBC", "RBC", "MEG", "PLAT"},
    (714756, "PB"): {"WBC", "RBC", "MEG", "PLAT"},
    (35000, "CF"): {"WBC"},
    (71000, "CF"): {"WBC"},
}
TOLERANCE = 0.1

# 遗留倍率 -> DPI（仅 BM/PB）
LEGACY_DPI_MAP = {40: 144750, 50: 357378, 100: 714756}


def _get_dpi_bucket(dpi: int, smear_type: str | None = None) -> tuple[int, str | None]:
    """根据 DPI 与涂片类型返回所属 bucket；超出范围时返回最近 bucket 和告警。"""
    st = (smear_type or "").strip().upper()
    if st == "CF":
        bases = CF_DPI_BASES
        warning_msg = DPI_OUT_OF_RANGE_WARNING_CF
    else:
        bases = BM_PB_DPI_BASES
        warning_msg = DPI_OUT_OF_RANGE_WARNING_BM_PB
        if dpi in LEGACY_DPI_MAP:
            return LEGACY_DPI_MAP[dpi], None
    for base in bases:
        low = int(base * (1 - TOLERANCE))
        high = int(base * (1 + TOLERANCE))
        if low <= dpi <= high:
            return base, None
    return min(bases, key=lambda base: abs(dpi - base)), warning_msg


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
    返回 (True, None)、(True, "告警描述") 或 (False, "错误描述")。
    allow_empty_types: create_task 时 target_cell_types 可为空，后续 upload 再校验。
    """
    st = (smear_type or "BM").strip().upper()
    bucket, warning = _get_dpi_bucket(dpi, smear_type=st)

    key = (bucket, st)
    if key not in VALID_COMBINATIONS:
        valid_st = sorted({k[1] for k in VALID_COMBINATIONS if k[0] == bucket})
        return False, f"Invalid combo: DPI={bucket} smear_type must be one of {valid_st}, got {st}"

    requested = _parse_cell_types(target_cell_types)
    if allow_empty_types and not requested:
        return True, warning

    allowed = VALID_COMBINATIONS[key]
    invalid = requested - allowed
    if invalid:
        return False, (
            f"Invalid combo: DPI={bucket} smear_type={st} target_cell_types must be "
            f"subset of {sorted(allowed)}, got invalid {sorted(invalid)}"
        )

    return True, warning
