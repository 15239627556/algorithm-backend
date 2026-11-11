# project_info.py
"""从项目同目录的 info 文件读取 dpi / heatmap_orientation 等配置。

项目文件可以是 {stem}.json、{stem}.roi.npz 或 {stem}.info.json，三者共用同一个 stem。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional


class ProjectInfoParams(NamedTuple):
    dpi: int
    heatmap_orientation: int
    tile_w: int
    tile_h: int
    smear_type: Optional[str]
    info_path: Path


_SLIDE_SUFFIXES = (".roi.npz", ".info.json", ".json")


def slide_stem(slide_path: Path) -> str:
    """去掉 .roi.npz / .info.json / .json 后缀，得到项目 stem。"""
    name = Path(slide_path).name
    for suffix in _SLIDE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(slide_path).stem


def resolve_info_path(slide_path: Path) -> Optional[Path]:
    """同路径下查找 info：{stem}.info.json 或 info.json。"""
    slide_path = Path(slide_path)
    stem = slide_stem(slide_path)
    candidates = [
        slide_path.parent / f"{stem}.info.json",
        slide_path.parent / "info.json",
    ]
    # 兼容 *_old.json / *_before_dedup.json：回退到去掉后缀后的 stem.info.json
    for suffix in ("_old", "_before_dedup", "_dedup"):
        if stem.endswith(suffix):
            base = stem[: -len(suffix)]
            candidates.append(slide_path.parent / f"{base}.info.json")
            break
    seen = set()
    for p in candidates:
        p = p.resolve()
        if p in seen:
            continue
        seen.add(p)
        if p.is_file():
            return p
    return None


def resolve_roi_path(slide_path: Path) -> Optional[Path]:
    """同路径下查找 {stem}.roi.npz。"""
    slide_path = Path(slide_path)
    roi = slide_path.parent / f"{slide_stem(slide_path)}.roi.npz"
    return roi if roi.is_file() else None


def load_project_info(info_path: Path) -> Dict[str, Any]:
    with open(info_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"info 文件应为 JSON 对象: {info_path}")
    return data


def info_int(info: Dict[str, Any], key: str, default: Optional[int] = None) -> int:
    if key not in info or info[key] is None:
        if default is None:
            raise KeyError(f"info 缺少字段 {key!r}")
        return int(default)
    return int(info[key])


def load_dpi_and_orientation(slide_path: Path) -> ProjectInfoParams:
    """
    从项目文件（.json / .roi.npz / .info.json）旁的 info 读取运行参数。
    返回 ProjectInfoParams(dpi, heatmap_orientation, tile_w, tile_h, smear_type, info_path)。
    - heatmap_orientation 缺省为 1（头部在左）
    - tile_width / tile_height 必填
    - smear_type 可选（BM/PB 等）
    """
    slide_path = Path(slide_path)
    info_path = resolve_info_path(slide_path)
    if info_path is None:
        raise FileNotFoundError(
            f"未找到 info 文件（{{stem}}.info.json 或 info.json）: {slide_path.parent}"
        )
    info = load_project_info(info_path)
    dpi = info_int(info, "dpi")
    heatmap_orientation = info_int(info, "heatmap_orientation", default=1)
    tile_w = info_int(info, "tile_width")
    tile_h = info_int(info, "tile_height")
    smear_raw = info.get("smear_type")
    smear_type = str(smear_raw).strip() if smear_raw not in (None, "") else None
    return ProjectInfoParams(
        dpi=dpi,
        heatmap_orientation=heatmap_orientation,
        tile_w=tile_w,
        tile_h=tile_h,
        smear_type=smear_type,
        info_path=info_path,
    )
