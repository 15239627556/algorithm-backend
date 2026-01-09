# contracts.py
from __future__ import annotations

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2] 
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from project.tiles import Tile

from typing import Optional, Iterable, List, Any
from data_structure import ScoreRegion
from config import BM40Config

class TileContractError(ValueError):
    """Tile 或其数据不符合要求时抛出"""
    pass

class ConfigContractError(ValueError):
    """配置参数不合法时抛出"""
    pass

def validate_tile_geometry(tile: Tile) -> None:
    """校验 tile 的全局几何信息"""
    if tile.x is None or tile.y is None:
        raise TileContractError(f"Tile 缺少全局坐标 (x/y), image_uid={tile.image_uid}")
    if tile.w <= 0 or tile.h <= 0:
        raise TileContractError(f"Tile 尺寸无效, w={tile.w}, h={tile.h}, image_uid={tile.image_uid}")

def validate_regions_in_tile(tile: Tile, regions: Iterable[ScoreRegion], config: BM40Config, *, allow_outside: bool = False) -> None:
    """
    校验 ScoreRegion 的几何是否合理。
    修正点：校验时需将瓦片宽高乘以 config.scale，以匹配缩放后的局部区域坐标。
    """
    validate_tile_geometry(tile)

    scaled_w = tile.w 
    scaled_h = tile.h

    for r in regions:
        if r.w <= 0 or r.h <= 0:
            raise TileContractError(
                f"ScoreRegion 尺寸无效, image_uid={tile.image_uid}, w={r.w}, h={r.h}"
            )

        if allow_outside:
            continue

        # 修正：判断区域是否超出缩放后的瓦片边界
        if r.lx < 0 or r.ly < 0 or (r.lx + r.w) > scaled_w or (r.ly + r.h) > scaled_h:
            raise TileContractError(
                f"ScoreRegion 超出瓦片边界, image_uid={tile.image_uid}, "
                f"region=({r.lx},{r.ly},{r.w},{r.h}), scaled_tile_wh=({scaled_w},{scaled_h})"
            )

def validate_tile_scores_structure(tile: Tile) -> None:
    """
    校验 meta['scores'] 的内部列表结构。
    """
    scores = tile.meta.get("scores")
    if scores is None:
        return
    
    if not isinstance(scores, list):
        raise TileContractError(f"Tile meta['scores'] 必须是列表格式, image_uid={tile.image_uid}")

    for idx, row in enumerate(scores):
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            raise TileContractError(
                f"Tile score 第 {idx} 行格式错误，长度需 >= 5 [lx, ly, lw, lh, score], "
                f"image_uid={tile.image_uid}"
            )

def validate_tile_cells(tile: Tile) -> None:
    """
    校验 Cell 的几何数据。
    """
    for cell in tile.cells:
        if cell.cell_xmin >= cell.cell_xmax or cell.cell_ymin >= cell.cell_ymax:
            raise TileContractError(
                f"Cell 坐标异常: xmin={cell.cell_xmin}, xmax={cell.cell_xmax}, "
                f"image_uid={tile.image_uid}"
            )

def validate_bm40_config(config: BM40Config) -> None:
    """
    校验配置参数。
    """
    if config.cell_size <= 0:
        raise ConfigContractError(f"cell_size 必须大于 0, 当前为 {config.cell_size}")
    
    if config.target_cell_num <= 0:
        raise ConfigContractError(f"target_cell_num 必须为正数, 当前为 {config.target_cell_num}")

    if config.scale <= 0:
        raise ConfigContractError(f"坐标缩放比例 config.scale 必须大于 0")

def validate_tile_full_contract(tile: Tile, config: BM40Config) -> None:
    """
    对 Tile 进行全量契约检查。
    """
    validate_tile_geometry(tile)
    validate_tile_scores_structure(tile)
    validate_tile_cells(tile)
    # 注意：如果此处需要校验 regions，应先通过 adapter 提取后再调用 validate_regions_in_tile