# adapters.py ：ScoreRegion + TileAdapter 接口 + 默认实现（兼容 meta["scores"]）
from __future__ import annotations

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2] 
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from project.tiles import Tile

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Optional

from .data_structure import ScoreRegion
from .contracts import TileContractError, validate_regions_in_tile
from .config import BM40Config



class TileAdapter(Protocol):
    """
    Adapter 的目标：屏蔽上游数据格式变化。
    算法层只认：
    - Tile（全局 x/y/w/h）
    - 从 Tile 提取出来的 ScoreRegion 列表
    """
    def tile_from_raw(self, raw: Dict[str, Any]) -> Tile:
        ...

    def get_score_regions(self, tile: Tile) -> List[ScoreRegion]:
        ...


@dataclass
class DefaultScoresAdapter:
    """
    默认适配器：兼容你当前的结构：
    tile.meta["scores"] 为 list[list]，
    每条形如 [lx, ly, w, h, score, cls]，cls 可忽略。
    """
    score_key: str = "scores"
    validate: bool = True
    allow_region_outside_tile: bool = False

    def tile_from_raw(self, raw: Dict[str, Any]) -> Tile:
        # 直接复用你已有的 Tile.from_dict（如果上游 raw 结构变化，你可以在这里做映射/清洗）
        return Tile.from_dict(raw)

    def get_score_regions(self, tile: Tile, config:BM40Config) -> List[ScoreRegion]:
        scores = tile.meta.get(self.score_key, [])
        if scores is None:
            return []

        if not isinstance(scores, list):
            raise TileContractError(
                f"tile.meta[{self.score_key!r}] must be list, image_uid={tile.image_uid}"
            )

        regions: List[ScoreRegion] = []
        for row in scores:
            if not isinstance(row, (list, tuple)):
                continue
            if len(row) < 6:
                continue

            try:
                lx = float(row[0]) * config.scale
                ly = float(row[1]) * config.scale 
                w = float(row[2]) * config.scale 
                h = float(row[3]) * config.scale 
                score = float(row[4])

            except Exception as e:
                raise TileContractError(
                    f"Invalid score row values, image_uid={tile.image_uid}, row={row}, reason={e}"
                )

            regions.append(ScoreRegion(lx=lx, ly=ly, w=w, h=h, score=score))

        if self.validate:
            validate_regions_in_tile(tile, regions, config, allow_outside=self.allow_region_outside_tile)

        return regions

