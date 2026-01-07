# layers.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from uuid import uuid4

from tiles import Tile
from cells import Cell


@dataclass
class Layer:
    """
    某一倍率下的完整图层（例如 40x 全扫层 / 100x 采样层）
    """
    dpi: int
    tiles: Dict[str, Tile] = field(default_factory=dict, repr=False)

    # ---------- Tile 管理 ----------
    def add_tile(self, x: int, y: int, w: int, h: int,
                 image_path: Optional[str] = None,
                 image_data: Optional[bytes] = None,
                 extra_meta: Optional[dict] = None) -> str:
        """
        添加一个瓦片到当前层。
        """
        image_uid = uuid4().hex
        tile = Tile(
            image_uid=image_uid,
            x=x,
            y=y,
            w=w,
            h=h,
            image_path=image_path,
            image_data=image_data,
            meta=extra_meta,
        )
        self.tiles[image_uid] = tile
        return image_uid

    def get_tile(self, image_uid) -> Optional[Tile]:
        return self.tiles.get(image_uid)

    def iter_tiles(self) -> List[Tile]:
        return list(self.tiles.values())

    # ---------- ROI 细胞查询 ----------
    def iter_cells_in_roi(self, xmin, ymin, xmax, ymax, is_Cell: bool = False) -> list[dict | Cell]:
        if xmin is None:
            xmin = 0
        if ymin is None:
            ymin = 0
        if xmax is None:
            xmax = float('inf')
        if ymax is None:
            ymax = float('inf')
        result: List[Dict | Cell] = []
        for tile in self.tiles.values():
            for cell in tile.cells:
                position_x = tile.meta.get('position_x', 0)
                position_y = tile.meta.get('position_y', 0)
                if not ((cell.cell_xmax + position_x) < xmin or (cell.cell_xmin + position_x) > xmax or
                        (cell.cell_ymax + position_y) < ymin or (cell.cell_ymin + position_y) > ymax):
                    if is_Cell:
                        result.append(cell)
                    else:
                        result.append(cell.to_dict())
        return result

    def to_dict(self) -> dict:
        return {
            "dpi": self.dpi,
            "tiles": {
                uid: tile.to_dict()
                for uid, tile in self.tiles.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Layer":
        layer = cls(dpi=data["dpi"])
        for uid, tile_data in data.get("tiles", {}).items():
            layer.tiles[uid] = Tile.from_dict(tile_data)
        return layer
