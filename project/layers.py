# layers.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from uuid import uuid4

from .tiles import Tile
from .cells import Cell


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
                 extra_meta: Optional[dict] = None) -> Tile:
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
        return tile

    def get_tile(self, image_uid) -> Optional[Tile]:
        return self.tiles.get(image_uid)

    def iter_tiles(self) -> List[Tile]:
        return list(self.tiles.values())

    # ---------- ROI 细胞查询 ----------
    def iter_cells_in_roi(self, xmin, ymin, xmax, ymax, is_Cell: bool = False) -> list[dict | Cell]:
        """返回在 ROI 内的细胞（全局坐标）。不修改原始 cell 对象。"""
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
            position_x = 0 if tile.x is None else int(tile.x)
            position_y = 0 if tile.y is None else int(tile.y)
            for cell in tile.cells:
                gxmin = cell.cell_xmin + position_x
                gymin = cell.cell_ymin + position_y
                gxmax = cell.cell_xmax + position_x
                gymax = cell.cell_ymax + position_y
                if not (gxmax < xmin or gxmin > xmax or gymax < ymin or gymin > ymax):
                    if is_Cell:
                        result.append(Cell(
                            cell_xmin=gxmin, cell_ymin=gymin, cell_xmax=gxmax, cell_ymax=gymax,
                            cell_type=cell.cell_type, cell_type_name=cell.cell_type_name,
                            class_confidence=cell.class_confidence, bbox_confidence=cell.bbox_confidence,
                            extra=cell.extra.copy() if cell.extra else {}
                        ))
                    else:
                        result.append({
                            "cell_xmin": gxmin, "cell_ymin": gymin, "cell_xmax": gxmax, "cell_ymax": gymax,
                            "cell_type": cell.cell_type, "cell_type_name": cell.cell_type_name,
                            "class_confidence": cell.class_confidence, "bbox_confidence": cell.bbox_confidence,
                            "extra": cell.extra.copy() if cell.extra else {}
                        })
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
