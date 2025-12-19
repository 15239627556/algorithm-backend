# tiles.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict

from cells import Cell


@dataclass
class Tile:
    """
    表示某一倍率层上的一个瓦片（拼图块）
    - x: int
    - y: int
    - w: int
    - h: int
    - image_path: Optional[str]
    - image_data: Optional[bytes]
    - meta:
     {
        "scores": list[list[float]],
        "num_rows": Optional[int],
        "num_cols": Optional[int],
        "row_index": Optional[int],
        "col_index": Optional[int]
    }
    - cells: List[Cell]  瓦片上的所有细胞检测结果（全局坐标）
    """
    # 瓦片在全局坐标中的左上角位置
    image_uid: str
    x: int
    y: int
    w: int
    h: int
    image_path: Optional[str] = None
    image_data: Optional[bytes] = None
    meta: Dict[str, list | int | str] = field(default_factory=dict)
    cells: List[Cell] = field(default_factory=list, repr=False)

    # ---------- 细胞管理 ----------
    def add_cells(self, cells: List[Cell]) -> None:
        """
        将一批细胞添加到指定瓦片中
        """
        self.cells.extend(cells)

    def add_cell(self, cell: Cell) -> None:
        """
        向当前瓦片添加一个细胞（cell 坐标必须为全局坐标）。
        """
        self.cells.append(cell)

    def release_image_data(self) -> None:
        """
        释放瓦片的图像数据，节省内存。
        """
        self.image_data = None

    def load_image_data(self) -> Dict[str, str]:
        """
        加载瓦片的图像数据（如果 image_path 可用）。
        """
        try:
            with open(self.image_path, 'rb') as f:
                self.image_data = f.read()
            msg = {'msg': 'image data loaded success'}
        except Exception as e:
            msg = {'msg': 'image data loaded failed', 'reason': str(e)}
        return msg

    def to_dict(self) -> dict:
        """
        转换为通用接口字段，便于未来需要返回瓦片信息时使用。
        """
        return {
            "image_uid": self.image_uid,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "image_path": self.image_path,
            "meta": self.meta,
            "cells": [cell.to_dict() for cell in self.cells]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Tile":
        """
        从通用接口字段的字典创建 Tile 实例。
        """
        tile = cls(
            image_uid=data["image_uid"],
            x=data["x"],
            y=data["y"],
            w=data["w"],
            h=data["h"],
            image_path=data.get("image_path"),
            meta=data.get("meta", {}),
        )
        tile.cells = [Cell.from_dict(cell_data) for cell_data in data.get("cells", [])]
        return tile
