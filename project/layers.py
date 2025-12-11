# layers.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

from cells import MagnificationLevel, Cell
from tiles import Tile


@dataclass
class Layer:
    """
    某一倍率下的完整图层（例如 40x 全扫层 / 100x 采样层）

    - 由 num_rows * num_cols 个瓦片组成（40x 层）
    - 或者是若干零散瓦片（100x 采样层，可以 num_rows/num_cols 为 0）
    """
    name: str
    magnification: MagnificationLevel
    num_rows: int
    num_cols: int
    tile_width: int
    tile_height: int

    # (row_index, col_index) -> Tile
    tiles: Dict[Tuple[int, int], Tile] = field(default_factory=dict)

    # ---------- Tile 管理 ----------

    def add_tile(self, tile: Tile) -> None:
        """
        添加一个瓦片到当前层。

        如果 (row_index, col_index) 已存在，会覆盖旧的 Tile。
        """
        key = (tile.row_index, tile.col_index)
        self.tiles[key] = tile

    def get_tile(self, row: int, col: int) -> Optional[Tile]:
        """
        获取某个瓦片，不存在则返回 None。
        """
        return self.tiles.get((row, col))

    def iter_tiles(self) -> List[Tile]:
        """
        以列表形式返回当前层的所有瓦片。
        """
        return list(self.tiles.values())

    # ---------- 缺失瓦片检查 ----------

    def check_missing_tiles(self) -> List[dict]:
        """
        检查缺失的瓦片，返回列表，格式对应：
        POST /api/v1/smear_analysis/check_missing_tiles 响应中的 missing_tiles。

        返回示例：
        [
            {
                "row_index": 0,
                "col_index": 1,
                "position_missing": True,
                "image_missing": True
            },
            ...
        ]

        对于 40x 层：根据 num_rows / num_cols 全量检查。
        对于 num_rows/num_cols 为 0 的层，可以按需扩展（此处简单返回已有瓦片 image_missing）。
        """
        missing: List[dict] = []

        # 如果没有配置网格大小，则只检查现有瓦片是否缺图像
        if self.num_rows <= 0 or self.num_cols <= 0:
            for (r, c), tile in self.tiles.items():
                missing.append({
                    "row_index": r,
                    "col_index": c,
                    "position_missing": False,
                    "image_missing": tile.meta.image_uid is None,
                })
            return missing

        # 对于有固定网格的层（如 40x）
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                key = (r, c)
                tile = self.tiles.get(key)
                if tile is None:
                    missing.append({
                        "row_index": r,
                        "col_index": c,
                        "position_missing": True,
                        "image_missing": True,
                    })
                else:
                    missing.append({
                        "row_index": r,
                        "col_index": c,
                        "position_missing": False,
                        "image_missing": tile.meta.image_uid is None,
                    })
        return missing

    # ---------- ROI 细胞查询 ----------

    def iter_cells_in_roi(
        self,
        roi: Tuple[int, int, int, int],
    ) -> List[Cell]:
        """
        返回当前层在指定 ROI 内的所有细胞。

        :param roi: (xmin, ymin, xmax, ymax) 全局坐标
        """
        xmin, ymin, xmax, ymax = roi
        result: List[Cell] = []
        for tile in self.tiles.values():
            for cell in tile.cells:
                # 简单矩形相交判断
                if not (cell.x_max < xmin or cell.x_min > xmax or
                        cell.y_max < ymin or cell.y_min > ymax):
                    result.append(cell)
        return result
