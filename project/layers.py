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

    # 只记录“缺失”的格子，不缺的就不存
    # key: (row, col) -> dict(row_index, col_index, position_missing, image_missing)
    _missing: Dict[Tuple[int, int], Dict] = field(default_factory=dict, init=False)

    def __post_init__(self):
        # 对于 40x 全扫层可以选择在这里“全量初始化缺失表”，
        # 也可以懒加载，视你业务而定
        if self.num_rows > 0 and self.num_cols > 0:
            # 初始时所有格子都当成缺失（没有 Tile）
            for r in range(self.num_rows):
                for c in range(self.num_cols):
                    self._missing[(r, c)] = {
                        "row_index": r,
                        "col_index": c,
                        "position_missing": True,  # 没有 Tile
                        "image_missing": True,  # 没有图像
                    }

    # ---------- Tile 管理 ----------

    def add_tile(self, tile: Tile) -> None:
        """
        添加一个瓦片到当前层。
        如果 (row_index, col_index) 已存在，会覆盖旧的 Tile。
        同时增量更新缺失缓存。
        """
        key = (tile.row_index, tile.col_index)
        self.tiles[key] = tile

        # 无固定网格：只关心 image_missing，不管 position_missing
        if self.num_rows <= 0 or self.num_cols <= 0:
            if tile.meta.image_uid is None:
                # 只有图像缺失才记录
                self._missing[key] = {
                    "row_index": tile.row_index,
                    "col_index": tile.col_index,
                    "position_missing": False,
                    "image_missing": True,
                }
            else:
                # 有图像就不缺
                self._missing.pop(key, None)
            return

        # 有固定网格（40x 层）：根据 image_uid 更新缺失表
        if tile.meta.image_uid is None:
            # 位置有 Tile，所以 position_missing=False，图缺失
            self._missing[key] = {
                "row_index": tile.row_index,
                "col_index": tile.col_index,
                "position_missing": False,
                "image_missing": True,
            }
        else:
            # 位置有 Tile，图像也有了，这个格子就不算缺失
            self._missing.pop(key, None)

    def get_tile(self, row: int, col: int) -> Optional[Tile]:
        return self.tiles.get((row, col))

    def iter_tiles(self) -> List[Tile]:
        return list(self.tiles.values())

    # ---------- 缺失瓦片检查 ----------

    def check_missing_tiles(self) -> List[dict]:
        """
        直接返回当前记录的缺失格子列表（只存缺失的）
        复杂度 O(K)，K 为缺失数，而不是 num_rows * num_cols。
        """
        # 注意：对于 40x，如果你在 __post_init__ 初始化了所有 missing，
        # 那么即使一个都没上传过，这里也会返回全网格的缺失。
        return list(self._missing.values())

    # ---------- ROI 细胞查询 ----------

    def iter_cells_in_roi(
            self,
            roi: Tuple[int, int, int, int],
    ) -> list[dict]:
        xmin, ymin, xmax, ymax = roi
        result: List[Dict] = []
        for tile in self.tiles.values():
            for cell in tile.cells:
                if not (cell.x_max < xmin or cell.x_min > xmax or
                        cell.y_max < ymin or cell.y_min > ymax):
                    result.append({"cell_xmin": cell.x_min, "cell_ymin": cell.y_min,
                                   "cell_xmax": cell.x_max, "cell_ymax": cell.y_max,
                                   "cell_type": cell.cell_type,
                                   "cell_type_name": cell.cell_type_name,
                                   "class_confidence": float(cell.class_confidence),
                                   "bbox_confidence": float(cell.bbox_confidence)})

        return result
