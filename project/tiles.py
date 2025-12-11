# tiles.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

from cells import MagnificationLevel, Cell


@dataclass
class TileMeta:
    """
    瓦片级别的附加信息（可扩展）

    举例：
    - image_uid: 图像唯一 ID（接口里有）
    - quality_score: 质量分数
    - stain_type: 染色类型
    - extra: 任何自定义字段
    """
    image_uid: Optional[str] = None
    quality_score: Optional[float] = None
    stain_type: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass
class Tile:
    """
    表示某一倍率层上的一个瓦片（拼图块）

    对应接口 /api/v1/smear_analysis/upload_tile:
    - row_index / col_index
    - position_x / position_y
    """
    magnification: MagnificationLevel
    row_index: int
    col_index: int

    # 瓦片在全局坐标中的左上角位置
    global_x: int
    global_y: int

    width: int
    height: int

    meta: TileMeta = field(default_factory=TileMeta)

    # 当前瓦片上的所有细胞（坐标已经是全局坐标）
    cells: List[Cell] = field(default_factory=list)

    def add_cell(self, cell: Cell) -> None:
        """
        向当前瓦片添加一个细胞（cell 坐标必须为全局坐标）。
        """
        self.cells.append(cell)

    def as_api_dict(self) -> dict:
        """
        转换为通用接口字段，便于未来需要返回瓦片信息时使用。
        """
        return {
            "row_index": self.row_index,
            "col_index": self.col_index,
            "position_x": self.global_x,
            "position_y": self.global_y,
            "tile_width": self.width,
            "tile_height": self.height,
            "image_uid": self.meta.image_uid,
        }
