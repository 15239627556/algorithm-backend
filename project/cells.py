# cells.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MagnificationLevel(Enum):
    """
    表示不同的倍率层级，例如 40x 全扫图、100x 采样图。
    """
    X40 = "40x"
    X100 = "100x"

    def __str__(self) -> str:
        return self.value


@dataclass
class Cell:
    """
    表示一个细胞检测结果（全局坐标 + 分类信息）

    对应接口字段：
    - cell_xmin / cell_ymin / cell_xmax / cell_ymax
    - cell_type / cell_type_name
    - class_confidence / bbox_confidence
    """
    id: str
    magnification: MagnificationLevel
    layer_name: str
    tile_row: int
    tile_col: int

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    cell_type_id: int
    cell_type_name: str
    class_confidence: float
    bbox_confidence: float

    extra: dict = field(default_factory=dict)

    def as_api_dict(self) -> dict:
        """
        转换为接口返回格式的字典，用于 /get_task_result 等接口。
        """
        return {
            "cell_xmin": self.x_min,
            "cell_ymin": self.y_min,
            "cell_xmax": self.x_max,
            "cell_ymax": self.y_max,
            "cell_type": self.cell_type_id,
            "cell_type_name": self.cell_type_name,
            "class_confidence": float(self.class_confidence),
            "bbox_confidence": float(self.bbox_confidence),
        }
