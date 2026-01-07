# cells.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Cell:
    """
    表示一个细胞检测结果（全局坐标 + 分类信息）
    对应接口字段：
    - cell_xmin / cell_ymin / cell_xmax / cell_ymax
    - cell_type / cell_type_name
    - class_confidence / bbox_confidence
    """
    cell_xmin: int
    cell_ymin: int
    cell_xmax: int
    cell_ymax: int
    cell_type: int
    cell_type_name: str
    class_confidence: float = 1.0
    bbox_confidence: float = 1.0

    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cell_xmin": self.cell_xmin,
            "cell_ymin": self.cell_ymin,
            "cell_xmax": self.cell_xmax,
            "cell_ymax": self.cell_ymax,
            "cell_type": self.cell_type,
            "cell_type_name": self.cell_type_name,
            "class_confidence": self.class_confidence,
            "bbox_confidence": self.bbox_confidence,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Cell":
        return cls(**data)
