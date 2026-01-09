from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np




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
    w: int
    h: int
    x: Optional[int] = None
    y: Optional[int] = None
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



@dataclass(frozen=True)
class ScoreRegion:
    """
    一个 tile 内部的“局部区域评分”（局部坐标系：以 tile 左上角为原点）
    - lx, ly, w, h: 区域在 tile 内的局部坐标与尺寸
    - score: 分值
    """
    lx: float
    ly: float
    w: float
    h: float
    score: float



@dataclass(frozen=True)
class Rect:
    """矩形数据结构"""
    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h
    


@dataclass(frozen=True)
class SelectionResult:
    """选区结果数据结构"""
    area_score: float             # 选区平均分（含空间惩罚）
    cell_count: int               # 选区内细胞总数
    angle: int                    # 旋转角度
    center_grid: Tuple[int, int]  # 网格中心坐标 (x, y)
    rect_size_grid: Tuple[int, int] # 窗口网格尺寸 (w, h)
    vertices_grid: np.ndarray     # 网格坐标系下的四个顶点 (4, 2)


@dataclass
class CellOutput:
    """任务内单个细胞的坐标"""
    cell_xmin: int
    cell_ymin: int
    cell_xmax: int
    cell_ymax: int



@dataclass
class TaskOutput:
    """单场百倍视野拍摄任务"""
    task_index: int
    view_type: str  # 取值范围: "WBC", "MEG"
    view_xmin: int
    view_ymin: int
    view_xmax: int
    view_ymax: int
    region_name: str  # 新增字段：记录属于哪个拍摄区域框（如：初始拍摄框、补拍1...）
    cell_list: List[CellOutput] = field(default_factory=list)