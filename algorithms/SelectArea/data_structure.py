from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np



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