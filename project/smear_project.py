# project.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import uuid
import os
import json
import pickle  # 可选：如果你要用 pickle 快速持久化整对象

from cells import MagnificationLevel, Cell
from tiles import Tile, TileMeta
from layers import Layer


# =========================
# 任务状态 & 返回码 & 采样任务
# =========================

class TaskStatus(Enum):
    """
    任务状态（对应文档中的 100/101/102 等状态码）
    """
    COMPLETED = 100  # 任务已完成
    RUNNING = 101  # 任务进行中
    TIMEOUT = 102  # 任务超时


@dataclass
class SamplingTask:
    """
    表示一个 100x 拍摄任务（骨髓选区）
    对应 /api/v1/smear_analysis/generate_100x_task_list_bm 的 task_list.
    """
    index: int
    view_type: str  # "WBC" / "MEG"
    magnification: MagnificationLevel

    view_xmin: int
    view_ymin: int
    view_xmax: int
    view_ymax: int

    cells: List[Cell] = field(default_factory=list)

    def as_api_dict(self) -> dict:
        """
        转换成接口返回格式。
        """
        return {
            "task_index": self.index,
            "view_type": self.view_type,
            "view_xmin": self.view_xmin,
            "view_ymin": self.view_ymin,
            "view_xmax": self.view_xmax,
            "view_ymax": self.view_ymax,
            "cell_list": [cell.as_api_dict() for cell in self.cells],
        }


# =========================
# 核心 Project 类
# =========================

class SmearProject:
    """
    表示一次完整的涂片扫描项目：
    - 支持 40x 平扫层
    - 支持 100x 采样层
    - 存储瓦片级与细胞级信息

    和接口的对应关系：
    - /create_task  -> from_create_task_request()
    - /upload_tile  -> add_tile()
    - /check_missing_tiles -> check_missing_tiles()
    - /get_task_result -> get_cells_in_roi()
    - /generate_100x_task_list_bm -> generate_100x_task_list_bm()
    """

    def __init__(
            self,
            task_id: str,
            smear_type: str,
            dpi: int,
            num_rows: int,
            num_cols: int,
            tile_width: int,
            tile_height: int,
    ) -> None:
        self.task_id = task_id
        self.smear_type = smear_type
        self.dpi = dpi
        self.tile_width = tile_width
        self.tile_height = tile_height

        self._task_status: TaskStatus = TaskStatus.RUNNING

        # 各个倍率层，通常至少包含一个 X40 层；
        # 100x 采样图可用单独层保存。
        self.layers: Dict[MagnificationLevel, Layer] = {}

        # 默认创建 40x 层
        self.layers[MagnificationLevel.X40] = Layer(
            name="40x_overview",
            magnification=MagnificationLevel.X40,
            num_rows=num_rows,
            num_cols=num_cols,
            tile_width=tile_width,
            tile_height=tile_height,
        )

    # ---------- 任务状态 ----------

    @property
    def task_status(self) -> TaskStatus:
        return self._task_status

    def set_task_status(self, status: TaskStatus) -> None:
        self._task_status = status

    # ---------- 图层管理 ----------

    def add_layer(self, layer: Layer) -> None:
        """添加一个倍率层，例如新建 100x 采样层。"""
        self.layers[layer.magnification] = layer

    def get_layer(self, magnification: MagnificationLevel) -> Optional[Layer]:
        """获取指定倍率层"""
        return self.layers.get(magnification)

    # ---------- 创建任务（从 HTTP 请求）----------

    @classmethod
    def from_create_task_request(cls, payload: dict) -> "SmearProject":
        """
        根据 /api/v1/smear_analysis/create_task 的请求参数创建 Project。

        请求参数示例：
        {
            "smear_type": "BM",
            "dpi": 300,
            "num_rows": 10,
            "num_cols": 20,
            "tile_width": 2448,
            "tile_height": 2048
        }
        """
        task_id = payload.get("task_id") or uuid.uuid4().hex
        smear_type = payload["smear_type"]
        dpi = int(payload["dpi"])
        num_rows = int(payload["num_rows"])
        num_cols = int(payload["num_cols"])
        tile_width = int(payload["tile_width"])
        tile_height = int(payload["tile_height"])
        return cls(
            task_id=task_id,
            smear_type=smear_type,
            dpi=dpi,
            num_rows=num_rows,
            num_cols=num_cols,
            tile_width=tile_width,
            tile_height=tile_height,
        )

    # ---------- 瓦片管理 ----------

    def add_tile(
            self,
            magnification: MagnificationLevel,
            row_index: int,
            col_index: int,
            position_x: int,
            position_y: int,
            image_uid: Optional[str] = None,
            extra_meta: Optional[dict] = None,
    ) -> Tile:
        """
        添加一个瓦片（对应 upload_tile）

        :param magnification: 倍率（X40 / X100）
        :param row_index: 瓦片行索引
        :param col_index: 瓦片列索引
        :param position_x: 全局左上角 X 坐标
        :param position_y: 全局左上角 Y 坐标
        :param image_uid: 图像唯一 ID，用于追踪
        :param extra_meta: 其它瓦片级别信息
        """
        layer = self.layers.get(magnification)
        if layer is None:
            # 如果对应层不存在，自动创建（方便 100x 层自动建）
            layer = Layer(
                name=f"{str(magnification)}_layer",
                magnification=magnification,
                num_rows=0,  # 若采样层不一定是完整网格，可设置为 0 或按需设置
                num_cols=0,
                tile_width=self.tile_width,
                tile_height=self.tile_height,
            )
            self.layers[magnification] = layer

        meta = TileMeta(
            image_uid=image_uid,
            extra=extra_meta or {},
        )
        tile = Tile(
            magnification=magnification,
            row_index=row_index,
            col_index=col_index,
            global_x=position_x,
            global_y=position_y,
            width=layer.tile_width,
            height=layer.tile_height,
            meta=meta,
        )
        # 添加瓦片到图层
        layer.add_tile(tile)
        return tile

    def upload_complete(self):
        # self.save_pickle('../backend/uploads')
        pass

    def check_missing_tiles(
            self,
            magnification: MagnificationLevel = MagnificationLevel.X40,
    ) -> List[dict]:
        """
        检查某个倍率层的缺失瓦片。

        用于 /api/v1/smear_analysis/check_missing_tiles.
        """
        layer = self.layers.get(magnification)
        if layer is None:
            return []
        return layer.check_missing_tiles()

    # ---------- 细胞管理 ----------

    def add_cells_to_tile(
            self,
            magnification: MagnificationLevel,
            row_index: int,
            col_index: int,
            cells: List[Cell],
    ) -> None:
        """
        将一批细胞添加到指定瓦片中
        """
        layer = self.layers.get(magnification)
        if layer is None:
            raise ValueError(f"Layer for magnification {magnification} not found.")

        tile = layer.get_tile(row_index, col_index)
        if tile is None:
            raise ValueError(f"Tile at ({row_index}, {col_index}) not found in {magnification} layer.")

        for cell in cells:
            tile.add_cell(cell)

    def get_cells_in_roi(
            self,
            magnification: MagnificationLevel,
            roi_xmin: int = 0,
            roi_ymin: int = 0,
            roi_xmax: Optional[int] = None,
            roi_ymax: Optional[int] = None,
    ) -> List[Cell]:
        """
        获取指定倍率层内，在 ROI 区域中的所有细胞。

        对应 /api/v1/smear_analysis/get_task_result 的 roi_xmin 等参数。
        """
        layer = self.layers.get(magnification)
        if layer is None:
            return []

        # 如果 ROI 边界为空，使用一个极大范围
        if roi_xmax is None:
            roi_xmax = 1 << 31
        if roi_ymax is None:
            roi_ymax = 1 << 31

        roi = (roi_xmin, roi_ymin, roi_xmax, roi_ymax)
        return layer.iter_cells_in_roi(roi)

    # ---------- 生成 100x 拍摄任务 (骨髓) ----------

    def generate_100x_task_list_bm(
            self,
            user_choice_area: dict,
            target_num_WBC: int,
            target_num_MEG: int,
            index_offset: int = 0,
            request_task_num: int = 100,
    ) -> List[SamplingTask]:
        """
        根据 40x 层的全局细胞分布，生成 100x 拍摄任务列表（骨髓）。

        注：此处只给出一个简单占位逻辑，实际项目中可在此处
        调用你现有的 select_and_generate_bestArea_capture_tasks 等算法。

        user_choice_area:
        {
            "x_min": int,
            "y_min": int,
            "x_max": int,
            "y_max": int
        }
        """
        # 简单示例：直接把 user_choice_area 当作一个拍摄区域
        xmin = user_choice_area["x_min"]
        ymin = user_choice_area["y_min"]
        xmax = user_choice_area["x_max"]
        ymax = user_choice_area["y_max"]

        # 获取 40x 层在该区域内的所有细胞
        cells_in_roi = self.get_cells_in_roi(
            magnification=MagnificationLevel.X40,
            roi_xmin=xmin,
            roi_ymin=ymin,
            roi_xmax=xmax,
            roi_ymax=ymax,
        )

        # 简化：只生成一个任务（实际可按目标数量切分多块视野）
        main_task = SamplingTask(
            index=0,
            view_type="WBC",  # 简化，实际可根据目标类型拆分
            magnification=MagnificationLevel.X100,
            view_xmin=xmin,
            view_ymin=ymin,
            view_xmax=xmax,
            view_ymax=ymax,
            cells=cells_in_roi,
        )

        # index_offset 和 request_task_num 在复杂策略下会用到，这里简单切片
        tasks = [main_task]
        tasks = tasks[index_offset:index_offset + request_task_num]
        return tasks

    def save_pickle(self, root_dir: str) -> str:
        """
        使用 pickle 持久化（不推荐跨语言，但简单粗暴）。
        文件名: <root_dir>/<task_id>.smear.pkl
        """
        import os
        import pickle

        os.makedirs(root_dir, exist_ok=True)
        path = os.path.join(root_dir, f"{self.task_id}.smear.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load_pickle(cls, task_id: str, root_dir: str) -> "SmearProject":
        import os
        import pickle

        path = os.path.join(root_dir, f"{task_id}.smear.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj
