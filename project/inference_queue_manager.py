# inference_queue_manager.py
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Deque, Optional, Tuple, List
from collections import deque
import uuid

import numpy as np

from cells import MagnificationLevel, Cell
from project.smear_project import SmearProject
from tiles import Tile

# 你的模型类（40x / 100x）
from algorithms.x40model.X40ImageModels import X40ImageModels


# from algorithms.x100model import X100ImageModels


# dispatcher = X100ImageModels.X100ImageModels(num_workers=1)


# =========================
# 任务 / 适配器 数据结构
# =========================

@dataclass
class TileModelTask:
    """
    表示“一个瓦片的一次模型推理任务”，对应一个模型内部 task_id。

    注意区分：
    - project_task_id: 你自己的大任务 ID（SmearProject.task_id）
    - model_task_id:   模型内部的 task id（enqueue_task 的返回值）
    """
    project_task_id: str
    magnification: MagnificationLevel
    row_index: int
    col_index: int

    model_task_id: object | str | int
    extra: dict = field(default_factory=dict)


@dataclass
class ModelAdapter:
    """
    每个倍率对应一个模型适配器：
    - 持有模型实例 model
    - 知道如何 enqueue_task
    - 知道如何解析 get_result 的输出为 List[Cell]
    """
    magnification: MagnificationLevel
    model: object
    # enqueue_fn(model, image: np.ndarray, **kwargs) -> model_task_id
    enqueue_fn: Callable[[object, np.ndarray], object]
    # parse_result_fn(result, *, project, job, tile) -> List[Cell]
    parse_result_fn: Callable[[object, SmearProject, TileModelTask, Tile], List[Cell]]


# =========================
# 结果解析函数（示例）
# =========================

def parse_result_as_cell_list(
        result: dict,
        project: SmearProject,
        job: TileModelTask,
        tile: Tile,
) -> List[Cell]:
    """
    通用解析函数：
    假设模型返回格式类似 /get_task_result 的结构：

    result = {
        "cell_list": [
            {
                "cell_xmin": int,
                "cell_ymin": int,
                "cell_xmax": int,
                "cell_ymax": int,
                "cell_type": int,
                "cell_type_name": str,
                "class_confidence": float,
                "bbox_confidence": float,
            },
            ...
        ]
    }
    """
    layer_name = project.get_layer(job.magnification).name  # 一定存在

    cells: List[Cell] = []
    for item in result.get("cell_list", []):
        cell = Cell(
            id=uuid.uuid4().hex,
            magnification=job.magnification,
            layer_name=layer_name,
            tile_row=job.row_index,
            tile_col=job.col_index,
            x_min=int(item["cell_xmin"]),
            y_min=int(item["cell_ymin"]),
            x_max=int(item["cell_xmax"]),
            y_max=int(item["cell_ymax"]),
            cell_type_id=int(item["cell_type"]),
            cell_type_name=str(item.get("cell_type_name", item["cell_type"])),
            class_confidence=float(item.get("class_confidence", 1.0)),
            bbox_confidence=float(item.get("bbox_confidence", 1.0)),
        )
        cells.append(cell)
    return cells


def parse_x100_result_from_rects(
        result: dict,
        project: SmearProject,
        job: TileModelTask,
        tile: Tile,
) -> List[Cell]:
    """
    针对 X100ImageModels 的默认解析：
    - result["cellRects"]: N x 4 (x, y, w, h)，这里假设是局部坐标（以 tile 左上角为原点）
    - result["cellTypes"]: N x k，取第一个作为 top1 类型
    - result["cellRatios"]: N x k，对应每种类型的占比（置信度）

    你可以根据实际模型输出调整这里。
    """
    layer_name = project.get_layer(job.magnification).name

    cell_rects = np.asarray(result["cellRects"]).astype(int)
    cell_types = np.asarray(result["cellTypes"]).astype(int)
    cell_ratios = np.asarray(result["cellRatios"]).astype(float)

    # 如果 100x 图像相对于全局坐标有偏移，则可以通过 job.extra 传入
    offset_x = int(job.extra.get("offset_x", tile.global_x))
    offset_y = int(job.extra.get("offset_y", tile.global_y))

    cells: List[Cell] = []
    for rect, types, ratios in zip(cell_rects, cell_types, cell_ratios):
        x, y, w, h = rect.tolist()
        x_min = offset_x + x
        y_min = offset_y + y
        x_max = x_min + w
        y_max = y_min + h

        cell_type_id = int(types[0]) if len(types) > 0 else -1
        class_conf = float(ratios[0]) if len(ratios) > 0 else 1.0

        cell = Cell(
            id=uuid.uuid4().hex,
            magnification=job.magnification,
            layer_name=layer_name,
            tile_row=job.row_index,
            tile_col=job.col_index,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            cell_type_id=cell_type_id,
            cell_type_name=str(cell_type_id),
            class_confidence=class_conf,
            bbox_confidence=1.0,
        )
        cells.append(cell)

    return cells


# =========================
# 队列管理类
# =========================

class TileInferenceQueueManager:
    """
    统一管理不同倍率的模型队列：
    - 负责把瓦片图片丢给对应模型（enqueue_task）
    - 轮询 get_result
    - 把结果解析为 Cell 列表，写回 SmearProject 对应瓦片

    用法示例：

    manager = TileInferenceQueueManager(project_registry)
    manager.register_default_x40_model()
    manager.register_default_x100_model()

    # 某个地方上传了瓦片图像（np.ndarray）
    manager.submit_tile(
        project_task_id=task_id,
        magnification=MagnificationLevel.X40,
        row_index=row,
        col_index=col,
        image=image_ndarray,
    )
    """

    def __init__(
            self,
            project_registry: Optional[Dict[str, SmearProject]] = None,
            poll_interval: float = 0.001,
    ) -> None:
        """
        :param project_registry: 可选的 {task_id: SmearProject} 初始字典。
                                 如果传进来，就直接引用这份 dict；
                                 如果不传，就内部自己维护一份。
        """
        # 如果外面给了一份 dict，我们就直接引用（不是 copy）
        self._projects: Dict[str, SmearProject] = project_registry if project_registry is not None else {}

        self._poll_interval = poll_interval

        self._adapters: Dict[MagnificationLevel, ModelAdapter] = {}

        self._queue: Deque[TileModelTask] = deque()
        self._queue_lock = threading.Lock()
        self._cond = threading.Condition()
        self._stop_flag = False

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self.register_default_x40_model(1)
        print(self._adapters.get(MagnificationLevel.X40).model)
        # self.register_default_x100_model(1)

    # ---------- Project 管理 ----------

    def register_project(self, project: SmearProject) -> None:
        """
        注册一个新的 SmearProject，使队列管理器后续可以根据 task_id 找到它。

        一般在 create_task 的时候调用：
        - project = SmearProject(...)
        - manager.register_project(project)
        """
        self._projects[project.task_id] = project

    def unregister_project(self, task_id: str) -> None:
        """
        当任务完成且不再需要时，可以把它从管理器中移除。
        """
        self._projects.pop(task_id, None)

    def get_project(self, task_id: str) -> Optional[SmearProject]:
        """
        方便外部访问管理器内部维护的项目。
        """
        return self._projects.get(task_id)

    # ---------- 模型注册 ----------

    def register_model_adapter(self, adapter: ModelAdapter) -> None:
        """
        注册一个倍率对应的模型适配器。
        """
        self._adapters[adapter.magnification] = adapter

    def register_default_x40_model(self, num_workers: int = 8) -> None:
        """
        方便直接使用默认的 40x 模型。
        """
        model = X40ImageModels(num_workers)

        def enqueue_fn(m, image: np.ndarray):
            # 这里假定 X40ImageModels.enqueue_task(image) 签名为这样
            return m.enqueue_task(image)

        adapter = ModelAdapter(
            magnification=MagnificationLevel.X40,
            model=model,
            enqueue_fn=enqueue_fn,
            # 这里假定 40x 模型也返回类似 {"cell_list": [...]} 的结构
            parse_result_fn=parse_result_as_cell_list,
        )
        self.register_model_adapter(adapter)

    def register_default_x100_model(self, num_workers: int = 1) -> None:
        """
        方便直接使用默认的 100x 模型。
        """
        # model = X100ImageModels.X100ImageModels(num_workers=num_workers)
        #
        # def enqueue_fn(m, image: np.ndarray):
        #     # 你当前的代码是: enqueue_task(image, cell_target_type)
        #     # 为了通用，这里只传 image。需要 cell_target_type 的话，可以通过 job.extra 再封装一个适配器。
        #     return m.enqueue_task(image)
        #
        # adapter = ModelAdapter(
        #     magnification=MagnificationLevel.X100,
        #     model=model,
        #     enqueue_fn=enqueue_fn,
        #     parse_result_fn=parse_x100_result_from_rects,
        # )
        # self.register_model_adapter(adapter)

    # ---------- 提交任务 ----------

    def submit_tile(
            self,
            project_task_id: str,
            magnification: MagnificationLevel,
            row_index: int,
            col_index: int,
            image: np.ndarray,
            extra: Optional[dict] = None,
    ) -> object:
        """
        提交一个瓦片推理任务。

        :param project_task_id: SmearProject.task_id
        :param magnification: 倍率（40x / 100x / 未来更多）
        :param row_index: 瓦片行号
        :param col_index: 瓦片列号
        :param image: 瓦片图像（numpy ndarray，BGR 或 RGB 由模型自己决定）
        :param extra: 额外信息（比如 100x 图像在全局坐标上的偏移等）
        :return: 模型内部的 task_id
        """
        adapter = self._adapters.get(magnification)
        print(adapter.model)
        if adapter is None:
            raise RuntimeError(f"No model adapter registered for magnification={magnification}")

        print('Submitting tile to model:', magnification, row_index, col_index)
        model_task_id = adapter.enqueue_fn(adapter.model, image)
        print('Model task id:', model_task_id)
        job = TileModelTask(
            project_task_id=project_task_id,
            magnification=magnification,
            row_index=row_index,
            col_index=col_index,
            model_task_id=model_task_id,
            extra=extra or {},
        )

        with self._queue_lock:
            self._queue.append(job)
            with self._cond:
                self._cond.notify()
        print('Model task id123:', model_task_id)
        print(self._queue.__len__(), 'tasks in queue')
        return model_task_id

    # ---------- 停止 ----------

    def stop(self) -> None:
        """
        结束后台线程（服务关闭时调用）。
        """
        self._stop_flag = True
        with self._cond:
            self._cond.notify_all()
        if self._worker.is_alive():
            self._worker.join()

    # ---------- 工作线程 ----------

    def _worker_loop(self) -> None:
        """
        从队列中取出任务，轮询模型结果，并写入 SmearProject。
        """
        while True:
            with self._cond:
                while not self._queue and not self._stop_flag:
                    self._cond.wait()
                if self._stop_flag and not self._queue:
                    return

            with self._queue_lock:
                if not self._queue:
                    continue
                job = self._queue.popleft()

            adapter = self._adapters.get(job.magnification)
            if adapter is None:
                # 没找到对应模型，忽略这个任务
                continue

            # 拿模型的结果
            result = adapter.model.get_result(job.model_task_id)
            if not result:
                # 结果尚未准备好，重新放回队列稍后再试
                with self._queue_lock:
                    self._queue.append(job)
                time.sleep(self._poll_interval)
                continue

            # 找到对应的项目和瓦片
            project = self.get_project(job.project_task_id)
            if project is None:
                # 对应项目已经被释放 / 不存在
                continue

            layer = project.get_layer(job.magnification)
            if layer is None:
                continue

            tile = layer.get_tile(job.row_index, job.col_index)
            if tile is None:
                continue

            # 解析模型结果 -> List[Cell]
            try:
                cells = adapter.parse_result_fn(result, project, job, tile)
            except Exception as e:
                # 可以在这里加入日志记录
                continue

            if cells:
                project.add_cells_to_tile(
                    magnification=job.magnification,
                    row_index=job.row_index,
                    col_index=job.col_index,
                    cells=cells,
                )
