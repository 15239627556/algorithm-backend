# inference_queue_manager.py
from __future__ import annotations

import multiprocessing as mp
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional
from collections import deque

import numpy as np
import cv2

from cells import MagnificationLevel, Cell
from project.smear_project import SmearProject
from tiles import Tile


@dataclass
class TileModelTask:
    project_task_id: str
    magnification: MagnificationLevel
    row_index: int
    col_index: int

    tile_bytes: bytes
    tile_meta: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


@dataclass
class ModelAdapter:
    magnification: MagnificationLevel
    parse_result_fn: Callable[[dict, SmearProject, "TileModelTask", Tile], List[Cell]]


def parse_result_as_cell_list(result: dict, project: SmearProject, job: TileModelTask, tile: Tile) -> List[Cell]:
    layer_name = project.get_layer(job.magnification).name

    cells: List[Cell] = []
    for item in result.get("haveCellCenterPoints", []):
        cells.append(
            Cell(
                id=uuid.uuid4().hex,
                magnification=job.magnification,
                layer_name=layer_name,
                tile_row=job.row_index,
                tile_col=job.col_index,
                x_min=int(item[0]),
                y_min=int(item[1]),
                x_max=int(item[2]),
                y_max=int(item[3]),
                cell_type=0,
                cell_type_name="有核细胞",
                class_confidence=float(item[4]),
                bbox_confidence=float(1.0),
            )
        )

    for item in result.get("bigCellRects", []):
        cells.append(
            Cell(
                id=uuid.uuid4().hex,
                magnification=job.magnification,
                layer_name=layer_name,
                tile_row=job.row_index,
                tile_col=job.col_index,
                x_min=int(item[0]),
                y_min=int(item[1]),
                x_max=int(item[2]),
                y_max=int(item[3]),
                cell_type=1,
                cell_type_name="巨核细胞",
                class_confidence=float(item[4]),
                bbox_confidence=float(1.0),
            )
        )
    return cells


def _inference_process_main(
        in_q: "mp.Queue[dict]",
        out_q: "mp.Queue[dict]",
        ready_evt: "mp.synchronize.Event",
        num_workers: int,
        poll_interval: float,
) -> None:
    import os
    print("### INFERENCE PROCESS PID =", os.getpid())

    # Import ONLY inside inference process to avoid touching TRT/CUDA in Flask process.
    from algorithms.x40model.X40ImageModels import X40ImageModels

    model = X40ImageModels(num_workers)

    try:
        ready_evt.set()
    except Exception:
        pass

    pending: Deque[tuple[Any, TileModelTask]] = deque()

    running = True
    grids_sub = np.full((23, 25), False, dtype=bool)
    grids_res = np.full((23, 25), False, dtype=bool)
    while running:
        try:
            msg = in_q.get(timeout=poll_interval)
        except Exception:
            msg = None
        if msg is not None:
            mtype = msg.get("type")
            if mtype == "SUBMIT":
                job: TileModelTask = msg["job"]
                grids_sub[job.row_index, job.col_index] = True
                img = cv2.imdecode(np.frombuffer(job.tile_bytes, np.uint8), cv2.IMREAD_COLOR)
                # if img is None:
                #     out_q.put({"type": "ERROR", "project_task_id": job.project_task_id,
                #                "row_index": job.row_index, "col_index": job.col_index,
                #                "magnification": job.magnification, "error": "cv2.imdecode failed"})
                # else:
                img = np.ascontiguousarray(img)
                model_task_id = model.enqueue_task(img)
                pending.append((model_task_id, job))

            elif mtype == "SYNC":
                try:
                    print('我进来了======')
                    model.synchronize()
                    print('我出来了======')
                except Exception as e:
                    out_q.put({"type": "ERROR", "error": repr(e), "where": "SYNC"})

            elif mtype == "STOP":
                running = False

        if pending:
            model_task_id, job = pending.popleft()
            try:
                result = model.get_result(model_task_id)
            except Exception as e:
                out_q.put({"type": "ERROR", "project_task_id": job.project_task_id,
                           "row_index": job.row_index, "col_index": job.col_index,
                           "magnification": job.magnification, "error": repr(e)})
                continue

            if result:
                grids_res[job.row_index, job.col_index] = True
                out_q.put(
                    {
                        "type": "RESULT",
                        "project_task_id": job.project_task_id,
                        "magnification": job.magnification,
                        "row_index": job.row_index,
                        "col_index": job.col_index,
                        "result": result,
                        "tile_meta": job.tile_meta,
                        "extra": job.extra,
                    }
                )
            else:
                pending.append((model_task_id, job))
        # rows, cols = np.where(grids_sub == False)
        # coords = list(zip(rows, cols))
        # print('grids_sub:', coords[:2])
        # rows, cols = np.where(grids_res == False)
        # coords = list(zip(rows, cols))
        # print('grids_res:', coords[:2])


class TileInferenceQueueManager:
    def __init__(
            self,
            project_registry: Optional[Dict[str, SmearProject]] = None,
            poll_interval: float = 0.001,
            model_num_workers: int = 1,
            ready_timeout_sec: float = 300.0,
            mp_ctx: Optional[mp.context.BaseContext] = None,
    ) -> None:
        self._projects: Dict[str, SmearProject] = project_registry if project_registry is not None else {}
        self._poll_interval = float(poll_interval)

        self._adapters: Dict[MagnificationLevel, ModelAdapter] = {}
        self.register_default_x40_adapter()

        self._mp = mp_ctx if mp_ctx is not None else mp.get_context("spawn")
        self._in_q: "mp.Queue[dict]" = self._mp.Queue(maxsize=512)
        self._out_q: "mp.Queue[dict]" = self._mp.Queue(maxsize=512)
        self._ready_evt = self._mp.Event()

        self._proc = self._mp.Process(
            target=_inference_process_main,
            args=(self._in_q, self._out_q, self._ready_evt, int(model_num_workers), self._poll_interval),
            daemon=True,
        )
        self._proc.start()

        if not self._ready_evt.wait(timeout=float(ready_timeout_sec)):
            raise TimeoutError(f"Inference process READY timeout after {ready_timeout_sec}s")

        self._stop_flag = False
        self._result_thread = threading.Thread(target=self._result_loop, daemon=True, name="inference-result-loop")
        self._result_thread.start()

        # ---- task progress tracking (main process) ----
        self._expected: Dict[str, int] = {}  # task_id -> expected tiles
        self._submitted: Dict[str, int] = {}  # task_id -> submitted tiles
        self._written: Dict[str, int] = {}  # task_id -> tiles written back to project (RESULT handled)
        self._cv = threading.Condition()
        self.count_result = 0

    def register_project(self, project: SmearProject) -> None:
        self._projects[project.task_id] = project

    def unregister_project(self, task_id: str) -> None:
        self._projects.pop(task_id, None)

    def get_project(self, task_id: str) -> Optional[SmearProject]:
        return self._projects.get(task_id)

    def register_default_x40_adapter(self) -> None:
        self._adapters[MagnificationLevel.X40] = ModelAdapter(
            magnification=MagnificationLevel.X40,
            parse_result_fn=parse_result_as_cell_list,
        )

    def submit_tile_bytes(
            self,
            project_task_id: str,
            magnification: MagnificationLevel,
            row_index: int,
            col_index: int,
            tile_bytes: bytes,
            tile_meta: Optional[dict] = None,
            extra: Optional[dict] = None,
    ) -> None:
        job = TileModelTask(
            project_task_id=project_task_id,
            magnification=magnification,
            row_index=int(row_index),
            col_index=int(col_index),
            tile_bytes=tile_bytes,
            tile_meta=tile_meta or {},
            extra=extra or {},
        )
        self._in_q.put({"type": "SUBMIT", "job": job})
        self.mark_submitted(project_task_id, 1)

    def set_expected_tiles(self, project_task_id: str, expected: int) -> None:
        with self._cv:
            self._expected[project_task_id] = int(expected)
            self._submitted.setdefault(project_task_id, 0)
            self._written.setdefault(project_task_id, 0)
            self._cv.notify_all()

    def mark_submitted(self, project_task_id: str, n: int = 1) -> None:
        with self._cv:
            self._submitted[project_task_id] = self._submitted.get(project_task_id, 0) + int(n)
            self._cv.notify_all()

    def wait_written_all(self, project_task_id: str, timeout: float = 300.0) -> None:
        """
        Wait until written >= expected (if expected set), otherwise wait written >= submitted.
        This guarantees RESULT has been handled (written back into project).
        """
        t0 = time.time()
        with self._cv:
            while True:
                exp = self._expected.get(project_task_id, None)
                sub = self._submitted.get(project_task_id, 0)
                wrt = self._written.get(project_task_id, 0)

                target = exp if exp is not None else sub
                if target > 0 and wrt >= target:
                    return

                if time.time() - t0 >= timeout:
                    raise TimeoutError(
                        f"wait_written_all timeout: task={project_task_id}, written={wrt}, target={target}, submitted={sub}, expected={exp}"
                    )
                self._cv.wait(timeout=0.2)

    def finish_tile(self, project_task_id: str, magnification: MagnificationLevel) -> None:
        self._in_q.put({"type": "SYNC", "project_task_id": project_task_id})

    def stop(self) -> None:
        self._stop_flag = True
        try:
            self._in_q.put({"type": "STOP"})
        except Exception:
            pass
        if self._proc.is_alive():
            self._proc.join(timeout=5.0)

    def _result_loop(self) -> None:
        while not self._stop_flag:
            try:
                msg = self._out_q.get(timeout=0.2)
            except Exception:
                continue
            self._handle_out_msg(msg)

    def _handle_out_msg(self, msg: dict) -> None:
        mtype = msg.get("type")
        if mtype == "RESULT":
            project_task_id = msg["project_task_id"]
            magnification = msg["magnification"]
            row_index = msg["row_index"]
            col_index = msg["col_index"]
            result = msg["result"]

            project = self.get_project(project_task_id)
            if project is None:
                print("[DROP] project None", project_task_id);
                return

            layer = project.get_layer(magnification)
            if layer is None:
                print("[DROP] layer None", project_task_id, magnification);
                return

            tile = layer.get_tile(row_index, col_index)
            if tile is None:
                print("[DROP] tile None", project_task_id, magnification, row_index, col_index);
                return

            adapter = self._adapters.get(magnification)
            if adapter is None:
                print("[DROP] adapter None", magnification, type(magnification));
                return

            job = TileModelTask(
                project_task_id=project_task_id,
                magnification=magnification,
                row_index=row_index,
                col_index=col_index,
                tile_bytes=b"",
                tile_meta=msg.get("tile_meta", {}),
                extra=msg.get("extra", {}),
            )

            try:
                cells = adapter.parse_result_fn(result, project, job, tile)
            except Exception:
                return

            if cells:
                project.add_cells_to_tile(magnification, row_index, col_index, cells)

            with self._cv:
                self._written[project_task_id] = self._written.get(project_task_id, 0) + 1
                self._cv.notify_all()

            self.count_result += 1

            print(row_index, col_index, len(cells))
            print(self.count_result)
            if self.count_result >= 570:
                print(self._expected[project_task_id])
                print(self._submitted[project_task_id])
                print(self._written[project_task_id])


        elif mtype == "ERROR":
            print("[InferenceProcess ERROR]", msg)
        else:
            return
