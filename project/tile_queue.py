# tile_queue.py
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, List


@dataclass(frozen=True)
class TileMsg:
    task_id: str
    row_index: int
    col_index: int
    tile_bytes: bytes
    tile_meta: Any = None


class TileTaskWrap:
    """
    单个 task 的瓦片队列处理器：
    - num_workers 个线程并发消费
    - finish() 投放 num_workers 个 None 哨兵，保证每个线程退出
    - get_timeout_sec 超时后触发 cleanup()（幂等）
    - task_done() 仅在 worker finally 调用，避免重复 task_done
    """

    def __init__(
            self,
            task_id: str,
            on_tile_callback: Callable[[TileMsg], None],
            on_finish_callback: Optional[Callable[[str], None]] = None,
            num_workers: int = 8,
            queue_maxsize: int = 0,
            get_timeout_sec: int = 300,
    ):
        self.task_id = task_id
        self.on_tile_callback = on_tile_callback
        self.on_finish_callback = on_finish_callback

        self.tile_queue: "queue.Queue[Optional[TileMsg]]" = queue.Queue(maxsize=queue_maxsize)

        self.num_workers = max(1, int(num_workers))
        self.get_timeout_sec = int(get_timeout_sec)

        self._stop_event = threading.Event()
        self._cleanup_once_lock = threading.Lock()
        self._cleaned_up = False

        self.finished = False
        self.is_timeout = False

        self._workers: List[threading.Thread] = []
        for i in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"tile-worker-{task_id}-{i + 1}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)

    def put_tile(self, row_index: int, col_index: int, tile_bytes: bytes, tile_meta: Any = None) -> None:
        if self._stop_event.is_set() or self._cleaned_up:
            raise RuntimeError(f"TileTaskWrap[{self.task_id}] stopping/cleaned; cannot accept new tiles.")
        self.tile_queue.put(TileMsg(self.task_id, row_index, col_index, tile_bytes, tile_meta))

    def finish(self) -> None:
        # 投放哨兵：每个 worker 都能退出
        for _ in range(self.num_workers):
            self.tile_queue.put(None)

    def join(self) -> None:
        self.tile_queue.join()

    def cleanup(self) -> None:
        # 幂等 cleanup：只执行一次
        with self._cleanup_once_lock:
            if self._cleaned_up:
                return
            self._cleaned_up = True
            self._stop_event.set()

            # 唤醒阻塞 worker：尽力 put_nowait(None)
            for _ in range(self.num_workers):
                try:
                    self.tile_queue.put_nowait(None)
                except queue.Full:
                    pass

            # 等待线程退出（短超时避免卡死）
            for t in self._workers:
                if t.is_alive():
                    t.join(timeout=2.0)

            self.finished = True
            if self.on_finish_callback:
                try:
                    self.on_finish_callback(self.task_id)
                except Exception:
                    pass

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self.tile_queue.get(timeout=self.get_timeout_sec)
            except Exception:
                # 超时无任务：触发一次 cleanup（幂等）
                self.is_timeout = True
                self.cleanup()
                break

            try:
                if item is None:
                    break

                if self._stop_event.is_set():
                    continue

                # 核心：将瓦片交给上层回调（这里做 decode/submit/add_tile 都行）
                self.on_tile_callback(item)

            except Exception as e:
                print(f"[{threading.current_thread().name}] Error processing tile: {e!r}")
            finally:
                try:
                    self.tile_queue.task_done()
                except ValueError as ve:
                    print(f"[{threading.current_thread().name}] task_done mismatch: {ve!r}")

        return


class TileQueueRouter:
    """
    全局路由器：
    - create_task(task_id, ...)
    - push_tile(task_id, ...)
    - finish_task(task_id)
    - join_task(task_id)
    - cleanup_task(task_id)
    """

    def __init__(self):
        self._tasks: Dict[str, TileTaskWrap] = {}
        self._lock = threading.Lock()

    def create_task(
            self,
            task_id: str,
            on_tile_callback: Callable[[TileMsg], None],
            on_finish_callback: Optional[Callable[[str], None]] = None,
            num_workers: int = 8,
            queue_maxsize: int = 0,
            get_timeout_sec: int = 300,
            overwrite: bool = False,
    ) -> TileTaskWrap:
        with self._lock:
            if (task_id in self._tasks) and (not overwrite):
                return self._tasks[task_id]
            if task_id in self._tasks:
                # 覆盖时先清理旧的
                try:
                    self._tasks[task_id].cleanup()
                except Exception:
                    pass

            wrap = TileTaskWrap(
                task_id=task_id,
                on_tile_callback=on_tile_callback,
                on_finish_callback=on_finish_callback,
                num_workers=num_workers,
                queue_maxsize=queue_maxsize,
                get_timeout_sec=get_timeout_sec,
            )
            self._tasks[task_id] = wrap
            return wrap

    def push_tile(self, task_id: str, row_index: int, col_index: int, tile_bytes: bytes, tile_meta: Any = None) -> None:
        with self._lock:
            wrap = self._tasks.get(task_id)
        if wrap is None:
            raise KeyError(f"Task '{task_id}' not found. Call create_task() first.")
        wrap.put_tile(row_index, col_index, tile_bytes, tile_meta)

    def finish_task(self, task_id: str) -> None:
        with self._lock:
            wrap = self._tasks.get(task_id)
        if wrap:
            wrap.finish()

    def join_task(self, task_id: str) -> None:
        with self._lock:
            wrap = self._tasks.get(task_id)
        if wrap:
            wrap.join()

    def cleanup_task(self, task_id: str, remove: bool = True) -> None:
        with self._lock:
            wrap = self._tasks.get(task_id)
        if wrap:
            wrap.cleanup()
        if remove:
            with self._lock:
                self._tasks.pop(task_id, None)

    def cleanup_all(self) -> None:
        with self._lock:
            task_ids = list(self._tasks.keys())
        for tid in task_ids:
            self.cleanup_task(tid, remove=True)
