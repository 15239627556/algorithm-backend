from project.project_computer import X40ProjectComputer
import queue
import threading
import numpy as np
import cv2
import gc
from typing import Optional, Any, Tuple, List


class ProjectWraps(X40ProjectComputer):
    """
    多线程版本：
    - 使用 num_workers 个 worker 并发从队列取图像、解码并调用 add_image
    - finish() 投放与 worker 数量相等的哨兵 None，保证每个线程都能退出
    - 队列长时间无任务（get_timeout_sec）会触发一次 cleanup()（幂等）
    - cleanup() 不与 worker 抢队列，不再执行 get_nowait()+task_done()，避免重复 task_done
    - 可选对 add_image 上锁（当父类非线程安全时）
    """

    def __init__(
            self,
            num_rows: int,
            num_cols: int,
            w: int = 2448,
            h: int = 2048,
            on_cleanup_callback=None,
            num_workers: int = 16,
            add_image_threadsafe: bool = True,  # 若父类 add_image 非线程安全，设为 False 会自动加锁
            queue_maxsize: int = 0,  # 队列上限；0 表示不限制
            get_timeout_sec: int = 300  # worker 从队列取任务的超时时间（秒）
    ):

        super().__init__(num_rows, num_cols, w, h)
        self.grid_new = np.full((num_rows, num_cols), None, dtype=bool)
        # 队列：元素为 (row_index, col_index, image_bytes, image_data) 或 None(哨兵)
        self.image_queue: "queue.Queue[Optional[Tuple[int,int,bytes,Any]]]" = queue.Queue(maxsize=queue_maxsize)

        # 状态
        self.finished = False
        self.is_timeout = False
        self.on_cleanup_callback = on_cleanup_callback

        # 并发控制
        self.num_workers = max(1, int(num_workers))
        self.get_timeout_sec = int(get_timeout_sec)
        self._stop_event = threading.Event()

        # cleanup 幂等控制
        self._cleanup_once_lock = threading.Lock()
        self._cleaned_up = False

        # add_image 线程安全控制
        self._add_lock = None if add_image_threadsafe else threading.Lock()

        # 启动 worker 线程
        self._workers: List[threading.Thread] = []
        for i in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"img-worker-{i + 1}",
                daemon=True
            )
            t.start()
            self._workers.append(t)

    # ========== 对外接口 ==========

    def put_image(self, row_index: int, col_index: int, image_bytes: bytes, image_data: Any):
        """
        投放一张图片任务。image_bytes 为编码后的图像字节（如 JPEG/PNG）。
        """
        # print("Length: ", len(image_bytes), "row=", row_index, "col=", col_index)
        if self._stop_event.is_set() or self._cleaned_up:
            raise RuntimeError("ProjectWraps is stopping/cleaned up; cannot accept new images.")
        self.image_queue.put((row_index, col_index, image_bytes, image_data))

    def finish(self):
        """
        投放 num_workers 个哨兵，表示不会再有新任务；所有 worker 取到哨兵后会安全退出。
        """
        for _ in range(self.num_workers):
            self.image_queue.put(None)
        print('结束任务，等待工作线程退出（已投放哨兵）')

    def image_queue_join(self):
        """等待所有图像处理任务完成（包括哨兵被消费）"""
        self.image_queue.join()
        sub_arr = self.grid_new[:self.num_rows]
        mask = sub_arr == None
        rows, cols = np.where(mask)
        coords = np.column_stack((rows, cols)).tolist()[:100]  # 最多打印前100个
        print('Image queue join completed')
        return [{'row_index': r, 'col_index': c} for r, c in coords]

    # ========== 线程主体 ==========

    def _worker_loop(self):
        """
        每个 worker 的主循环：从队列取任务 -> 解码 -> add_image。
        保证对每个 get() 到的项目（包括哨兵）恰好调用一次 task_done()（统一在 finally）。
        """
        while not self._stop_event.is_set():
            try:
                item = self.image_queue.get(timeout=self.get_timeout_sec)
            except Exception as e:
                # 任一线程在超时无任务时触发一次 cleanup（幂等）
                print(f'[{threading.current_thread().name}] Image queue is empty, triggering cleanup...')
                self.is_timeout = True
                self.cleanup()
                break

            try:
                if item is None:
                    print(f'[{threading.current_thread().name}] Received sentinel; exiting.')
                    break
                if self._stop_event.is_set():
                    continue
                row_index, col_index, image_bytes, image_data = item
                np_arr = np.frombuffer(image_bytes, dtype=np.uint8).copy()
                img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img_np is None:
                    raise ValueError("cv2.imdecode failed to decode image bytes")
                img_np = img_np.copy()

                # 调用父类 add_image（必要时加锁）
                if self._add_lock is None:
                    self.add_image(row_index, col_index, img_np, image_data)
                    self.grid_new[row_index, col_index] = True
                else:
                    with self._add_lock:
                        self.add_image(row_index, col_index, img_np, image_data)
                        self.grid_new[row_index, col_index] = True
                del img_np, np_arr

            except Exception as e:
                print(f'[{threading.current_thread().name}] Error processing image: {e!r}')
            finally:
                try:
                    self.image_queue.task_done()
                except ValueError as ve:
                    print(f'[{threading.current_thread().name}] task_done mismatch: {ve!r}')
                try:
                    del item, row_index, col_index, image_data, image_bytes
                except Exception:
                    pass

        return

    def cleanup(self):
        """
        幂等清理（只执行一次）：
        - 标记停止并投放哨兵唤醒阻塞的 worker（不与 worker 争抢队列/不主动 task_done）
        - 等待线程退出
        - 调用父类 cleanup（若存在）
        - 强制 gc
        - 触发回调
        """
        with self._cleanup_once_lock:
            if self._cleaned_up:
                return
            self._cleaned_up = True
            print('Starting cleanup...')
            self._stop_event.set()
            for _ in range(self.num_workers):
                try:
                    self.image_queue.put_nowait(None)
                except queue.Full:
                    pass
            for t in self._workers:
                if t.is_alive():
                    t.join(timeout=2.0)
            try:
                super_cleanup = getattr(super(), "cleanup", None)
                if callable(super_cleanup):
                    super_cleanup()
            except Exception as e:
                print(f'Error calling super().cleanup(): {e!r}')
            collected = gc.collect()
            print(f"GC collected: {collected} objects")
            self.finished = True
            if self.on_cleanup_callback:
                try:
                    self.on_cleanup_callback(self)
                except Exception as e:
                    print(f'Error in on_cleanup_callback: {e!r}')
