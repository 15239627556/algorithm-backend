import threading
import time
from collections import deque
# from algorithms.x40match.matcher_post import *

class MatchingFetcherThread(threading.Thread):

    def __init__(self, project, matcher, edge_lock):
        super().__init__()
        self._project = project
        self._matcher = matcher
        self._task_queue = deque()
        self._condition = threading.Condition()
        self._stop_signal = False
        self._edge_lock = edge_lock
        self._queue_lock = threading.Lock()

    def set_stop_signal(self):
        self._stop_signal = True
        with self._condition:
            self._condition.notify_all()

    def add_task(self, task_id, neighbor, row, col):
        with self._queue_lock:
            self._task_queue.append((task_id, neighbor, row, col))
            with self._condition:
                self._condition.notify()

    def run(self):
        while True:
            with self._condition:
                while not self._task_queue:
                    if self._stop_signal and len(self._task_queue) == 0:
                        return
                    self._condition.wait()
            with self._queue_lock:
                task_id, neighbor, row, col = self._task_queue.popleft()
            (dx, dy), response = self._matcher.get_result(task_id)
            if response < -1.5:
                self.add_task(task_id, neighbor, row, col)
                time.sleep(0.001)
            elif response < -0.5:
                pass
                # print("TASKID", task_id, "Low response:", response, "for edge", (row, col), "->", neighbor)
            else:
                # print("TASKID", task_id, "Edge", (row, col), "->", neighbor, "response:", response)
                dx *= DOWNSAMPLE
                dy *= DOWNSAMPLE
                with self._edge_lock:
                    self._project._grid.add_edge((row, col), neighbor, translation=(dx, dy), response=response)
                self._project._release_node(*neighbor)
                self._project._release_node(row, col)


class ModelFetcherThread(threading.Thread):

    def __init__(self, project, model, edge_lock):
        super().__init__()
        self._model = model
        self._project = project
        self._task_queue = deque()
        self._condition = threading.Condition()
        self._stop_signal = False
        self._queue_lock = threading.Lock()
        self._edge_lock = edge_lock

    def set_stop_signal(self):
        self._stop_signal = True
        with self._condition:
            self._condition.notify_all()

    def add_task(self, task_id, row, col):
        with self._queue_lock:
            self._task_queue.append((task_id, row, col))
            with self._condition:
                self._condition.notify()

    def run(self):
        while True:
            with self._condition:
                while not self._task_queue:
                    if self._stop_signal and len(self._task_queue) == 0:
                        return
                    self._condition.wait()
            with self._queue_lock:
                task_id, row, col = self._task_queue.popleft()
            result = self._model.get_result(task_id)
            if not result:
                self.add_task(task_id, row, col)
                time.sleep(0.001)
            else:
                self._project._result[(row, col)] = result
                self._project._release_node(row, col)
