import logging
import os
import os.path
import time
from threading import Thread, Lock
from functools import wraps
import cProfile
import pstats

upload_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads')
images_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')

_logger = logging.getLogger(__name__)


def thread_decorator(f):
    """
    异步装饰器
    :param f:
    :return:
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs, daemon=True)
        thr.start()

        return thr

    return wrapper


def get_new_task_id() -> str:
    """
    # Generate a random UUID without dashes
    获取新的任务ID
    :return: 新的任务ID
    """
    from uuid import uuid4

    return str(uuid4()).replace('-', '')


class _WindowProfiler:
    """在时间窗口内聚合 cProfile，到期输出平均耗时 TopN。"""

    def __init__(self, name: str, window_sec: float, top_n: int):
        self.name = name
        self.window_sec = window_sec
        self.top_n = top_n
        self._lock = Lock()
        self._stats: pstats.Stats | None = None
        self._calls = 0
        self._window_start = time.perf_counter()

    def record(self, profiler: cProfile.Profile) -> None:
        report = None
        with self._lock:
            if self._stats is None:
                self._stats = pstats.Stats(profiler)
                self._window_start = time.perf_counter()
            else:
                self._stats.add(profiler)
            self._calls += 1
            elapsed = time.perf_counter() - self._window_start
            if elapsed >= self.window_sec and self._calls > 0:
                report = self._build_report(elapsed)
                self._stats = None
                self._calls = 0
                self._window_start = time.perf_counter()
        if report:
            _logger.info("%s", report)

    def _build_report(self, elapsed: float) -> str:
        assert self._stats is not None
        stats = self._stats.stats  # {(file, line, func): (cc, nc, tt, ct, callers)}
        rows = []
        for (filename, line, func), (cc, nc, tt, ct, _callers) in stats.items():
            if nc <= 0:
                continue
            # 跳过装饰器自身/包装函数噪声
            if func in ("wrapper", "record", "_build_report", "profile_route"):
                continue
            avg_ms = (tt / nc) * 1000.0
            per_outer_ms = (tt / self._calls) * 1000.0 if self._calls else 0.0
            label = f"{func}  ({os.path.basename(filename)}:{line})"
            rows.append((avg_ms, per_outer_ms, tt * 1000.0, nc, label))

        # 按单次平均自用耗时(avg tottime)降序，取 TopN
        rows.sort(key=lambda r: r[0], reverse=True)
        top = rows[: self.top_n]

        lines = [
            f"--- 窗口性能 Top{self.top_n} [{self.name}] "
            f"window={elapsed:.1f}s calls={self._calls} ---",
            f"{'秩':>3}  {'avg_ms':>10}  {'per_call_ms':>12}  {'total_ms':>10}  {'ncalls':>8}  方法",
        ]
        for i, (avg_ms, per_outer_ms, total_ms, nc, label) in enumerate(top, 1):
            lines.append(
                f"{i:>3}  {avg_ms:10.3f}  {per_outer_ms:12.3f}  {total_ms:10.1f}  {nc:8d}  {label}"
            )
        lines.append(
            "说明: avg_ms=方法单次平均自用耗时(tottime/ncalls); "
            "per_call_ms=均摊到每次 upload 的自用耗时(tottime/外层calls)"
        )
        return "\n".join(lines)


def profile_route(route_func=None, *, window_sec: float | None = None, top_n: int = 10):
    """
    时间窗口内聚合 cProfile，到期输出平均耗时最高的前 top_n 个方法。

    环境变量 PROFILE_WINDOW_SEC 可覆盖窗口秒数（默认 30）。
    用法:
        @profile_route
        @profile_route(window_sec=60, top_n=10)
    """

    def decorator(func):
        sec = window_sec
        if sec is None:
            sec = float(os.environ.get("PROFILE_WINDOW_SEC", "30"))
        aggregator = _WindowProfiler(func.__name__, window_sec=sec, top_n=top_n)

        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            try:
                return func(*args, **kwargs)
            finally:
                profiler.disable()
                aggregator.record(profiler)

        return wrapper

    if route_func is not None:
        return decorator(route_func)
    return decorator
