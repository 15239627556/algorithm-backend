import os
import sys
import time
import atexit
import queue
import logging
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener

root_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(root_dir, "backend")
algorithms_dir = os.path.join(root_dir, "algorithms")
project_dir = os.path.join(root_dir, "project")
sys.path.append(root_dir)
sys.path.append(backend_dir)
sys.path.append(algorithms_dir)
sys.path.append(project_dir)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes.ImgFilter import ImgFilter
from backend.routes.task import task
from config import APP_HOST, APP_PORT, THREAD_POOL_SIZE, sufa_version, is_doc

os.makedirs("backend/uploads", exist_ok=True)
os.makedirs("backend/tmp", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# ---------------------------------------------------------------------------
# 日志：三文件 + 控制台
#
# 轮转：RotatingFileHandler — 当前文件写到约 max_bytes 后重命名为 .1、.2…，再新建空文件；
#       最多保留 1 个当前文件 + backup_count 个备份（旧备份会被删）。
# app.log / error.log — 默认单文件上限 10MB，备份 5 个
# access.log     — 单文件上限略大（请求多），但仍可配置，避免「单文件无限涨」
#
# app.log       — app logger、business(backend.*)、stdout tee；仅 INFO/WARNING
# error.log     — ERROR+（app.logger + root）
# access.log    — HTTP 中间件一行一条摘要
#
# propagate=False：不往上冒泡，避免与 root 重复输出。
# uvicorn.access 提到 WARNING：避免与 access_logger 双份访问日志。
# print → Tee 进 app.logger；StreamHandler 必须绑 __stdout__，否则会递归。
# backend.* — 与 app.logger 共用 handler（root 默认 WARNING 会吞掉子 logger 的 INFO）。
# ---------------------------------------------------------------------------

LOG_DIR = os.path.join(root_dir, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
_MB = 1024 * 1024

_LOG_APP_MB = 10
_LOG_APP_BACKUPS = 5
_ERROR_LOG_MB = 10
_ERROR_LOG_BACKUPS = 5
_ACCESS_LOG_MB = 50
_ACCESS_LOG_BACKUPS = 10
_CONSOLE = getattr(sys, "__stdout__", sys.stdout)
_FMT = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s] %(message)s")
_FMT_ACC = logging.Formatter("[%(asctime)s] %(message)s")

app_logger = logging.getLogger("app")


class _LtLevel(logging.Filter):
    """只保留低于给定级别的记录（用来让 app.log 不含 ERROR）。"""

    def __init__(self, level: int) -> None:
        super().__init__()
        self._level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < self._level


class _NameAllow(logging.Filter):
    """只接受指定 logger 域的记录（用于把 root 冒上来的非 ERROR 记录挡在 app.log 之外）。"""

    def __init__(self, names: list[str]) -> None:
        super().__init__()
        self._names = tuple(names)

    def filter(self, record: logging.LogRecord) -> bool:
        return any(record.name == n or record.name.startswith(n + ".") for n in self._names)


_log_listeners: list[QueueListener] = []


class _StdoutTeeAppLog:
    """把 write 进 sys.stdout 的内容镜像到 app.logger（进 app.log）。"""

    def __init__(self, raw, app_log: logging.Logger) -> None:
        self._raw = raw
        self._log = app_log
        self._buf = ""

    def write(self, s: str | bytes) -> int:
        if not s:
            return 0
        if isinstance(s, str):
            text = s
        elif isinstance(s, (bytes, bytearray)):
            text = s.decode(self.encoding, errors="replace")
        else:
            try:
                text = bytes(s).decode(self.encoding, errors="replace")
            except (TypeError, ValueError, UnicodeDecodeError):
                text = str(s)

        self._raw.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._log.info("[stdout] %s", line)
        return len(text)

    def flush(self) -> None:
        if self._buf:
            self._log.info("[stdout] %s", self._buf)
            self._buf = ""
        self._raw.flush()

    def isatty(self) -> bool:
        fn = getattr(self._raw, "isatty", None)
        return bool(fn and fn())

    @property
    def encoding(self):
        return getattr(self._raw, "encoding", None) or "utf-8"

    def __getattr__(self, name):
        return getattr(self._raw, name)


def _setup_logging() -> logging.Logger:
    """一次性挂好 handler；返回 access_logger 供中间件使用。"""

    def file_handler(
        path: str,
        level: int,
        fmt: logging.Formatter,
        *,
        mb: int = _LOG_APP_MB,
        backups: int = _LOG_APP_BACKUPS,
        filt: logging.Filter | None = None,
    ):
        h = RotatingFileHandler(
            os.path.join(LOG_DIR, path),
            maxBytes=mb * _MB,
            backupCount=backups,
            encoding="utf-8",
        )
        h.setLevel(level)
        h.setFormatter(fmt)
        if filt is not None:
            h.addFilter(filt)
        return h

    def stream(level: int, fmt: logging.Formatter, filt: logging.Filter | None = None):
        h = logging.StreamHandler(_CONSOLE)
        h.setLevel(level)
        h.setFormatter(fmt)
        if filt is not None:
            h.addFilter(filt)
        return h

    def queue_listener(*handlers: logging.Handler) -> "queue.Queue":
        q: "queue.Queue" = queue.Queue(-1)
        listener = QueueListener(q, *handlers, respect_handler_level=True)
        listener.start()
        _log_listeners.append(listener)
        return q

    def attach_q(log: logging.Logger, level: int, q: "queue.Queue") -> None:
        log.handlers.clear()
        log.setLevel(level)
        log.propagate = False
        log.addHandler(QueueHandler(q))

    error_fh = file_handler(
        "error.log",
        logging.ERROR,
        _FMT,
        mb=_ERROR_LOG_MB,
        backups=_ERROR_LOG_BACKUPS,
    )
    app_fh = file_handler("app.log", logging.INFO, _FMT, filt=_LtLevel(logging.ERROR))
    console_app = stream(logging.INFO, _FMT)
    access_fh = file_handler(
        "access.log",
        logging.INFO,
        _FMT_ACC,
        mb=_ACCESS_LOG_MB,
        backups=_ACCESS_LOG_BACKUPS,
    )
    console_acc = stream(logging.INFO, _FMT_ACC)

    app_domain = _NameAllow(["app", "backend"])
    app_fh.addFilter(app_domain)
    console_app.addFilter(app_domain)

    q_main = queue_listener(app_fh, error_fh, console_app)
    attach_q(app_logger, logging.INFO, q_main)
    attach_q(logging.getLogger("backend"), logging.INFO, q_main)

    acc = logging.getLogger("app.access")
    q_acc = queue_listener(access_fh, console_acc)
    attach_q(acc, logging.INFO, q_acc)

    # 压低 uvicorn 自带 access，避免与 access_logger 双份
    for name in ("uvicorn.access", "uvicorn.error"):
        uv = logging.getLogger(name)
        if name == "uvicorn.access":
            uv.setLevel(logging.WARNING)
            uv.propagate = False

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.WARNING)
    root.addHandler(QueueHandler(q_main))

    sys.stdout = _StdoutTeeAppLog(_CONSOLE, app_logger)
    return acc


access_logger = _setup_logging()


@atexit.register
def _flush_logs() -> None:
    """进程退出时停止后台监听线程，确保队列中残留日志被写完。"""
    for listener in _log_listeners:
        try:
            listener.stop()
        except Exception:
            pass


class AccessLogMiddleware:
    """纯 ASGI 访问日志，避免 BaseHTTPMiddleware 缓冲大 body。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.time()
        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = (time.time() - start) * 1000
            client = "-"
            if scope.get("client"):
                client = scope["client"][0]
            access_logger.info(
                "%s %s %s %s %.1fms",
                client,
                scope.get("method", "-"),
                scope.get("path", "-"),
                status_code,
                duration,
            )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 同步路由跑在 anyio 线程池；调大以提升阻塞推理/IO 的并发能力
    try:
        import anyio

        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = THREAD_POOL_SIZE
        app_logger.info("anyio thread pool size set to %s", THREAD_POOL_SIZE)
    except Exception:
        app_logger.exception("Failed to configure anyio thread pool size")

    try:
        from backend.tools.model_control import warmup_pinned_models_at_startup

        warmup_pinned_models_at_startup()
    except Exception:
        app_logger.exception("Triton pinned model warmup failed at startup")

    yield


_docs = "/" if is_doc else None
_openapi = "/openapi.json" if is_doc else None
_redoc = "/redoc" if is_doc else None

app = FastAPI(
    title="算法服务",
    version=sufa_version,
    description=f"{sufa_version}版本，算法服务，端口号为{APP_PORT}",
    docs_url=_docs,
    redoc_url=_redoc,
    openapi_url=_openapi,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.include_router(task)
app.include_router(ImgFilter)
# 最外层包访问日志（add_middleware 对纯 ASGI 类也可用）
app.add_middleware(AccessLogMiddleware)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    import os
    workers = min(os.cpu_count(), 6)
    # workers>1 时各进程独立 ROI 缓存（TTL 30min，见 task_service._roi_cache_*）
    uvicorn.run(
        "app:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=False,
        workers=workers,
        timeout_keep_alive=50,
        limit_concurrency=None,
    )
