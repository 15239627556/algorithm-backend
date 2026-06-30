# --- 必须放在 app.py 的第一行和第二行！ ---
from gevent import monkey
monkey.patch_all()

import os
import sys
import time
import atexit
import queue
import logging
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener

root_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(root_dir, 'backend')
algorithms_dir = os.path.join(root_dir, 'algorithms')
project_dir = os.path.join(root_dir, 'project')
sys.path.append(root_dir)
sys.path.append(backend_dir)
sys.path.append(algorithms_dir)
sys.path.append(project_dir)

from flask import Flask, request, Request
from flask_cors import CORS
from flask_restx import Api
import tempfile
from backend.routes.task import task
# from backend.routes.ImgFilter import ImgFilter
from config import FLASK_HOST, FLASK_PORT, sufa_version, is_doc

class FastMemoryRequest(Request):
    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        # 将内存阈值提高到 15MB (15 * 1024 * 1024)
        # 只要上传的图片小于 15MB，它就会一直驻留在 RAM (内存) 中，绝不写磁盘！
        return tempfile.SpooledTemporaryFile(max_size=15 * 1024 * 1024, mode="wb+")

is_doc = '/' if is_doc else False

api = Api(
    version=sufa_version, title='算法服务',
    description=f'{sufa_version}版本，算法服务，端口号为{FLASK_PORT}',
    doc=is_doc
    )

app = Flask(__name__, static_url_path='/uploads', static_folder='uploads')
app.request_class = FastMemoryRequest
api.init_app(app)
CORS(app, supports_credentials=True)
app.secret_key = 'Donghuan@2020'

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TMP_FOLDER'] = 'tmp'
os.makedirs('backend/uploads', exist_ok=True)
os.makedirs('backend/tmp', exist_ok=True)
api.add_namespace(task)
# api.add_namespace(ImgFilter)

@app.route("/health")
def health():
    return {"status": "ok"}, 200


# ---------------------------------------------------------------------------
# 日志：三文件 + 控制台
#
# 轮转：RotatingFileHandler — 当前文件写到约 max_bytes 后重命名为 .1、.2…，再新建空文件；
#       最多保留 1 个当前文件 + backup_count 个备份（旧备份会被删）。
# app.log / error.log — 默认单文件上限 10MB，备份 5 个
# access.log     — 单文件上限略大（请求多），但仍可配置，避免「单文件无限涨」
#
# app.log       — Flask app.logger、business(backend.*)、stdout tee；仅 INFO/WARNING
# error.log     — ERROR+（app.logger + root）
# access.log    — after_request 一行一条 HTTP 摘要
#
# propagate=False：不往上冒泡，避免与 root 重复输出。
# werkzeug 提到 WARNING：避免与 access_logger 双份访问日志。
# print → Tee 进 app.logger；StreamHandler 必须绑 __stdout__，否则会递归。
# backend.* — 与 app.logger 共用 handler（root 默认 WARNING 会吞掉子 logger 的 INFO）。
# ---------------------------------------------------------------------------

LOG_DIR = os.path.join(root_dir, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
_MB = 1024 * 1024

# 与下面 file_handler() 默认一致，集中改一处即可
_LOG_APP_MB = 10
_LOG_APP_BACKUPS = 5
_ERROR_LOG_MB = 10
_ERROR_LOG_BACKUPS = 5
# 访问日志与 QPS 成正比；单文件上限仍强制轮转（可按流量把 _ACCESS_LOG_MB 调大/调小）
_ACCESS_LOG_MB = 50
_ACCESS_LOG_BACKUPS = 10
_CONSOLE = getattr(sys, "__stdout__", sys.stdout)
_FMT = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s] %(message)s")
_FMT_ACC = logging.Formatter("[%(asctime)s] %(message)s")


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


# 持有所有后台日志监听线程；进程退出时统一 stop() 以 flush 队列中残留日志
_log_listeners: list[QueueListener] = []


class _StdoutTeeAppLog:
    """把 write 进 sys.stdout 的内容镜像到 app.logger（进 app.log）。"""

    def __init__(self, raw, app_log: logging.Logger) -> None:
        self._raw = raw
        self._log = app_log
        self._buf = ""

    def write(self, s: str | bytes) -> int:
        # Flask/Click 等对 stdout.write 既有 str，也有 bytes / buffer；文本模式 __stdout__ 只收 str
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


def _setup_logging(flask_app: Flask):
    """一次性挂好 handler；返回 access_logger 供路由钩子里使用。"""

    def file_handler(path: str, level: int, fmt: logging.Formatter, *,
                     mb: int = _LOG_APP_MB, backups: int = _LOG_APP_BACKUPS,
                     filt: logging.Filter | None = None):
        # RotatingFileHandler：见文件头「轮转」说明
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

    # —— 用队列把 logger 与真实 handler 解耦：logger 只挂 QueueHandler，
    #    真实 file/console handler 的 emit（格式化 + 落盘）都在后台线程执行 ——
    def queue_listener(*handlers: logging.Handler) -> "queue.Queue":
        q: "queue.Queue" = queue.Queue(-1)  # 无界队列，避免高峰丢日志
        # respect_handler_level=True：后台 listener 仍尊重各 handler 的 level/filter
        listener = QueueListener(q, *handlers, respect_handler_level=True)
        listener.start()
        _log_listeners.append(listener)
        return q

    def attach_q(log: logging.Logger, level: int, q: "queue.Queue") -> None:
        log.handlers.clear()
        log.setLevel(level)
        log.propagate = False
        log.addHandler(QueueHandler(q))

    # —— 真实 handler（其 emit 只在后台线程跑）——
    # error.log 单实例：谁触发就只 emit 一次，不会重复追加两条
    error_fh = file_handler(
        "error.log", logging.ERROR, _FMT,
        mb=_ERROR_LOG_MB, backups=_ERROR_LOG_BACKUPS,
    )
    app_fh = file_handler("app.log", logging.INFO, _FMT, filt=_LtLevel(logging.ERROR))
    console_app = stream(logging.INFO, _FMT)
    access_fh = file_handler(
        "access.log", logging.INFO, _FMT_ACC,
        mb=_ACCESS_LOG_MB, backups=_ACCESS_LOG_BACKUPS,
    )
    console_acc = stream(logging.INFO, _FMT_ACC)

    # app.log / 控制台只收 app 域（flask.app + backend），挡掉 root 冒上来的非 ERROR 记录
    app_domain = _NameAllow([flask_app.logger.name, "backend"])
    app_fh.addFilter(app_domain)
    console_app.addFilter(app_domain)

    # 主域队列：flask.app、backend、root 共用同一队列与 error_fh 实例
    q_main = queue_listener(app_fh, error_fh, console_app)
    attach_q(flask_app.logger, logging.INFO, q_main)
    attach_q(logging.getLogger("backend"), logging.INFO, q_main)

    # 访问日志独立队列
    acc = logging.getLogger("flask.access")
    q_acc = queue_listener(access_fh, console_acc)
    attach_q(acc, logging.INFO, q_acc)

    # 保留 werkzeug 自带 handler，只抬高级别，避免清掉后连 WARNING 也看不见
    wz = logging.getLogger("werkzeug")
    wz.setLevel(logging.WARNING)
    wz.propagate = False

    # root：走主队列，error_fh(ERROR) 负责落盘；非 ERROR 被 app_domain 过滤掉
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.WARNING)
    root.addHandler(QueueHandler(q_main))

    sys.stdout = _StdoutTeeAppLog(_CONSOLE, flask_app.logger)
    return acc


access_logger = _setup_logging(app)


@atexit.register
def _flush_logs() -> None:
    """进程退出时停止后台监听线程，确保队列中残留日志被写完。"""
    for listener in _log_listeners:
        try:
            listener.stop()
        except Exception:
            pass


@app.before_request
def _log_request_start():
    request._log_start = time.time()
    if is_doc!='/':
        if request.path.endswith('/swagger.json'):
            return "Forbidden", 403


@app.after_request
def _log_request(response):
    duration = (time.time() - getattr(request, "_log_start", 0)) * 1000
    access_logger.info(
        "%s %s %s %s %.1fms",
        request.remote_addr,
        request.method,
        request.path,
        response.status_code,
        duration,
    )
    return response

# 启动项目预加载模型
try:
    from backend.tools.model_control import warmup_pinned_models_at_startup

    warmup_pinned_models_at_startup()
except Exception:
    app.logger.exception("Triton pinned model warmup failed at startup")


if __name__ == '__main__':
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
