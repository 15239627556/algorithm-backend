import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler

root_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(root_dir, 'backend')
algorithms_dir = os.path.join(root_dir, 'algorithms')
project_dir = os.path.join(root_dir, 'project')
sys.path.append(root_dir)
sys.path.append(backend_dir)
sys.path.append(algorithms_dir)
sys.path.append(project_dir)

from flask import Flask, request
from flask_cors import CORS
from flask_restx import Api

from backend.routes.task import task
from backend.routes.ImgFilter import ImgFilter
from config import FLASK_HOST, FLASK_PORT, sufa_version, is_doc

is_doc = '/' if is_doc else False

api = Api(
    version=sufa_version, title='算法服务',
    description=f'{sufa_version}版本，算法服务，端口号为{FLASK_PORT}',
    doc=is_doc
    )

app = Flask(__name__, static_url_path='/uploads', static_folder='uploads')
api.init_app(app)
CORS(app, supports_credentials=True)
app.secret_key = 'Donghuan@2020'

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
os.makedirs('backend/uploads', exist_ok=True)
os.makedirs('backend/images', exist_ok=True)
api.add_namespace(task)
api.add_namespace(ImgFilter)

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


class _StdoutTeeAppLog:
    """把 write 进 sys.stdout 的内容镜像到 app.logger（进 app.log）。"""

    def __init__(self, raw, app_log: logging.Logger) -> None:
        self._raw = raw
        self._log = app_log
        self._buf = ""

    def write(self, s: str | bytes) -> int:
        # Flask/Click 等有时写 str，极少数路径写 bytes；真实 stdout 为文本模式时只接收 str
        if not s:
            return 0
        if isinstance(s, bytes):
            s = s.decode(self.encoding, errors="replace")
        self._raw.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._log.info("[stdout] %s", line)
        return len(s)

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

    def attach(log: logging.Logger, level: int, *handlers: logging.Handler) -> None:
        log.handlers.clear()
        log.setLevel(level)
        log.propagate = False
        for h in handlers:
            log.addHandler(h)

    # 两个 logger 共用同一 error 实体：谁触发就只 emit 一次，不会重复追加两条
    error_fh = file_handler(
        "error.log", logging.ERROR, _FMT,
        mb=_ERROR_LOG_MB, backups=_ERROR_LOG_BACKUPS,
    )

    attach(
        flask_app.logger,
        logging.INFO,
        file_handler("app.log", logging.INFO, _FMT, filt=_LtLevel(logging.ERROR)),
        error_fh,
        stream(logging.INFO, _FMT),
    )

    acc = logging.getLogger("flask.access")
    attach(
        acc,
        logging.INFO,
        file_handler(
            "access.log", logging.INFO, _FMT_ACC,
            mb=_ACCESS_LOG_MB, backups=_ACCESS_LOG_BACKUPS,
        ),
        stream(logging.INFO, _FMT_ACC),
    )

    # 保留 werkzeug 自带 handler，只抬高级别，避免清掉后连 WARNING 也看不见
    wz = logging.getLogger("werkzeug")
    wz.setLevel(logging.WARNING)
    wz.propagate = False

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.WARNING)
    root.addHandler(error_fh)

    attach(logging.getLogger("backend"), logging.INFO, *list(flask_app.logger.handlers))
    sys.stdout = _StdoutTeeAppLog(_CONSOLE, flask_app.logger)
    return acc


access_logger = _setup_logging(app)


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


# Triton：常驻 pipeline（config.TRITON_PINNED_PIPELINE_NAME）在进程启动时预加载，不参与 LRU 淘汰（见 backend.tools.model_control）
try:
    from backend.tools.model_control import warmup_pinned_models_at_startup

    warmup_pinned_models_at_startup()
except Exception:
    app.logger.exception("Triton pinned model warmup failed at startup")


if __name__ == '__main__':
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
