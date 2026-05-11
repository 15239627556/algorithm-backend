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
# app.log       — Flask app.logger、business(backend.*)、stdout tee，仅 INFO/WARNING（ERROR 不进此文件便于阅读）
# error.log     — ERROR+（app.logger + root；root 接住第三方库的告警）
# access.log    — 下面 after_request 里一行一条 HTTP 摘要
#
# propagate=False：日志不往上冒泡，避免与 root handler 双重输出。
# werkzeug logger 提到 WARNING：否则每个请求两行访问日志（它一行 + access_logger 一行）。
# print：sys.stdout 换成 Tee → 控制台照旧，并按行记入 app.logger → app.log；
#        StreamHandler 必须绑在真实的 __stdout__ 上，不能绑 Tee，否则会 logging 递归。
# backend.*：命名空间挂上与 app.logger 同一批 handler（root 默认 WARNING，否则 getLogger(__name__).info 被吃掉）。
# ---------------------------------------------------------------------------

LOG_DIR = os.path.join(root_dir, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
_MB = 1024 * 1024
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

    def write(self, s: str) -> int:
        if not s:
            return 0
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
                     mb: int = 10, backups: int = 5,
                     filt: logging.Filter | None = None):
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
    error_fh = file_handler("error.log", logging.ERROR, _FMT)

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
        file_handler("access.log", logging.INFO, _FMT_ACC, mb=50, backups=10),
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
