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

# ========== 日志配置开始 ==========
LOG_DIR = os.path.join(root_dir, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

_fmt = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s] %(message)s")
_fmt_access = logging.Formatter("[%(asctime)s] %(message)s")


class _MaxLevelFilter(logging.Filter):
    """只放行严格低于 level 的日志，用于把 ERROR 及以上从 app.log 中剔除。"""

    def __init__(self, level: int) -> None:
        super().__init__()
        self._level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < self._level


def _make_rotating_handler(filename: str, max_bytes: int, backups: int,
                           level: int, fmt: logging.Formatter,
                           filt: logging.Filter | None = None) -> RotatingFileHandler:
    h = RotatingFileHandler(
        os.path.join(LOG_DIR, filename),
        maxBytes=max_bytes,
        backupCount=backups,
        encoding="utf-8",
    )
    h.setLevel(level)
    h.setFormatter(fmt)
    if filt is not None:
        h.addFilter(filt)
    return h


def _make_stream_handler(level: int, fmt: logging.Formatter,
                         filt: logging.Filter | None = None) -> logging.StreamHandler:
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(level)
    h.setFormatter(fmt)
    if filt is not None:
        h.addFilter(filt)
    return h


# ---------- 1. 应用日志（app.logger）：正常日志 -> app.log，错误日志 -> error.log ----------
# 关键点：关闭 propagate，避免冒泡到 root logger 被再写一次
app.logger.setLevel(logging.INFO)
app.logger.handlers.clear()
app.logger.propagate = False

# app.log：只保留 INFO / WARNING（< ERROR）
_info_only = _MaxLevelFilter(logging.ERROR)
app.logger.addHandler(_make_rotating_handler(
    "app.log", 10 * 1024 * 1024, 5, logging.INFO, _fmt, _info_only))

# error.log：只保留 ERROR / CRITICAL
app.logger.addHandler(_make_rotating_handler(
    "error.log", 10 * 1024 * 1024, 5, logging.ERROR, _fmt))

# 控制台：INFO+ 都打印，便于开发观察
app.logger.addHandler(_make_stream_handler(logging.INFO, _fmt))

# ---------- 2. 访问日志：独立文件 access.log ----------
access_logger = logging.getLogger("flask.access")
access_logger.setLevel(logging.INFO)
access_logger.handlers.clear()
access_logger.propagate = False
access_logger.addHandler(_make_rotating_handler(
    "access.log", 50 * 1024 * 1024, 10, logging.INFO, _fmt_access))
access_logger.addHandler(_make_stream_handler(logging.INFO, _fmt_access))

# ---------- 3. 抑制重复日志源 ----------
# 3.1 werkzeug 自带的访问日志（形如 `IP - - [..] "GET /x" 200 -`）与我们的 access_logger 重复，
#     提升其级别到 WARNING，只保留启动/错误信息。
_werkzeug_logger = logging.getLogger("werkzeug")
_werkzeug_logger.setLevel(logging.WARNING)
_werkzeug_logger.propagate = False

# 3.2 root logger：清理掉第三方库可能通过 basicConfig 加上的默认 StreamHandler，
#     让所有业务日志都经由我们显式配置的 logger 输出，避免 "控制台同一行出现两次"
_root_logger = logging.getLogger()
for _h in list(_root_logger.handlers):
    _root_logger.removeHandler(_h)
_root_logger.setLevel(logging.WARNING)
# root 只接收未被 propagate 截断的第三方库日志，统一写到 error.log（避免丢失三方告警）
_root_logger.addHandler(_make_rotating_handler(
    "error.log", 10 * 1024 * 1024, 5, logging.WARNING, _fmt))


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


# ========== 日志配置结束 ==========

# Triton：常驻 pipeline（config.TRITON_PINNED_PIPELINE_NAME）在进程启动时预加载，不参与 LRU 淘汰（见 project.model_control）
try:
    from project.model_control import warmup_pinned_models_at_startup

    warmup_pinned_models_at_startup()
except Exception:
    app.logger.exception("Triton pinned model warmup failed at startup")


if __name__ == '__main__':
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
