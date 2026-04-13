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
from config import FLASK_HOST, FLASK_PORT

api = Api(version='2025.12.12', title='算法服务',
          description=f'2025.12.12版本，算法服务，端口号为{FLASK_PORT}', doc="/")
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

_fmt = "[%(asctime)s] %(levelname)s [%(name)s] %(message)s"
_fmt_access = "[%(asctime)s] %(message)s"

# 1. 应用主日志（app.logger）：INFO 及以上，输出到文件 + 控制台
app.logger.setLevel(logging.INFO)
app.logger.handlers.clear()

_app_fh = RotatingFileHandler(
    os.path.join(LOG_DIR, "app.log"),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_app_fh.setLevel(logging.INFO)
_app_fh.setFormatter(logging.Formatter(_fmt))
app.logger.addHandler(_app_fh)

_app_ch = logging.StreamHandler(sys.stdout)
_app_ch.setLevel(logging.INFO)
_app_ch.setFormatter(logging.Formatter(_fmt))
app.logger.addHandler(_app_ch)

# 2. 错误日志：仅 ERROR
_error_fh = RotatingFileHandler(
    os.path.join(LOG_DIR, "error.log"),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_error_fh.setLevel(logging.ERROR)
_error_fh.setFormatter(logging.Formatter(_fmt))
app.logger.addHandler(_error_fh)

# 3. 访问日志：IP、方法、路径、状态码、耗时
access_logger = logging.getLogger("flask.access")
access_logger.setLevel(logging.INFO)
access_logger.propagate = False

_access_fh = RotatingFileHandler(
    os.path.join(LOG_DIR, "access.log"),
    maxBytes=50 * 1024 * 1024,
    backupCount=10,
    encoding="utf-8",
)
_access_fh.setLevel(logging.INFO)
_access_fh.setFormatter(logging.Formatter(_fmt_access))
access_logger.addHandler(_access_fh)

_access_ch = logging.StreamHandler(sys.stdout)
_access_ch.setLevel(logging.INFO)
_access_ch.setFormatter(logging.Formatter(_fmt_access))
access_logger.addHandler(_access_ch)


@app.before_request
def _log_request_start():
    request._log_start = time.time()


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
    print("### FLASK MAIN PID =", os.getpid())
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
