import os, sys
import logging
from logging.handlers import RotatingFileHandler

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

api = Api(version='2025.12.09', title='算法服务',
          description='2025.12.09版本，算法服务，端口号为3889', doc="/")
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
# 错误日志
error_handler = RotatingFileHandler("error.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
))
app.logger.addHandler(error_handler)

# 访问日志
access_logger = logging.getLogger("access")
access_handler = RotatingFileHandler("access.log", maxBytes=50 * 1024 * 1024, backupCount=10, encoding="utf-8")
access_handler.setLevel(logging.INFO)
access_handler.setFormatter(logging.Formatter(
    '[%(asctime)s] %(levelname)s: %(message)s'
))
access_logger.addHandler(access_handler)


@app.before_request
def log_request():
    access_logger.info(f"{request.remote_addr} {request.method} {request.path}")


# ========== 日志配置结束 ==========

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3889, debug=False, threaded=True)
