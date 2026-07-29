# -*- coding: utf-8 -*-
# 暂时不用这个文件
"""Gunicorn + gevent 启动配置。用法: gunicorn -c gunicorn.conf.py app:app"""
import multiprocessing
import os

bind = f"{os.environ.get('APP_HOST', '0.0.0.0')}:{os.environ.get('APP_PORT', '3090')}"
worker_class = "gevent"
# gevent 以 I/O 为主：默认 min(CPU 核数, 上限)；GUNICORN_WORKERS 可强制指定，GUNICORN_WORKERS_MAX 可调上限
_cpu_count = multiprocessing.cpu_count() or 1
_workers_max = int(os.environ.get("GUNICORN_WORKERS_MAX", "8"))
_default_workers = min(_cpu_count, _workers_max)
workers = int(os.environ.get("GUNICORN_WORKERS", str(_default_workers)))
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
keepalive = int(os.environ.get("GUNICORN_KEEPALIVE", "5"))
# 访问日志由 app.py 的 access_logger 写入 logs/access.log，避免与 Gunicorn 重复
accesslog = None
errorlog = "-"
