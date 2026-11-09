# -*- coding: utf-8 -*-
"""Web 启动入口。

setcover 的顶层求解跑在子进程里，子进程会重新执行当前主模块。
这个文件被重新执行时不做导入；请用它启动服务，不要让 app.py 成为主模块。
"""

if __name__ == "__main__":
    import uvicorn

    from app import APP_HOST, APP_PORT, _install_worker_exit_logger

    _install_worker_exit_logger()
    uvicorn.run(
        "app:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=False,
        workers=2,
        timeout_keep_alive=50,
        limit_concurrency=None,
    )
