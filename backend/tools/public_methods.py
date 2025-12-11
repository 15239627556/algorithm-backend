import os.path
from threading import Thread
from functools import wraps
import cProfile
import io, pstats

upload_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads')
images_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')


def thread_decorator(f):
    """
    异步装饰器
    :param f:
    :return:
    """

    def wrapper(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()

    return wrapper


def get_new_task_id() -> str:
    """
    # Generate a random UUID without dashes
    获取新的任务ID
    :return: 新的任务ID
    """
    from uuid import uuid4

    return str(uuid4()).replace('-', '')


def profile_route(route_func):
    @wraps(route_func)
    def wrapper(*args, **kwargs):
        # 创建分析器
        profiler = cProfile.Profile()
        profiler.enable()

        # 执行被装饰的函数
        result = route_func(*args, **kwargs)

        profiler.disable()

        # 生成分析报告
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
        ps.print_stats(20)  # 显示耗时前20的函数

        print(f"\n--- 性能分析报告 [{route_func.__name__}] ---")
        print(s.getvalue())

        return result

    return wrapper
