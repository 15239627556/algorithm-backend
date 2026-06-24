import json
import os
import threading
from io import BytesIO
import time
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional

from cachetools import TTLCache

from backend.tools.MESSAGE_DICT import RetCode, RetDesc
from backend.tools.public_methods import thread_decorator, upload_folder
from backend.tools.combo_validator import validate_combo
from backend.tools.json_safe_writer import serialize_non_json_fields
from backend.tools.dedup_cells_across_tiles import dedup_cells_across_tiles
from backend.tools.filter_edge_incomplete_cells import (
    filter_cell_dicts_edge_incomplete,
    filter_edge_incomplete_cells,
)
from PIL import Image

from project.smear_project import SmearProject
from project.cells import Cell
from backend.tools.triton_client import infer, get_model_by_dpi
from backend.tools.model_control import warmup_model
from algorithms.SelectArea.main_wbc import *
from algorithms.SelectArea.main_meg import *
from algorithms.SelectArea.setcover import solve, SetCoverSolverParameter

logger = logging.getLogger(__name__)


def _async_finish_after_update_coordinates() -> bool:
    """为 True 时 update_coordinates 在更新坐标后立即返回，去重/过滤/落盘在后台线程执行。"""
    v = os.environ.get("UPDATE_COORDINATES_ASYNC_FINISH", "1").strip().lower()
    return v not in ("0", "false", "no", "off", "")


# In-memory cache TTL (10 min), clear on restart. TTLCache is thread-safe.
CACHE_TTL_SEC = int(os.environ.get("ROI_CACHE_TTL", "600"))
ROI_CACHE_MAXSIZE = int(os.environ.get("ROI_CACHE_MAXSIZE", "200"))
PROJECT_X100_CACHE_MAXSIZE = int(os.environ.get("PROJECT_X100_CACHE_MAXSIZE", "50"))

# Task 内存上下文：自最后一次读/写起超过该时间则从 self.tasks 淘汰（可再经 load_data 从磁盘恢复）
TASK_CONTEXT_IDLE_TTL_SEC = int(os.environ.get("TASK_CONTEXT_IDLE_TTL", "1800"))


def _ensure_json_serializable(obj):
    """将 scores（可能含 numpy、嵌套列表）转为 JSON 可序列化的 Python 原生类型"""
    if obj is None:
        return []
    if hasattr(obj, "tolist") and not isinstance(obj, (list, tuple)):
        return obj.flatten().tolist() if hasattr(obj, "flatten") else obj.tolist()
    if isinstance(obj, (list, tuple)):
        result = []
        for x in obj:
            if isinstance(x, (list, tuple)):
                result.append(_ensure_json_serializable(x))
            elif hasattr(x, "item"):
                result.append(float(x.item()))
            else:
                result.append(float(x))
        return result
    try:
        return [float(obj.item())] if hasattr(obj, "item") else [float(obj)]
    except (TypeError, ValueError):
        return []


def _task_info_path(task_id: str) -> str:
    return os.path.join(upload_folder, f"{task_id}.info.json")


def _save_task_info(task_id: str, info: dict) -> None:
    """持久化 task_info（matcher 的 key 转为 "r,c" 字符串以便 JSON 序列化）"""
    os.makedirs(upload_folder, exist_ok=True)
    payload = dict(info)
    matcher = payload.get("matcher", {})
    if matcher:
        payload["matcher"] = {f"{k[0]},{k[1]}": v for k, v in matcher.items()}
    with open(_task_info_path(task_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_task_info(task_id: str) -> dict | None:
    """从磁盘加载 task_info"""
    path = _task_info_path(task_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    matcher_data = data.get("matcher", {})
    if matcher_data:
        data["matcher"] = {tuple(int(x) for x in k.split(",")): v for k, v in matcher_data.items()}
    return data


def _parse_edge_cell_filter_flag(value) -> bool:
    """multipart/form 里布尔常为字符串，避免 bool('false') == True。"""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("0", "false", "no", "off", ""):
        return False
    return True


@dataclass
class TaskContext:
    """单任务上下文，聚合 project、info、lock（已去掉 num_rows/num_cols 与 grid）"""
    project: SmearProject
    info: dict
    lock: threading.Lock = field(default_factory=threading.Lock)
    coord_mutex: threading.Lock = field(default_factory=threading.Lock)
    finish_thread: Optional[threading.Thread] = field(default=None, repr=False)


class TaskService:
    def __init__(self):
        self.tasks: Dict[str, TaskContext] = {}
        self._tasks_lock = threading.Lock()
        self._task_last_access: Dict[str, float] = {}
        self.roi_cache = TTLCache(maxsize=ROI_CACHE_MAXSIZE, ttl=CACHE_TTL_SEC)
        self.roi_cache_lock = threading.Lock()
        self.project_x100 = TTLCache(maxsize=PROJECT_X100_CACHE_MAXSIZE, ttl=CACHE_TTL_SEC)
        self.project_x100_lock = threading.Lock()

    def _evict_idle_tasks_unlocked(self) -> None:
        if not self._task_last_access:
            return
        now = time.time()
        deadline = now - TASK_CONTEXT_IDLE_TTL_SEC
        stale = [tid for tid, t in self._task_last_access.items() if t < deadline]
        for tid in stale:
            self.tasks.pop(tid, None)
            self._task_last_access.pop(tid, None)

    def _touch_task(self, task_id: str) -> None:
        """在任意成功访问/写入该任务后调用，刷新 idle 计时，并清理其它过期任务。勿在 ctx.lock 内调用。"""
        with self._tasks_lock:
            self._evict_idle_tasks_unlocked()
            if task_id in self.tasks:
                self._task_last_access[task_id] = time.time()

    def load_data(self, task_id):
        with self._tasks_lock:
            self._evict_idle_tasks_unlocked()
            if task_id in self.tasks:
                self._task_last_access[task_id] = time.time()
                return None
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, f"{task_id}.json")
        if not os.path.exists(file_path):
            return {"ret_code": RetCode.CLIENT_ERROR.value,
                    "ret_desc": 'Task ID not found',
                    'reason': 'Task ID not found',
                    'msg': f"file for task_id '{task_id}' not found."}
        project = SmearProject.load_json(file_path)
        info = _load_task_info(task_id)
        if info is None:
            dpi = project.list_layers()[0].dpi
            layer = project.get_layer(dpi)
            tiles = list(layer.iter_tiles())
            info = {
                "dpi": layer.dpi,
                "smear_type": project.smear_type,
                "tile_width": tiles[0].w if tiles else 2448,
                "tile_height": tiles[0].h if tiles else 2048,
                "task_status": RetCode.TASK_FINISHED.value if tiles else RetCode.TASK_RUNNING.value,
                "finished": True,
                "matcher": {},
            }
            for t in tiles:
                r, c = t.meta.get("row_index"), t.meta.get("col_index")
                if r is not None and c is not None:
                    info.setdefault("matcher", {})[(r, c)] = t.image_uid
        with self._tasks_lock:
            self._evict_idle_tasks_unlocked()
            self.tasks[task_id] = TaskContext(project=project, info=info)
            self._task_last_access[task_id] = time.time()

    def create_task(self, task_info: dict) -> dict:
        dpi = task_info.get('dpi')
        smear_type = task_info.get('smear_type')
        target_cell_types = task_info.get('target_cell_types')
        ok, err = validate_combo(dpi, smear_type, target_cell_types, allow_empty_types=False)
        if not ok:
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': err,
                'reason': err,
            }
        task_id = uuid.uuid4().hex
        task_info['smear_type'] = task_info.get('smear_type', 'BM')
        task_info['task_status'] = RetCode.TASK_RUNNING.value
        task_info['heatmap_orientation'] = int(task_info.get('heatmap_orientation', -1))
        task_info['finished'] = False
        task_info['matcher'] = {}
        project = SmearProject(smear_type=task_info['smear_type'])
        project.add_layer(task_info['dpi'])

        with self._tasks_lock:
            self._evict_idle_tasks_unlocked()
            self.tasks[task_id] = TaskContext(project=project, info=task_info)
            self._task_last_access[task_id] = time.time()
        _save_task_info(task_id, task_info)
        model_name, model_warning = get_model_by_dpi(
            dpi,
            smear_type=task_info.get('smear_type', 'BM'),
            algorithm_types=task_info.get('target_cell_types', ''),
            return_warning=True,
        )
        warmup_model(model_name)
        warning = err or model_warning
        if warning:
            logger.warning("创建任务 DPI 告警：%s, dpi=%s, model=%s", warning, dpi, model_name)
        logger.info('创建任务成功：%s', task_id)

        response = {
            'task_id': task_id,
            'ret_code': RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value
        }
        if warning:
            response['warning'] = warning
        return response

    def update_coordinates(self, task_id, tiles_msg):
        t0 = time.time()
        if task_id not in self.tasks:
            result = self.load_data(task_id)
            if result:
                return result
        t_load = time.time()
        ctx = self.tasks[task_id]
        self._touch_task(task_id)
        project = ctx.project
        dpi = ctx.info.get('dpi', 144750)
        layer = project.get_layer(dpi)
        project_info = ctx.info

        with ctx.coord_mutex:
            with ctx.lock:
                matcher = project_info.get('matcher')
                if not matcher:
                    matcher = {}
                    project_info['matcher'] = matcher
                prev = ctx.finish_thread
                if prev is not None and prev.is_alive():
                    prev.join(timeout=3600)

                tiles_list = layer.iter_tiles()
                rowcol_to_tile = {}
                for tile in tiles_list:
                    r, c = tile.meta.get("row_index"), tile.meta.get("col_index")
                    if r is not None and c is not None:
                        rowcol_to_tile[(int(r), int(c))] = tile

                failed_tiles = []
                for tile_info in tiles_msg:
                    row_index = int(tile_info['row_index'])
                    col_index = int(tile_info['col_index'])
                    position_x = int(tile_info['position_x'])
                    position_y = int(tile_info['position_y'])
                    image_uid = matcher.get((row_index, col_index))
                    if image_uid is None:
                        tile = rowcol_to_tile.get((row_index, col_index))
                        if tile is None:
                            tile_info['reason'] = 'tile not found'
                            failed_tiles.append(tile_info)
                        else:
                            tile.x = position_x
                            tile.y = position_y
                    else:
                        try:
                            tile = layer.get_tile(image_uid)
                            if tile is None:
                                tile_info['reason'] = 'tile not found'
                                failed_tiles.append(tile_info)
                            else:
                                tile.x = position_x
                                tile.y = position_y
                        except Exception as e:
                            tile_info['reason'] = str(e)
                            failed_tiles.append(tile_info)

        t_done = time.time()
        logger.info(
            "update_coordinates task_id=%s load_ms=%.2f coord_update_ms=%.2f",
            task_id[:8],
            (t_load - t0) * 1000,
            (t_done - t_load) * 1000,
        )

        async_finish = _async_finish_after_update_coordinates()
        if async_finish:
            def _run():
                try:
                    with ctx.lock:
                        self._finish_task_impl(task_id)
                finally:
                    self._touch_task(task_id)

            th = threading.Thread(
                target=_run,
                name=f"finish-{task_id[:8]}",
                daemon=True,
            )
            ctx.finish_thread = th
            th.start()
        else:
            with ctx.lock:
                self._finish_task_impl(task_id)
            self._touch_task(task_id)

        out: Dict[str, Any] = {
            'ret_code': RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value,
            'failed_tiles': failed_tiles,
        }
        if async_finish:
            out['finish_in_background'] = True
        return out

    def upload_image(self, task_id, row_index, col_index, tile_image):
        """任务模式：上传拼图块到指定任务，DPI/smear_type 从 task_info 取"""
        filename = tile_image.filename
        # 记录日志，task_id,file_name,接收到图片的时间,转换为时分秒毫秒
        logger.info("task_id=%s, file_name=%s, 接收到图片的时间：%s", task_id, filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
        image_bytes = tile_image.read()
        if row_index is None or col_index is None:
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': 'Task mode requires row_index and col_index',
                'reason': 'Task mode requires row_index and col_index',
            }
        if task_id not in self.tasks:
            result = self.load_data(task_id)
            if result:
                return result
            if task_id not in self.tasks:
                return {
                    'ret_code': RetCode.CLIENT_ERROR.value,
                    'ret_desc': 'Task ID not found',
                    'reason': 'Task ID not found',
                }

        ctx = self.tasks[task_id]
        row_index, col_index = int(row_index), int(col_index)
        # with ctx.lock:
        #     matcher = ctx.info.get('matcher')
        #     if not matcher:
        #         matcher = {}
        #         ctx.info['matcher'] = matcher
        matcher = ctx.info.get('matcher')
        image_uid = matcher.get((row_index, col_index))
        project = ctx.project
        dpi = ctx.info.get('dpi', 144750)
        layer = project.get_layer(dpi)
        task_info = ctx.info
        smear_type = task_info.get('smear_type', 'BM')
        target_cell_types = task_info.get('target_cell_types', '')

        if image_uid is None:
            tile = layer.add_tile(
                x=None, y=None,
                w=task_info['tile_width'],
                h=task_info['tile_height'],
                image_data=None,
                image_path=filename,
                extra_meta={
                    'row_index': row_index,
                    'col_index': col_index,
                }
            )
            image_uid = tile.image_uid
            matcher[(row_index, col_index)] = image_uid
        else:
            tile = layer.get_tile(image_uid)
        try:
            # 记录日志，task_id,file_name,发送请求的时间
            logger.info("task_id=%s, file_name=%s, 发送请求的时间：%s", task_id, filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
            result = infer(
                image_bytes,
                dpi=int(dpi),
                smear_type=smear_type,
                algorithm_types=target_cell_types or "",
                filename=filename,
            )
            # 记录日志，task_id,file_name,推理完成的时间
            logger.info("task_id=%s, file_name=%s, 推理完成的时间：%s", task_id, filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
            cells = result["cells"]
            scores = result.get("scores", [])
            cell_list = result.get("cell_list", [])
            task_info["wbc_pixel_count"] = result.get("wbc_pixel_count", 0) + task_info.get("wbc_pixel_count", 0)
            task_info["red_pixel_count"] = result.get("red_pixel_count", 0) + task_info.get("red_pixel_count", 0)
            # tile.meta["wbc_pixel_count"] = wbc_pixel_count
            # tile.meta["red_pixel_count"] = red_pixel_count
            tile.meta["scores"] = _ensure_json_serializable(scores)
            if cells:
                tile.add_cells(cells)
            # 记录日志，task_id,file_name,返回结果的时间
            logger.info("task_id=%s, file_name=%s, 返回结果的时间：%s", task_id, filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
            return {
                'ret_code': RetCode.API_SUCCESS.value,
                'ret_desc': RetDesc.API_SUCCESS.value,
                'image_uid': image_uid,
                'cell_list': cell_list,
            }
        except Exception as e:
            logger.exception("Triton inference failed for task %s: %s", task_id, e)
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': str(e),
                'reason': str(e),
            }

    def _finish_task_impl(self, task_id: str) -> None:
        """所有 tile 已上传并推理完成：去重、过滤、落盘。须在持有 ctx.lock 时调用。"""
        t5 = time.time()
        ctx = self.tasks.get(task_id)
        if not ctx:
            return
        try:
            project = ctx.project
            dpi = ctx.info.get('dpi', 144750)
            layer = project.get_layer(dpi)
            tiles = layer.iter_tiles()
            tiles = dedup_cells_across_tiles(tiles)
            t6 = time.time()
            logger.info(
                "dedup_cells_across_tiles task_id=%s ms=%.2f",
                task_id[:8],
                (t6 - t5) * 1000,
            )
            info = ctx.info
            task_tw = info.get("tile_width")
            task_th = info.get("tile_height")
            if task_tw is None and tiles:
                task_tw = tiles[0].w
            if task_th is None and tiles:
                task_th = tiles[0].h
            if task_tw is not None and task_th is not None:
                filter_edge_incomplete_cells(
                    tiles,
                    task_tile_w=int(task_tw),
                    task_tile_h=int(task_th),
                )
            else:
                logger.warning(
                    "Task %s: skip filter_edge_incomplete_cells (no tile_width/tile_height)",
                    task_id,
                )
            for one_tile in tiles:
                layer.tiles[one_tile.image_uid] = one_tile
            t9 = time.time()
            logger.info(
                "filter_edge_incomplete_cells task_id=%s ms=%.2f",
                task_id[:8],
                (t9 - t6) * 1000,
            )
            project.save_json(os.path.join(upload_folder, f"{task_id}.json"))
            t7 = time.time()
            logger.info(
                "save_json task_id=%s ms=%.2f",
                task_id[:8],
                (t7 - t9) * 1000,
            )
            ctx.info['task_status'] = RetCode.TASK_FINISHED.value
            ctx.info['finished'] = True
            _save_task_info(task_id, ctx.info)
            t8 = time.time()
            logger.info(
                "save_task_info task_id=%s ms=%.2f finish_total_ms=%.2f",
                task_id[:8],
                (t8 - t7) * 1000,
                (t8 - t5) * 1000,
            )
            logger.info("Task %s finished and saved.", task_id)
        except Exception as e:
            if task_id in self.tasks:
                self.tasks[task_id].info['task_status'] = RetCode.TASK_TIMEOUT.value
            logger.exception("_finish_task_impl error: %s", e)

    def check_image(self, task_id: str) -> dict:
        """不再检测缺失块，直接返回成功与空 missing_tiles"""
        try:
            if task_id not in self.tasks:
                result = self.load_data(task_id)
                if result:
                    return result
            ctx = self.tasks.get(task_id)
            if not ctx:
                return {
                    'ret_code': RetCode.CLIENT_ERROR.value,
                    'ret_desc': 'Task ID not found',
                    'reason': 'Task ID not found',
                }
            self._touch_task(task_id)
            return {
                'ret_code': RetCode.API_SUCCESS.value,
                'ret_desc': RetDesc.API_SUCCESS.value,
                'missing_tiles': [],
            }
        except Exception as e:
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': str(e),
                'reason': str(e)
            }

    def task_status(self, task_id: str) -> dict:
        if task_id not in self.tasks:
            result = self.load_data(task_id)
            if result:
                return result
        ctx = self.tasks.get(task_id)
        if not ctx:
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': 'Task ID not found',
                'reason': 'Task ID not found'
        }
        self._touch_task(task_id)
        return {
            'ret_code': RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value,
            'task_status': ctx.info.get('task_status')
        }

    def analyze_slide(self, task_id: str, analyze_names: list) -> dict:
        """
        玻片分析（骨髓玻片增生分析等）。
        cellularity(增生程度) = red_pixel_count / wbc_pixel_count，保留2位小数。
        """
        if task_id not in self.tasks:
            load_result = self.load_data(task_id)
            if load_result is not None:
                return load_result
            if task_id not in self.tasks:
                return {
                    'ret_code': RetCode.CLIENT_ERROR.value,
                    'ret_desc': 'Task ID not found',
                    'reason': 'Task ID not found',
                    'result': {},
                }
        self._touch_task(task_id)
        info = self.tasks[task_id].info
        if not info.get('finished', False):
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': 'Task not completed',
                'reason': 'Task not completed',
                'result': {},
            }
        result = {}
        if 'cellularity' in analyze_names:
            wbc = info.get('wbc_pixel_count', 0) or 0
            red = info.get('red_pixel_count', 0) or 0
            if wbc <= 0:
                result['cellularity'] = None
            else:
                result['cellularity'] = round(red / wbc, 2)
        return {
            'ret_code': RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value,
            'result': result,
        }

    def get_result(self, task_id, roi_xmin, roi_ymin, roi_xmax, roi_ymax, index_offset, request_task_num):
        if task_id not in self.tasks:
            result = self.load_data(task_id)
            if result:
                return result
        ctx = self.tasks[task_id]
        self._touch_task(task_id)
        project = ctx.project
        dpi = ctx.info.get('dpi', 144750)
        layer = project.get_layer(dpi)

        roi_xmin = 0 if roi_xmin is None else int(roi_xmin)
        roi_ymin = 0 if roi_ymin is None else int(roi_ymin)
        roi_xmax = float("inf") if roi_xmax is None else int(roi_xmax)
        roi_ymax = float("inf") if roi_ymax is None else int(roi_ymax)
        offset = max(0, int(index_offset or 0))
        limit = max(0, int(request_task_num or 0))
        roi_key = (roi_xmin, roi_ymin, roi_xmax, roi_ymax)

        roi_cache_key = (task_id, roi_key)
        with self.roi_cache_lock:
            hit = self.roi_cache.get(roi_cache_key)

        if hit is None:
            cells_all = layer.iter_cells_in_roi(roi_xmin, roi_ymin, roi_xmax, roi_ymax, is_Cell=True)
            total = len(cells_all)
            hit = (total, cells_all)
            with self.roi_cache_lock:
                self.roi_cache[roi_cache_key] = hit
        else:
            total, cells_all = hit

        page_cells = cells_all[offset: offset + limit]
        page_dicts = [c.to_dict() for c in page_cells]
        if (offset + limit) >= total:
            with self.roi_cache_lock:
                self.roi_cache.pop(roi_cache_key, None)

        return {
            "ret_code": RetCode.API_SUCCESS.value,
            "ret_desc": RetDesc.API_SUCCESS.value,
            "cell_count": total,
            "cell_list": page_dicts,
            "index_offset": offset,
        }

    def get_task_list_x100(
        self,
        task_id: str,
        task_type: str,
        user_choice_area,
        view_width,
        view_height,
        kwargs: dict | None,
        required_num: dict | None,
    ):
        if task_id not in self.tasks:
            result = self.load_data(task_id)
            if result:
                return result
        ctx = self.tasks.get(task_id)
        if not ctx:
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': 'Task ID not found',
                'reason': 'Task ID not found',
                'result': {},
            }
        self._touch_task(task_id)
        info = ctx.info
        if not info.get('finished', False):
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': 'Task not completed',
                'reason': 'Task not completed',
                'result': {},
            }
        smear_type = (ctx.info or {}).get("smear_type")
        dpi = info.get("dpi")
        heatmap_orientation = info.get('heatmap_orientation', -1)
        if not smear_type:
            smear_type = "BM"

        if not isinstance(kwargs, dict):
            kwargs = {}
        if not isinstance(required_num, dict):
            required_num = {}

        # 去掉分页参数，每次都返回所有任务
        # index_offset = int(kwargs.get("index_offset", 0) or 0)
        # request_task_num = int(kwargs.get("request_task_num", 100) or 100)

        normalized_task_type = (task_type or "").strip().upper()
        allowed_task_types = {"WBC", "MEG", "WBC_MEG", "RBC"}
        if normalized_task_type not in allowed_task_types:
            return {
                "ret_code": RetCode.CLIENT_ERROR.value,
                "ret_desc": f"Invalid task_type: {task_type}. Allowed: {sorted(list(allowed_task_types))}",
                "reason": f"Invalid task_type: {task_type}. Allowed: {sorted(list(allowed_task_types))}",
            }

        def _get_required_int(key: str) -> int | None:
            value = required_num.get(key)
            if value is None:
                return 0
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        required_wbc = _get_required_int("WBC") * 3
        required_meg = _get_required_int("MEG") * 3
        required_rbc = _get_required_int("RBC") * 3

        if smear_type == "BM":
            if normalized_task_type == "WBC":
                if not required_wbc or required_wbc <= 0:
                    return {
                        "ret_code": RetCode.CLIENT_ERROR.value,
                        "ret_desc": "Missing required_num.WBC for BM WBC",
                        "reason": "Missing required_num.WBC for BM WBC",
                    }
            elif normalized_task_type == "MEG":
                if not required_meg or required_meg <= 0:
                    return {
                        "ret_code": RetCode.CLIENT_ERROR.value,
                        "ret_desc": "Missing required_num.MEG for BM MEG",
                        "reason": "Missing required_num.MEG for BM MEG",
                    }
                if not isinstance(kwargs.get("wbc_points"), list) or not kwargs.get("wbc_points"):
                    return {
                        "ret_code": RetCode.CLIENT_ERROR.value,
                        "ret_desc": "Missing kwargs.wbc_points for BM MEG",
                        "reason": "Missing kwargs.wbc_points for BM MEG",
                    }
            elif normalized_task_type == "WBC_MEG":
                if not required_wbc or required_wbc <= 0:
                    return {
                        "ret_code": RetCode.CLIENT_ERROR.value,
                        "ret_desc": "Missing required_num.WBC or required_num.WBC is 0 for BM WBC_MEG",
                        "reason": "Missing required_num.WBC or required_num.WBC is 0 for BM WBC_MEG",
                    }
                # 屏蔽这种情况,允许MGE为0
                if required_meg is None:
                    return {
                        "ret_code": RetCode.CLIENT_ERROR.value,
                        "ret_desc": "Missing required_num.MEG for BM WBC_MEG",
                        "reason": "Missing required_num.MEG for BM WBC_MEG",
                    }
            elif normalized_task_type == "RBC":
                return {
                    "ret_code": RetCode.CLIENT_ERROR.value,
                    "ret_desc": "Invalid combo: BM does not support task_type=RBC",
                    "reason": "Invalid combo: BM does not support task_type=RBC",
                }
        elif smear_type == "PB":
            if normalized_task_type != "WBC":
                return {
                    "ret_code": RetCode.CLIENT_ERROR.value,
                    "ret_desc": f"Invalid combo: PB only supports task_type=WBC, got {task_type}",
                    "reason": f"Invalid combo: PB only supports task_type=WBC, got {task_type}",
                }
            if not required_wbc and required_wbc <= 0:
                return {
                    "ret_code": RetCode.CLIENT_ERROR.value,
                    "ret_desc": "Missing required_num.WBC for PB WBC",
                    "reason": "Missing required_num.WBC for PB WBC",
                }
        else:
            return {
                "ret_code": RetCode.CLIENT_ERROR.value,
                "ret_desc": f"Unsupported smear_type: {smear_type}",
                "reason": f"Unsupported smear_type: {smear_type}",
            }
        project = ctx.project
        if smear_type == "BM" and normalized_task_type in {"WBC", "WBC_MEG"}:
            bm_cfg = BM40Config(
                user_choice_area=user_choice_area,
                target_cell_num_WBC=required_wbc,
                x100_rect_width=int(view_width),
                x100_rect_height=int(view_height),
                heatmap_orientation=heatmap_orientation,
                dpi=dpi,
                View_type="WBC",
                Smear_type=smear_type,
            )
            pipeline = WBCSamplingPipeline(bm_cfg)
            wbc_tasks = pipeline.run(project)
            wbc_task_rects = [task.to_dict() for task in wbc_tasks]

            if normalized_task_type == "WBC":
                final_task_list = wbc_task_rects
            else:
                if required_meg > 0:
                    # WBC_MEG：与 main_meg.py 一致，仅使用 view_type=="WBC" 的视野，转为 [[x,y,w,h]] 再 run_meg
                    wbc_rects_meg: list[list[float]] = []
                    for task in wbc_tasks:
                        if task.view_type != "WBC":
                            continue
                        x = float(task.view_xmin)
                        y = float(task.view_ymin)
                        w = float(task.view_xmax - task.view_xmin)
                        h = float(task.view_ymax - task.view_ymin)
                        wbc_rects_meg.append([x, y, w, h])
                    if not wbc_rects_meg:
                        return {
                            "ret_code": RetCode.CLIENT_ERROR.value,
                            "ret_desc": "从 WBC 结果中未解析到任何 WBC 视野，无法计算 MEG 排序参考。",
                            "reason": "从 WBC 结果中未解析到任何 WBC 视野，无法计算 MEG 排序参考。",
                        }
                    bm_cfg.target_cell_num_MEG = required_meg
                    bm_cfg.View_type = "MEG"
                    try:
                        meg_pipeline = MegSamplingPipeline(bm_cfg)
                        meg_tasks = meg_pipeline.run_meg(
                            project=project, wbc_rects=wbc_rects_meg
                        )
                        meg_task_rects = [task.to_dict() for task in meg_tasks]
                    except Exception as e:
                        logger.exception("MEG roi_selection failed: %s", e)
                        return {
                            "ret_code": RetCode.CLIENT_ERROR.value,
                            "ret_desc": str(e),
                            "reason": str(e),
                        }
                    final_task_list = wbc_task_rects + meg_task_rects
                else:
                    final_task_list = wbc_task_rects

        elif smear_type == "BM" and normalized_task_type == "MEG":
            bm_cfg = BM40Config(
                user_choice_area=user_choice_area,
                target_cell_num_WBC=required_wbc or 0,
                x100_rect_width=int(view_width),
                x100_rect_height=int(view_height),
                heatmap_orientation=heatmap_orientation,
                dpi=dpi,
                Smear_type="BM",
                View_type="MEG"
            )
            bm_cfg.target_cell_num_MEG = required_meg

            wbc_points = kwargs.get("wbc_points") or []
            wbc_rects: list[list[float]] = []
            for p in wbc_points:
                if not isinstance(p, dict):
                    continue
                try:
                    x = float(p.get("x"))
                    y = float(p.get("y"))
                    w = float(p.get("w"))
                    h = float(p.get("h"))
                except (TypeError, ValueError):
                    continue
                wbc_rects.append([x, y, w, h])

            if not wbc_rects:
                return {
                    "ret_code": RetCode.CLIENT_ERROR.value,
                    "ret_desc": "Invalid kwargs.wbc_points: empty or not parseable",
                    "reason": "Invalid kwargs.wbc_points: empty or not parseable",
                }
            try:
                meg_pipeline = MegSamplingPipeline(bm_cfg)
                meg_tasks = meg_pipeline.run_meg(project=project, wbc_rects=wbc_rects)
                final_task_list = [task.to_dict() for task in meg_tasks]
            except Exception as e:
                logger.exception("MEG roi_selection failed: %s", e)
                return {
                    "ret_code": RetCode.CLIENT_ERROR.value,
                    "ret_desc": str(e),
                    "reason": str(e),
                }

        elif smear_type == "PB" and normalized_task_type == "WBC":
            bm_cfg = BM40Config(
                user_choice_area=user_choice_area,
                target_cell_num_WBC=required_wbc,
                x100_rect_width=int(view_width),
                x100_rect_height=int(view_height),
                heatmap_orientation=heatmap_orientation,
                dpi=dpi,
                View_type="WBC",
                Smear_type="PB",
            )
            pipeline = WBCSamplingPipeline(bm_cfg)
            wbc_tasks = pipeline.run(project)
            final_task_list = [task.to_dict() for task in wbc_tasks]
        else:
            return {
                "ret_code": RetCode.CLIENT_ERROR.value,
                "ret_desc": f"roi_selection not implemented for smear_type={smear_type}, task_type={task_type}",
                "reason": f"roi_selection not implemented for smear_type={smear_type}, task_type={task_type}",
            }
        return {
            "ret_code": RetCode.API_SUCCESS.value,
            "ret_desc": RetDesc.API_SUCCESS.value,
            "task_list_num": len(final_task_list),
            "task_list": final_task_list,
        }

    def generate_views(
        self,
        rects: list = None,
        view_width: int = 384,
        view_height: int = 283,
        pad: int = 100,
    ) -> dict:
        """
        根据 rects [[x,y,w,h],...] 生成最少视野框覆盖（set cover）。
        参考 task_wbc.generate_wbc_view_tasks 核心逻辑：rects → centers → solve → rects_x100
        """
        import numpy as np

        if not rects:
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': 'Must provide rects',
                'reason': 'Must provide rects',
                'rects': [],
            }

        arr = np.array(rects, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 4)
        if arr.size == 0 or arr.shape[1] < 4:
            return {
                'ret_code': RetCode.API_SUCCESS.value,
                'ret_desc': RetDesc.API_SUCCESS.value,
                'rects': [],
                'rect_count': 0,
                'rect_count_input': 0,
            }

        # rects [x,y,w,h] → cell_bounds [xmin,ymin,xmax,ymax] → centers
        cell_bounds = np.column_stack([
            arr[:, 0],
            arr[:, 1],
            arr[:, 0] + arr[:, 2],
            arr[:, 1] + arr[:, 3],
        ])
        centers = 0.5 * (cell_bounds[:, 0:2] + cell_bounds[:, 2:4])

        x_min_all, y_min_all = cell_bounds[:, 0:2].min(axis=0) - pad
        x_max_all, y_max_all = cell_bounds[:, 2:4].max(axis=0) + pad
        bounding_rect = np.array([
            x_min_all, y_min_all,
            x_max_all - x_min_all + 1,
            y_max_all - y_min_all + 1,
        ], dtype=np.int32)

        params = SetCoverSolverParameter(
            rect_width=view_width,
            rect_height=view_height,
        )
        try:
            rects_x100 = solve(centers, bounding_rect, params)
        except Exception as e:
            logger.exception("Setcover solve failed: %s", e)
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': str(e),
                'reason': str(e),
                'rects': [],
            }

        return {
            'ret_code': RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value,
            'rects': rects_x100.tolist(),
            'rect_count': len(rects_x100),
            'rect_count_input': len(rects),
        }

    def get_task_result_x100(self, task_id, image_file, target_cell_types, dpi,
                             edge_cell_filter, smear_type,
                             position_xmin, position_ymin, position_xmax, position_ymax):
        """
        细胞图像分析。支持两种模式：
        - 任务模式：task_id + position 必填，结果保存到项目
        - 单张识别：无 task_id，dpi+algorithm_types 必填，直接返回推理结果
        """
        image_bytes = image_file.read()
        edge_cell_filter = _parse_edge_cell_filter_flag(edge_cell_filter)

        ok, err = validate_combo(int(dpi), smear_type, target_cell_types or "")
        if not ok:
            return {
                "ret_code": RetCode.CLIENT_ERROR.value,
                "ret_desc": err,
                "reason": err,
            }
        warning = err

        try:
            result = infer(
                image_bytes,
                dpi=int(dpi),
                smear_type=smear_type,
                algorithm_types=target_cell_types or "",
            )
        except Exception as e:
            logger.exception("Triton infer failed: %s", e)
            return {
                "ret_code": RetCode.CLIENT_ERROR.value,
                "ret_desc": str(e),
                "reason": str(e),
            }
        cells = result.get("cells", [])
        cell_list = result.get("cell_list", [])
        if not cell_list and cells:
            cell_list = [{
                'cell_xmin': c.cell_xmin, 'cell_ymin': c.cell_ymin,
                'cell_xmax': c.cell_xmax, 'cell_ymax': c.cell_ymax,
                'tops': [{'cell_type': c.cell_type, 'cell_type_name': c.cell_type_name,
                          'class_confidence': c.class_confidence, 'bbox_confidence': c.bbox_confidence}]
            } for c in cells]
        # 当DPI为714756 +-10%的时候不过滤边缘细胞，因为模型自带过滤功能
        if int(dpi) in (714756, 786432):
            edge_cell_filter = False
        if edge_cell_filter and cell_list:
            try:
                with Image.open(BytesIO(image_bytes)) as im:
                    tw_img, th_img = im.size
                cell_list = filter_cell_dicts_edge_incomplete(
                    cell_list, tw_img, th_img
                )
            except Exception as e:
                logger.warning(
                    "get_task_result_x100: edge_cell_filter skipped (image decode failed): %s",
                    e,
                )
        response = {
            "ret_code": RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value,
            'cell_count': len(cell_list),
            'cell_list': cell_list
        }
        if warning:
            response['warning'] = warning
        return response
