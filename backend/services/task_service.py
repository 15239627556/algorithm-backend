import json
import os
import threading
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, Any

from backend.tools.MESSAGE_DICT import RetCode, RetDesc
from backend.tools.public_methods import thread_decorator, upload_folder
from backend.tools.combo_validator import validate_combo
from backend.tools.json_safe_writer import serialize_non_json_fields
from backend.tools.dedup_cells_across_tiles import dedup_cells_across_tiles
from project.smear_project import SmearProject
from project.cells import Cell
from project.triton_client import infer, get_model_by_dpi
from project.model_control import warmup_model
from algorithms.SelectArea.main import *

logger = logging.getLogger(__name__)


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


@dataclass
class TaskContext:
    """单任务上下文，聚合 project、info、lock（已去掉 num_rows/num_cols 与 grid）"""
    project: SmearProject
    info: dict
    lock: threading.Lock = field(default_factory=threading.Lock)


class TaskService:
    def __init__(self):
        self.tasks: Dict[str, TaskContext] = {}
        self.roi_cache: Dict[str, Dict[tuple, Any]] = {}
        self.roi_cache_lock = threading.Lock()
        self.project_x100: Dict[str, list] = {}

    def load_data(self, task_id):
        if task_id in self.tasks:
            return None
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, f"{task_id}.json")
        if not os.path.exists(file_path):
            return {"ret_code": RetCode.CLIENT_ERROR.value,
                    "ret_desc": RetDesc.CLIENT_ERROR.value,
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
        self.tasks[task_id] = TaskContext(project=project, info=info)

    def create_task(self, task_info: dict) -> dict:
        dpi = task_info.get('dpi')
        smear_type = task_info.get('smear_type')
        target_cell_types = task_info.get('target_cell_types')
        ok, err = validate_combo(dpi, smear_type, target_cell_types, allow_empty_types=False)
        if not ok:
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': RetDesc.CLIENT_ERROR.value,
                'reason': err,
            }
        task_id = uuid.uuid4().hex
        task_info['smear_type'] = task_info.get('smear_type', 'BM')
        task_info['task_status'] = RetCode.TASK_RUNNING.value
        task_info['finished'] = False
        task_info['matcher'] = {}
        project = SmearProject(smear_type=task_info['smear_type'])
        project.add_layer(task_info['dpi'])

        self.tasks[task_id] = TaskContext(project=project, info=task_info)
        _save_task_info(task_id, task_info)
        model_name = get_model_by_dpi(dpi, smear_type=task_info.get('smear_type', 'BM'), algorithm_types=task_info.get('target_cell_types', ''))
        warmup_model(model_name)
        logger.info('创建任务成功：%s', task_id)

        return {
            'task_id': task_id,
            'ret_code': RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value
        }

    def update_coordinates(self, task_id, tiles_msg):
        if task_id not in self.tasks:
            result = self.load_data(task_id)
            if result:
                return result
        ctx = self.tasks[task_id]
        project = ctx.project
        dpi = ctx.info.get('dpi', 144750)
        layer = project.get_layer(dpi)
        project_info = ctx.info
        with ctx.lock:
            matcher = project_info.get('matcher')
            if not matcher:
                matcher = {}
                project_info['matcher'] = matcher
        failed_tiles = []
        for tile_info in tiles_msg:
            row_index = int(tile_info['row_index'])
            col_index = int(tile_info['col_index'])
            position_x = int(tile_info['position_x'])
            position_y = int(tile_info['position_y'])
            image_uid = matcher.get((row_index, col_index))
            if image_uid is None:
                tiles = layer.iter_tiles()
                flag = 0
                for tile in tiles:
                    if tile.meta.get('row_index') == row_index and tile.meta.get('col_index') == col_index:
                        tile.x = position_x
                        tile.y = position_y
                        flag = 1
                        break
                if flag == 0:
                    tile_info['reason'] = 'tile not found'
                    failed_tiles.append(tile_info)
            else:
                try:
                    tile = layer.get_tile(image_uid)
                    tile.x = position_x
                    tile.y = position_y
                except Exception as e:
                    tile_info['reason'] = str(e)
                    failed_tiles.append(tile_info)
        self._finish_task_sync(task_id)
        return {
            'ret_code': RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value,
            'failed_tiles': failed_tiles
        }

    def upload_image(self, task_id, row_index, col_index, tile_image):
        """任务模式：上传拼图块到指定任务，DPI/smear_type 从 task_info 取"""
        image_bytes = tile_image.read()

        if row_index is None or col_index is None:
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': RetDesc.CLIENT_ERROR.value,
                'reason': 'Task mode requires row_index and col_index',
            }
        if task_id not in self.tasks:
            result = self.load_data(task_id)
            if result:
                return result
            if task_id not in self.tasks:
                return {
                    'ret_code': RetCode.CLIENT_ERROR.value,
                    'ret_desc': RetDesc.CLIENT_ERROR.value,
                    'reason': 'Task ID not found',
                }

        ctx = self.tasks[task_id]
        row_index, col_index = int(row_index), int(col_index)
        with ctx.lock:
            matcher = ctx.info.get('matcher')
            if not matcher:
                matcher = {}
                ctx.info['matcher'] = matcher
        image_uid = matcher.get((row_index, col_index))
        project = ctx.project
        dpi = ctx.info.get('dpi', 144750)
        layer = project.get_layer(dpi)
        task_info = ctx.info
        file_name = tile_image.filename

        smear_type = task_info.get('smear_type', 'BM')
        target_cell_types = task_info.get('target_cell_types', '')

        if image_uid is None:
            tile = layer.add_tile(
                x=None, y=None,
                w=task_info['tile_width'],
                h=task_info['tile_height'],
                image_data=None,
                image_path=file_name,
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
            result = infer(
                image_bytes,
                dpi=int(dpi),
                smear_type=smear_type,
                algorithm_types=target_cell_types or "",
            )
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
                'ret_desc': RetDesc.CLIENT_ERROR.value,
                'reason': str(e),
            }

    def _finish_task_sync(self, task_id: str):
        """所有 tile 已上传并推理完成，同步去重并保存"""
        try:
            ctx = self.tasks.get(task_id)
            if not ctx:
                return
            project = ctx.project
            dpi = ctx.info.get('dpi', 144750)
            layer = project.get_layer(dpi)
            tiles = layer.iter_tiles()
            tiles = dedup_cells_across_tiles(tiles)
            for one_tile in tiles:
                layer.tiles[one_tile.image_uid] = one_tile
            project.save_json(os.path.join(upload_folder, f"{task_id}.json"))
            ctx.info['task_status'] = RetCode.TASK_FINISHED.value
            ctx.info['finished'] = True
            _save_task_info(task_id, ctx.info)
            logger.info("Task %s finished and saved.", task_id)
        except Exception as e:
            if task_id in self.tasks:
                self.tasks[task_id].info['task_status'] = RetCode.TASK_TIMEOUT.value
            logger.exception("_finish_task_sync error: %s", e)

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
                    'ret_desc': RetDesc.CLIENT_ERROR.value,
                    'reason': 'Task ID not found',
                }
            return {
                'ret_code': RetCode.API_SUCCESS.value,
                'ret_desc': RetDesc.API_SUCCESS.value,
                'missing_tiles': [],
            }
        except Exception as e:
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': RetDesc.CLIENT_ERROR.value,
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
                'ret_desc': RetDesc.CLIENT_ERROR.value,
'reason': 'Task ID not found'
        }
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
                    'ret_desc': RetDesc.CLIENT_ERROR.value,
                    'reason': 'Task ID not found',
                    'result': {},
                }
        info = self.tasks[task_id].info
        if not info.get('finished', False):
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': RetDesc.CLIENT_ERROR.value,
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

        with self.roi_cache_lock:
            task_cache = self.roi_cache.setdefault(task_id, {})
            hit = task_cache.get(roi_key)

        if hit is None:
            cells_all = layer.iter_cells_in_roi(roi_xmin, roi_ymin, roi_xmax, roi_ymax, is_Cell=True)
            total = len(cells_all)
            hit = (total, cells_all)
            with self.roi_cache_lock:
                task_cache[roi_key] = hit
                if len(task_cache) > 8:
                    task_cache.pop(next(iter(task_cache)))
        else:
            total, cells_all = hit

        total, cells_all = hit
        page_cells = cells_all[offset: offset + limit]
        page_dicts = [c.to_dict() for c in page_cells]
        if (offset + limit) >= total:
            with self.roi_cache_lock:
                task_cache = self.roi_cache.get(task_id)
                if task_cache and roi_key in task_cache:
                    task_cache.pop(roi_key)

        return {
            "ret_code": RetCode.API_SUCCESS.value,
            "ret_desc": RetDesc.API_SUCCESS.value,
            "cell_count": total,
            "cell_list": page_dicts,
            "index_offset": offset,
        }

    def get_task_list_x100(self, task_id, user_choice_area, view_width, view_height, target_list,
                           index_offset, request_task_num):
        if task_id not in self.tasks:
            result = self.load_data(task_id)
            if result:
                return result
        target_num_WBC = None
        target_num_MEG = None
        for one in target_list:
            if one['smear_type'] == 'BM' and "WBC" in one['target_cell_type']:
                target_num_WBC = one['count']
            if one['smear_type'] == 'BM' and "MEG" in one['target_cell_type']:
                target_num_MEG = one['count']
        if target_num_WBC is None:
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': RetDesc.CLIENT_ERROR.value,
                'reason': 'Invalid params: missing BM WBC target count'
            }
        project = self.tasks[task_id].project
        area_str = json.dumps(user_choice_area or {}, sort_keys=True) if isinstance(user_choice_area, dict) else str(user_choice_area)
        x100_key = f"{task_id}_{area_str}_{view_width}_{view_height}_{target_num_WBC}"
        if not self.project_x100.get(x100_key):
            bm_cfg = BM40Config(user_choice_area=user_choice_area,
                                target_cell_num=target_num_WBC,
                                x100_rect_width=view_width,
                                x100_rect_height=view_height)
            pipeline = WBCSamplingPipeline(bm_cfg)
            final_task_list = pipeline.run(project)
            final_task_list = [task.to_dict() for task in final_task_list]
            self.project_x100[x100_key] = final_task_list
        else:
            final_task_list = self.project_x100[x100_key]
        task_list = final_task_list[index_offset:index_offset + request_task_num]
        if (index_offset + request_task_num) >= len(final_task_list):
            self.project_x100.pop(x100_key, None)
        return {
            'ret_code': RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value,
            'task_list_num': len(final_task_list),
            'task_list': task_list
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

        ok, err = validate_combo(int(dpi), smear_type, target_cell_types or "")
        if not ok:
            return {
                "ret_code": RetCode.CLIENT_ERROR.value,
                "ret_desc": RetDesc.CLIENT_ERROR.value,
                "reason": err,
            }

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
                "ret_desc": RetDesc.CLIENT_ERROR.value,
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
        return {
            "ret_code": RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value,
            'cell_count': len(cell_list),
            'cell_list': cell_list
        }
