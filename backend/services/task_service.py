import os
import shutil
import threading
from io import BytesIO
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import orjson

from backend.tools.MESSAGE_DICT import RetCode, RetDesc
from backend.tools.public_methods import thread_decorator, upload_folder
from backend.tools.combo_validator import validate_combo, _get_dpi_bucket, _parse_cell_types
from backend.tools.json_safe_writer import serialize_non_json_fields
from backend.tools.filter_edge_incomplete_cells import (
    filter_cell_dicts_edge_incomplete,
    filter_cell_dicts_edge_elongated_1pct,
    filter_cell_dicts_small_wbc_714756,
    filter_edge_incomplete_cells,
)
from PIL import Image

from project.smear_project import SmearProject
from project.roi_store import RoiDataset
from project.cells import Cell
from backend.tools.triton_client import infer, get_model_by_dpi, resolve_triton_route
from backend.tools.x100_image_infer import (
    infer_x100_on_bgr,
    map_cell_list_from_scaled,
    map_cells_from_scaled,
    prepare_x100_bgr,
)
from backend.tools.model_control import warmup_model, ensure_model_loaded
from algorithms.SelectArea.main_wbc import *
from algorithms.SelectArea.main_meg import *
from algorithms.SelectArea.setcover import solve, SetCoverSolverParameter
from algorithms.SelectArea.dedup_cells_across_tiles import dedup_cells_across_tiles_per_type


logger = logging.getLogger(__name__)

# 进程内 task_info 缓存：仅 upload_image 热路径使用，其余接口以磁盘落盘为准
_task_info_cache: dict[str, dict] = {}
_task_info_cache_lock = threading.Lock()

# 进程内 ROI 数据集缓存：/roi_selection 热路径，每个 uvicorn worker 独立一份
_roi_dataset_cache: dict[str, tuple[RoiDataset, float]] = {}
_roi_dataset_cache_lock = threading.Lock()
_ROI_DATASET_CACHE_TTL_SEC = 5 * 60  # 5 min

# 进程内 SmearProject 缓存：/roi_selection 等选区 pipeline 热路径
_project_cache: dict[str, tuple[SmearProject, float]] = {}
_project_cache_lock = threading.Lock()


def _roi_dataset_cache_ttl_sec() -> float:
    raw = os.environ.get("ROI_DATASET_CACHE_TTL_SEC", str(_ROI_DATASET_CACHE_TTL_SEC)).strip()
    try:
        ttl = float(raw)
    except ValueError:
        ttl = float(_ROI_DATASET_CACHE_TTL_SEC)
    return max(0.0, ttl)


def _roi_cache_get(task_id: str) -> RoiDataset | None:
    """读取本进程 ROI 缓存；过期则淘汰；命中则滑动续期 TTL。"""
    now = time.monotonic()
    ttl = _roi_dataset_cache_ttl_sec()
    with _roi_dataset_cache_lock:
        entry = _roi_dataset_cache.get(task_id)
        if entry is None:
            return None
        roi, expires_at = entry
        if now >= expires_at:
            del _roi_dataset_cache[task_id]
            return None
        _roi_dataset_cache[task_id] = (roi, now + ttl)
        return roi


def _roi_cache_set(task_id: str, roi: RoiDataset) -> None:
    """写入本进程 ROI 缓存（TTL 默认 5min，可用 ROI_DATASET_CACHE_TTL_SEC 覆盖）。"""
    ttl = _roi_dataset_cache_ttl_sec()
    expires_at = time.monotonic() + ttl
    with _roi_dataset_cache_lock:
        _roi_dataset_cache[task_id] = (roi, expires_at)
    logger.info(
        "roi cache_set pid=%s task_id=%s ttl_sec=%.0f cells=%d tiles=%d",
        os.getpid(),
        task_id[:8],
        ttl,
        roi.cells.size,
        len(roi.tiles),
    )


def _project_cache_get(task_id: str) -> SmearProject | None:
    """读取本进程 SmearProject 缓存；过期则淘汰；命中则滑动续期 TTL。"""
    now = time.monotonic()
    ttl = _roi_dataset_cache_ttl_sec()
    with _project_cache_lock:
        entry = _project_cache.get(task_id)
        if entry is None:
            return None
        project, expires_at = entry
        if now >= expires_at:
            del _project_cache[task_id]
            return None
        _project_cache[task_id] = (project, now + ttl)
        return project


def _project_cache_set(task_id: str, project: SmearProject) -> None:
    """写入本进程 SmearProject 缓存（TTL 与 ROI 缓存一致）。"""
    ttl = _roi_dataset_cache_ttl_sec()
    expires_at = time.monotonic() + ttl
    with _project_cache_lock:
        _project_cache[task_id] = (project, expires_at)
    logger.info(
        "project cache_set pid=%s task_id=%s ttl_sec=%.0f layers=%d",
        os.getpid(),
        task_id[:8],
        ttl,
        len(project.layers),
    )


def _async_finish_after_update_coordinates() -> bool:
    """为 True 时 update_coordinates 立即返回，合并坐标/去重/过滤/落盘均在后台执行。"""
    v = os.environ.get("UPDATE_COORDINATES_ASYNC_FINISH", "1").strip().lower()
    return v not in ("0", "false", "no", "off", "")


def _run_in_background(fn, *, name: str = "bg") -> None:
    """后台执行 CPU/阻塞任务，接口立即返回（daemon 线程）。"""
    th = threading.Thread(target=fn, name=name, daemon=True)
    th.start()

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


def _task_tiles_dir(task_id: str) -> str:
    """单块推理结果目录：uploads/{task_id}/tiles/{row}_{col}.json"""
    return os.path.join(upload_folder, task_id, "tiles")


def _tile_result_path(task_id: str, row_index: int, col_index: int) -> str:
    return os.path.join(_task_tiles_dir(task_id), f"{row_index}_{col_index}.json")


def _save_task_info(task_id: str, info: dict) -> None:
    """持久化 task_info（orjson，适配多进程共享）。"""
    os.makedirs(upload_folder, exist_ok=True)
    path = _task_info_path(task_id)
    tmp = f"{path}.{os.getpid()}.tmp"
    payload = {k: v for k, v in info.items() if k != "matcher"}
    with open(tmp, "wb") as f:
        f.write(orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY))
    os.replace(tmp, path)
    with _task_info_cache_lock:
        _task_info_cache[task_id] = dict(payload)


def _load_task_info_from_disk(task_id: str) -> dict | None:
    """从磁盘加载 task_info（不读缓存）。"""
    path = _task_info_path(task_id)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def _load_task_info_cached(task_id: str) -> dict | None:
    """upload_image 专用：优先缓存，未命中再读盘并回填缓存。"""
    with _task_info_cache_lock:
        cached = _task_info_cache.get(task_id)
        if cached is not None:
            return dict(cached)
    data = _load_task_info_from_disk(task_id)
    if data is not None:
        with _task_info_cache_lock:
            _task_info_cache[task_id] = dict(data)
    return data


def _write_tile_result(task_id: str, row_index: int, col_index: int, payload: dict) -> str:
    """原子写入单块结果 JSON（orjson）。"""
    path = _tile_result_path(task_id, row_index, col_index)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY))
    os.replace(tmp, path)
    return path


def _read_tile_result(task_id: str, row_index: int, col_index: int) -> dict | None:
    path = _tile_result_path(task_id, row_index, col_index)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def _cells_to_dicts(cells) -> list[dict]:
    out = []
    for c in cells or []:
        if hasattr(c, "to_dict"):
            out.append(c.to_dict())
        elif isinstance(c, dict):
            out.append(c)
    return out


def _bbox_key(d: dict) -> tuple[int, int, int, int]:
    return (
        int(d["cell_xmin"]),
        int(d["cell_ymin"]),
        int(d["cell_xmax"]),
        int(d["cell_ymax"]),
    )


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


def _task_project_path(task_id: str) -> str:
    return os.path.join(upload_folder, f"{task_id}.json")


def _task_project_pickle_path(task_id: str) -> str:
    """选区旧版 pickle 路径（兼容回退）。"""
    return os.path.join(upload_folder, f"{task_id}.roi.pkl")


def _task_roi_npz_path(task_id: str) -> str:
    """选区 numpy 快路径（/roi_selection 优先使用）。"""
    return os.path.join(upload_folder, f"{task_id}.roi.npz")


def _task_not_found_error(**extra) -> dict:
    out = {
        'ret_code': RetCode.CLIENT_ERROR.value,
        'ret_desc': 'Task ID not found',
        'reason': 'Task ID not found',
    }
    out.update(extra)
    return out


def _require_task_info(task_id: str) -> tuple[dict | None, dict | None]:
    """从磁盘加载 task info（不读缓存，任务完成以落盘为准）。返回 (info, error_response)。"""
    info = _load_task_info_from_disk(task_id)
    if info is None:
        return None, _task_not_found_error()
    return info, None


def _require_task_info_for_upload(task_id: str) -> tuple[dict | None, dict | None]:
    """upload_image 专用：允许读缓存。返回 (info, error_response)。"""
    info = _load_task_info_cached(task_id)
    if info is None:
        return None, _task_not_found_error()
    return info, None


def _require_project(task_id: str) -> tuple[SmearProject | None, dict | None, dict | None]:
    """
    选区 pipeline 热路径：进程缓存 → roi.npz → roi.pkl → 大 JSON。
    返回 (project, info, error_response)。
    """
    info, err = _require_task_info(task_id)
    if err:
        return None, None, err

    cached = _project_cache_get(task_id)
    if cached is not None:
        logger.debug("project cache_hit pid=%s task_id=%s", os.getpid(), task_id[:8])
        return cached, info, None

    npz_path = _task_roi_npz_path(task_id)
    if os.path.exists(npz_path):
        try:
            t0 = time.time()
            roi = _roi_cache_get(task_id)
            if roi is None:
                roi = RoiDataset.load(npz_path)
                _roi_cache_set(task_id, roi)
            project = roi.to_project()
            _project_cache_set(task_id, project)
            logger.info(
                "project load_npz pid=%s task_id=%s ms=%.2f cells=%d tiles=%d",
                os.getpid(),
                task_id[:8],
                (time.time() - t0) * 1000,
                roi.cells.size,
                len(roi.tiles),
            )
            return project, info, None
        except Exception as e:
            logger.warning(
                "project npz load failed task_id=%s path=%s err=%s; fallback",
                task_id[:8],
                npz_path,
                e,
            )

    pkl_path = _task_project_pickle_path(task_id)
    if os.path.exists(pkl_path):
        try:
            t0 = time.time()
            project = SmearProject.load_pickle(pkl_path)
            _project_cache_set(task_id, project)
            logger.info(
                "project load_pickle pid=%s task_id=%s ms=%.2f",
                os.getpid(),
                task_id[:8],
                (time.time() - t0) * 1000,
            )
            return project, info, None
        except Exception as e:
            logger.warning(
                "project pickle load failed task_id=%s path=%s err=%s; fallback json",
                task_id[:8],
                pkl_path,
                e,
            )

    path = _task_project_path(task_id)
    if not os.path.exists(path):
        return None, info, {
            'ret_code': RetCode.CLIENT_ERROR.value,
            'ret_desc': 'Task project not found (not finished?)',
            'reason': 'Task project not found (not finished?)',
        }
    try:
        t0 = time.time()
        project = SmearProject.load_json(path)
        _project_cache_set(task_id, project)
        logger.info(
            "project load_json pid=%s task_id=%s ms=%.2f",
            os.getpid(),
            task_id[:8],
            (time.time() - t0) * 1000,
        )
        return project, info, None
    except Exception as e:
        logger.exception("project json load failed task_id=%s: %s", task_id[:8], e)
        return None, info, {
            'ret_code': RetCode.CLIENT_ERROR.value,
            'ret_desc': f'Failed to load task project: {e}',
            'reason': str(e),
        }


def _require_roi_dataset(task_id: str) -> tuple[RoiDataset | None, dict | None, dict | None]:
    """
    ROI 热路径：本 worker 进程缓存（TTL 5min）→ roi.npz → roi.pkl → 大 JSON。
    用于 /get_task_result；多 worker 下各进程独立加载并缓存。
    返回 (RoiDataset, info, error_response)。
    """
    info, err = _require_task_info(task_id)
    if err:
        return None, None, err

    cached = _roi_cache_get(task_id)
    if cached is not None:
        logger.debug("roi cache_hit pid=%s task_id=%s", os.getpid(), task_id[:8])
        return cached, info, None

    dpi = info.get("dpi", 144750)

    npz_path = _task_roi_npz_path(task_id)
    if os.path.exists(npz_path):
        try:
            t0 = time.time()
            roi = RoiDataset.load(npz_path)
            logger.info(
                "roi load_npz pid=%s task_id=%s ms=%.2f cells=%d tiles=%d",
                os.getpid(),
                task_id[:8],
                (time.time() - t0) * 1000,
                roi.cells.size,
                len(roi.tiles),
            )
            _roi_cache_set(task_id, roi)
            return roi, info, None
        except Exception as e:
            logger.warning(
                "roi npz load failed task_id=%s path=%s err=%s; fallback",
                task_id[:8],
                npz_path,
                e,
            )

    pkl_path = _task_project_pickle_path(task_id)
    if os.path.exists(pkl_path):
        try:
            t0 = time.time()
            project = SmearProject.load_pickle(pkl_path)
            roi = RoiDataset.from_project(project, dpi)
            logger.info(
                "roi load_pickle+convert pid=%s task_id=%s ms=%.2f",
                os.getpid(),
                task_id[:8],
                (time.time() - t0) * 1000,
            )
            _roi_cache_set(task_id, roi)
            return roi, info, None
        except Exception as e:
            logger.warning(
                "roi pickle load failed task_id=%s path=%s err=%s; fallback json",
                task_id[:8],
                pkl_path,
                e,
            )

    project, info, err = _require_project(task_id)
    if err:
        return None, info, err
    try:
        t0 = time.time()
        roi = RoiDataset.from_project(project, dpi)
        logger.info(
            "roi load_json+convert pid=%s task_id=%s ms=%.2f",
            os.getpid(),
            task_id[:8],
            (time.time() - t0) * 1000,
        )
        _roi_cache_set(task_id, roi)
        return roi, info, None
    except Exception as e:
        logger.exception("roi json convert failed task_id=%s: %s", task_id[:8], e)
        return None, info, {
            "ret_code": RetCode.CLIENT_ERROR.value,
            "ret_desc": f"Failed to load ROI dataset: {e}",
            "reason": str(e),
        }


class TaskService:
    """无状态服务类：方法仅通过磁盘交互，适配多进程/多 worker。"""

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
        task_info['wbc_pixel_count'] = 0
        task_info['red_pixel_count'] = 0
        os.makedirs(_task_tiles_dir(task_id), exist_ok=True)
        _save_task_info(task_id, task_info)

        # 40 倍平扫：创建时经 ensure_model_loaded 预热全部 Triton 端点（互斥/LRU）
        model_name, model_warning = warmup_model(
            dpi,
            smear_type=task_info.get('smear_type', 'BM'),
            algorithm_types=task_info.get('target_cell_types', ''),
            all_gpus=True,
        )
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

    def _build_project_from_tiles(
        self, task_id: str, tiles_msg: list, info: dict
    ) -> tuple[SmearProject, dict, list]:
        """按 tiles_msg 读分块 json 合并为 SmearProject，并汇总像素计数。"""
        t0 = time.time()
        dpi = int(info.get('dpi', 144750))
        smear_type = info.get('smear_type', 'BM')
        project = SmearProject(smear_type=smear_type)
        layer = project.add_layer(dpi)

        failed_tiles = []
        wbc_pixel_count = 0
        red_pixel_count = 0
        tile_w = int(info.get('tile_width', 2448))
        tile_h = int(info.get('tile_height', 2048))

        for tile_info in tiles_msg or []:
            try:
                row_index = int(tile_info['row_index'])
                col_index = int(tile_info['col_index'])
                position_x = int(tile_info['position_x'])
                position_y = int(tile_info['position_y'])
            except (KeyError, TypeError, ValueError) as e:
                bad = dict(tile_info) if isinstance(tile_info, dict) else {'raw': tile_info}
                bad['reason'] = f'invalid tiles_msg item: {e}'
                failed_tiles.append(bad)
                continue

            data = _read_tile_result(task_id, row_index, col_index)
            if data is None:
                tile_info = dict(tile_info)
                tile_info['reason'] = 'tile result json not found'
                failed_tiles.append(tile_info)
                continue

            tw = int(data.get('w', tile_w))
            th = int(data.get('h', tile_h))
            image_path = data.get('image_path') or f"{row_index}_{col_index}.jpg"
            tile = layer.add_tile(
                x=position_x,
                y=position_y,
                w=tw,
                h=th,
                image_data=None,
                image_path=image_path,
                extra_meta={
                    'row_index': row_index,
                    'col_index': col_index,
                    'scores': data.get('scores') or [],
                },
            )
            cells = [Cell.from_dict(c) for c in (data.get('cells') or [])]
            if cells:
                tile.add_cells(cells)
            wbc_pixel_count += int(data.get('wbc_pixel_count') or 0)
            red_pixel_count += int(data.get('red_pixel_count') or 0)

        info['wbc_pixel_count'] = wbc_pixel_count
        info['red_pixel_count'] = red_pixel_count
        logger.info(
            "update_coordinates merge task_id=%s merge_ms=%.2f tiles=%s failed=%s",
            task_id[:8],
            (time.time() - t0) * 1000,
            len(tiles_msg or []),
            len(failed_tiles),
        )
        return project, info, failed_tiles

    def _merge_and_finish(self, task_id: str, tiles_msg: list, info: dict) -> list:
        """合并坐标并 finish；供同步/后台线程共用。"""
        try:
            project, info, failed_tiles = self._build_project_from_tiles(task_id, tiles_msg, info)
            if failed_tiles:
                logger.warning(
                    "update_coordinates failed_tiles task_id=%s count=%s sample=%s",
                    task_id[:8],
                    len(failed_tiles),
                    failed_tiles[:3],
                )
            self._finish_task_impl(task_id, project, info)
            return failed_tiles
        except Exception as e:
            info['task_status'] = RetCode.TASK_TIMEOUT.value
            try:
                _save_task_info(task_id, info)
            except Exception:
                logger.exception("failed to persist task status after merge error")
            logger.exception("_merge_and_finish error task_id=%s: %s", task_id[:8], e)
            return []

    def update_coordinates(self, task_id, tiles_msg):
        """按 tiles_msg 读取分块 json，写入全局坐标后合并为大项目并 finish。"""
        info, err = _require_task_info(task_id)
        if err:
            return err

        tiles_msg = list(tiles_msg or [])
        async_finish = _async_finish_after_update_coordinates()
        if async_finish:
            info_snapshot = dict(info)
            _run_in_background(
                lambda: self._merge_and_finish(task_id, tiles_msg, info_snapshot),
                name=f"merge-finish-{task_id[:8]}",
            )
            return {
                'ret_code': RetCode.API_SUCCESS.value,
                'ret_desc': RetDesc.API_SUCCESS.value,
                'failed_tiles': [],
                'finish_in_background': True,
            }

        failed_tiles = self._merge_and_finish(task_id, tiles_msg, info)
        return {
            'ret_code': RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value,
            'failed_tiles': failed_tiles,
        }

    def upload_image(self, task_id, row_index, col_index, tile_image):
        """
        任务模式上传拼图块：推理后将结果以 orjson 写入磁盘，并实时返回 cell_list。
        """
        filename = f"{row_index}_{col_index}.jpg"
        image_bytes = tile_image.read() if hasattr(tile_image, "read") else tile_image
        if row_index is None or col_index is None:
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': 'Task mode requires row_index and col_index',
                'reason': 'Task mode requires row_index and col_index',
            }

        info, err = _require_task_info_for_upload(task_id)
        if err:
            return err

        row_index, col_index = int(row_index), int(col_index)
        dpi = int(info.get('dpi', 144750))
        smear_type = info.get('smear_type', 'BM')
        target_cell_types = info.get('target_cell_types', '')

        try:
            result = infer(
                image_bytes,
                dpi=dpi,
                smear_type=smear_type,
                algorithm_types=target_cell_types,
                filename=filename,
            )
            cells = result.get("cells") or []
            scores = _ensure_json_serializable(result.get("scores", []))
            cell_list = result.get("cell_list") or []
            if not cell_list and cells:
                cell_list = [{
                    'cell_xmin': c.cell_xmin, 'cell_ymin': c.cell_ymin,
                    'cell_xmax': c.cell_xmax, 'cell_ymax': c.cell_ymax,
                    'tops': [{'cell_type': c.cell_type, 'cell_type_name': c.cell_type_name,
                              'class_confidence': c.class_confidence, 'bbox_confidence': c.bbox_confidence}]
                } for c in cells]
            tile_w = int(info.get('tile_width', 2448))
            tile_h = int(info.get('tile_height', 2048))
            cell_list = filter_cell_dicts_edge_elongated_1pct(
                cell_list, tile_w, tile_h
            )
            keep_bboxes = {_bbox_key(d) for d in cell_list}
            cells_payload = [
                d for d in _cells_to_dicts(cells) if _bbox_key(d) in keep_bboxes
            ]
            payload = {
                'cells': cells_payload,
                'scores': scores,
                'wbc_pixel_count': int(result.get("wbc_pixel_count") or 0),
                'red_pixel_count': int(result.get("red_pixel_count") or 0),
            }
            _write_tile_result(task_id, row_index, col_index, payload)
            return {
                'ret_code': RetCode.API_SUCCESS.value,
                'ret_desc': RetDesc.API_SUCCESS.value,
                'cell_list': cell_list,
            }
        except Exception as e:
            logger.exception("Triton inference failed for task %s: %s", task_id, e)
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': str(e),
                'reason': str(e),
            }

    def _finish_task_impl(self, task_id: str, project: SmearProject, info: dict) -> None:
        """去重、过滤、落盘为大 JSON + 更新 info（后台可跑，不依赖内存任务表）。"""
        t5 = time.time()
        try:
            dpi = info.get('dpi', 144750)
            layer = project.get_layer(dpi)
            tiles = dedup_cells_across_tiles_per_type(layer.iter_tiles())
            t6 = time.time()
            logger.info(
                "dedup_cells_across_tiles task_id=%s ms=%.2f",
                task_id[:8],
                (t6 - t5) * 1000,
            )
            for one_tile in tiles:
                layer.tiles[one_tile.image_uid] = one_tile
            t9 = time.time()
            logger.info(
                "filter_edge_incomplete_cells task_id=%s ms=%.2f",
                task_id[:8],
                (t9 - t6) * 1000,
            )
            project.save_json(_task_project_path(task_id))
            t7 = time.time()
            logger.info(
                "save_json task_id=%s ms=%.2f",
                task_id[:8],
                (t7 - t9) * 1000,
            )
            try:
                dpi = info.get("dpi", 144750)
                RoiDataset.from_project(project, dpi).save(_task_roi_npz_path(task_id))
                logger.info(
                    "save_roi_npz task_id=%s ms=%.2f",
                    task_id[:8],
                    (time.time() - t7) * 1000,
                )
            except Exception as e:
                logger.warning(
                    "save_roi_npz failed task_id=%s (roi will fallback to json/pkl): %s",
                    task_id[:8],
                    e,
                )
            info['task_status'] = RetCode.TASK_FINISHED.value
            info['finished'] = True
            info.pop('matcher', None)
            _save_task_info(task_id, info)
            # 大 JSON / info 落盘成功后清理分块临时目录 uploads/{task_id}/
            tile_tmp_root = os.path.join(upload_folder, task_id)
            try:
                shutil.rmtree(tile_tmp_root, ignore_errors=False)
                logger.info("removed tile temp dir task_id=%s path=%s", task_id[:8], tile_tmp_root)
            except Exception as e:
                logger.warning("failed to remove tile temp dir %s: %s", tile_tmp_root, e)
            t8 = time.time()
            logger.info(
                "save_task_info task_id=%s ms=%.2f finish_total_ms=%.2f",
                task_id[:8],
                (t8 - t7) * 1000,
                (t8 - t5) * 1000,
            )
            logger.info("Task %s finished and saved.", task_id)
        except Exception as e:
            info['task_status'] = RetCode.TASK_TIMEOUT.value
            try:
                _save_task_info(task_id, info)
            except Exception:
                logger.exception("failed to persist task status after finish error")
            logger.exception("_finish_task_impl error: %s", e)

    def check_image(self, task_id: str) -> dict:
        """不再检测缺失块，直接返回成功与空 missing_tiles"""
        try:
            _, err = _require_task_info(task_id)
            if err:
                return err
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
        info, err = _require_task_info(task_id)
        if err:
            return err
        return {
            'ret_code': RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value,
            'task_status': info.get('task_status')
        }

    def analyze_slide(self, task_id: str, analyze_names: list) -> dict:
        """
        玻片分析（骨髓玻片增生分析等）。
        cellularity(增生程度) = red_pixel_count / wbc_pixel_count，保留2位小数。
        """
        info, err = _require_task_info(task_id)
        if err:
            err = dict(err)
            err['result'] = {}
            return err
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
        roi, info, err = _require_roi_dataset(task_id)
        if err:
            return err
        if not info.get('finished', False):
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': 'Task not completed',
                'reason': 'Task not completed',
            }

        roi_xmin = 0 if roi_xmin is None else int(roi_xmin)
        roi_ymin = 0 if roi_ymin is None else int(roi_ymin)
        roi_xmax = float("inf") if roi_xmax is None else int(roi_xmax)
        roi_ymax = float("inf") if roi_ymax is None else int(roi_ymax)
        offset = max(0, int(index_offset or 0))
        limit = max(0, int(request_task_num or 0))

        t0 = time.time()
        matched = roi.cells_in_roi(roi_xmin, roi_ymin, roi_xmax, roi_ymax)
        total = int(matched.size)
        page_records = matched[offset: offset + limit] if limit > 0 else matched[offset:offset]
        page_dicts = roi.cell_records_to_dicts(page_records)
        logger.info(
            "get_result task_id=%s total=%d page=%d ms=%.2f",
            task_id[:8],
            total,
            len(page_dicts),
            (time.time() - t0) * 1000,
        )

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
        logger.info("roi_selection task_id=%s, task_type=%s, user_choice_area=%s, view_width=%s, view_height=%s, kwargs=%s, required_num=%s", task_id, task_type, user_choice_area, view_width, view_height, kwargs, required_num)
        roi, info, err = _require_roi_dataset(task_id)
        if err:
            err = dict(err)
            err.setdefault('result', {})
            return err
        if not info.get('finished', False):
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': 'Task not completed',
                'reason': 'Task not completed',
                'result': {},
            }
        smear_type = (info or {}).get("smear_type")
        dpi = info.get("dpi")
        tile_w = info.get("tile_width")
        tile_h = info.get("tile_height")
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

        required_wbc = _get_required_int("WBC")
        required_meg = _get_required_int("MEG") * 3
        required_rbc = _get_required_int("RBC")

        if smear_type == "BM":
            if normalized_task_type == "WBC":
                if not required_wbc or required_wbc <= 0:
                    return {
                        "ret_code": RetCode.ROI_ERROR.value,
                        "ret_desc": "Missing required_num.WBC for BM WBC",
                        "reason": "Missing required_num.WBC for BM WBC",
                    }
            elif normalized_task_type == "MEG":
                if not required_meg or required_meg <= 0:
                    return {
                        "ret_code": RetCode.ROI_ERROR.value,
                        "ret_desc": "Missing required_num.MEG for BM MEG",
                        "reason": "Missing required_num.MEG for BM MEG",
                    }
                if not isinstance(kwargs.get("wbc_points"), list) or not kwargs.get("wbc_points"):
                    return {
                        "ret_code": RetCode.ROI_ERROR.value,
                        "ret_desc": "Missing kwargs.wbc_points for BM MEG",
                        "reason": "Missing kwargs.wbc_points for BM MEG",
                    }
            elif normalized_task_type == "WBC_MEG":
                # 允许 MEG 为 0；仅 WBC 必填
                if not required_wbc or required_wbc <= 0:
                    return {
                        "ret_code": RetCode.ROI_ERROR.value,
                        "ret_desc": "Missing required_num.WBC or required_num.WBC is 0 for BM WBC_MEG",
                        "reason": "Missing required_num.WBC or required_num.WBC is 0 for BM WBC_MEG",
                    }
            elif normalized_task_type == "RBC":
                return {
                    "ret_code": RetCode.ROI_ERROR.value,
                    "ret_desc": "Invalid combo: BM does not support task_type=RBC",
                    "reason": "Invalid combo: BM does not support task_type=RBC",
                }
        elif smear_type == "PB":
            if normalized_task_type != "WBC":
                return {
                    "ret_code": RetCode.ROI_ERROR.value,
                    "ret_desc": f"Invalid combo: PB only supports task_type=WBC, got {task_type}",
                    "reason": f"Invalid combo: PB only supports task_type=WBC, got {task_type}",
                }
            if not required_wbc or required_wbc <= 0:
                return {
                    "ret_code": RetCode.ROI_ERROR.value,
                    "ret_desc": "Missing required_num.WBC for PB WBC",
                    "reason": "Missing required_num.WBC for PB WBC",
                }
        else:
            return {
                "ret_code": RetCode.ROI_ERROR.value,
                "ret_desc": f"Unsupported smear_type: {smear_type}",
                "reason": f"Unsupported smear_type: {smear_type}",
            }
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
                tile_w=tile_w,
                tile_h=tile_h,
            )
            pipeline = WBCSamplingPipeline(bm_cfg)
            wbc_tasks = pipeline.run(roi=roi)
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
                            "ret_code": RetCode.ROI_ERROR.value,
                            "ret_desc": "从 WBC 结果中未解析到任何 WBC 视野，无法计算 MEG 排序参考。",
                            "reason": "从 WBC 结果中未解析到任何 WBC 视野，无法计算 MEG 排序参考。",
                        }
                    bm_cfg.target_cell_num_MEG = required_meg
                    bm_cfg.View_type = "MEG"
                    try:
                        meg_pipeline = MegSamplingPipeline(bm_cfg)
                        meg_tasks = meg_pipeline.run_meg(
                            roi=roi, wbc_rects=wbc_rects_meg
                        )
                        meg_task_rects = [task.to_dict() for task in meg_tasks]
                    except Exception as e:
                        logger.exception("MEG roi_selection failed: %s", e)
                        return {
                            "ret_code": RetCode.ROI_ERROR.value,
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
                View_type="MEG",
                tile_w=tile_w,
                tile_h=tile_h,
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
                    "ret_code": RetCode.ROI_ERROR.value,
                    "ret_desc": "Invalid kwargs.wbc_points: empty or not parseable",
                    "reason": "Invalid kwargs.wbc_points: empty or not parseable",
                }
            try:
                meg_pipeline = MegSamplingPipeline(bm_cfg)
                meg_tasks = meg_pipeline.run_meg(roi=roi, wbc_rects=wbc_rects)
                final_task_list = [task.to_dict() for task in meg_tasks]
            except Exception as e:
                logger.exception("MEG roi_selection failed: %s", e)
                return {
                    "ret_code": RetCode.ROI_ERROR.value,
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
                tile_w=tile_w,
                tile_h=tile_h,
            )
            pipeline = WBCSamplingPipeline(bm_cfg)
            wbc_tasks = pipeline.run(roi=roi)
            final_task_list = [task.to_dict() for task in wbc_tasks]
        else:
            return {
                "ret_code": RetCode.CLIENT_ERROR.value,
                "ret_desc": f"roi_selection not implemented for smear_type={smear_type}, task_type={task_type}",
                "reason": f"roi_selection not implemented for smear_type={smear_type}, task_type={task_type}",
            }
        logger.info("roi_selection finished task_id=%s, ", task_id)
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

        model_name, model_warning = get_model_by_dpi(
            int(dpi),
            smear_type=smear_type,
            algorithm_types=target_cell_types or "",
            return_warning=True,
        )
        warning = err or model_warning

        input_dpi = int(dpi)
        try:
            bgr, orig_w, orig_h, scale_ratio, model_dpi, max_w, max_h = prepare_x100_bgr(
                image_bytes, input_dpi, model_name,
            )
        except ValueError as e:
            return {
                "ret_code": RetCode.CLIENT_ERROR.value,
                "ret_desc": str(e),
                "reason": str(e),
            }

        infer_dpi = model_dpi if scale_ratio != 1.0 else input_dpi
        dpi_bucket, _ = _get_dpi_bucket(input_dpi)

        gpu_id, _ = resolve_triton_route()
        ok, load_err = ensure_model_loaded(model_name, gpu_id=gpu_id)
        if not ok:
            return {
                "ret_code": RetCode.CLIENT_ERROR.value,
                "ret_desc": load_err,
                "reason": load_err,
            }

        try:
            result = infer_x100_on_bgr(
                bgr,
                infer_dpi=infer_dpi,
                smear_type=smear_type,
                target_cell_types=target_cell_types or "",
                filename=image_file.filename,
                gpu_id=gpu_id,
                max_w=max_w,
                max_h=max_h,
            )
        except Exception as e:
            logger.exception("Triton infer failed: %s", e)
            return {
                "ret_code": RetCode.CLIENT_ERROR.value,
                "ret_desc": str(e),
                "reason": str(e),
            }

        warning = warning or result.get("warning")
        cells = result.get("cells", [])
        cell_list = result.get("cell_list", [])
        if scale_ratio != 1.0:
            cells = map_cells_from_scaled(cells, scale_ratio)
            cell_list = map_cell_list_from_scaled(cell_list, scale_ratio)
        if not cell_list and cells:
            cell_list = [{
                'cell_xmin': c.cell_xmin, 'cell_ymin': c.cell_ymin,
                'cell_xmax': c.cell_xmax, 'cell_ymax': c.cell_ymax,
                'tops': [{'cell_type': c.cell_type, 'cell_type_name': c.cell_type_name,
                          'class_confidence': c.class_confidence, 'bbox_confidence': c.bbox_confidence}]
            } for c in cells]
        if dpi_bucket == 714756:
            edge_cell_filter = False
            if cell_list:
                try:
                    cell_list = filter_cell_dicts_edge_elongated_1pct(
                        cell_list, orig_w, orig_h
                    )
                    cell_list = filter_cell_dicts_small_wbc_714756(
                        cell_list, input_dpi
                    )
                except Exception as e:
                    logger.warning(
                        "get_task_result_x100: 714756 cell filter skipped: %s",
                        e,
                    )
        if edge_cell_filter and cell_list:
            try:
                cell_list = filter_cell_dicts_edge_incomplete(
                    cell_list, orig_w, orig_h
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
