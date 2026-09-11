import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import orjson
from config import cellularity_file_path_prefix

from backend.tools.MESSAGE_DICT import RetCode, RetDesc
from backend.tools.public_methods import thread_decorator, upload_folder
from backend.tools.combo_validator import validate_combo
from backend.tools.json_safe_writer import serialize_non_json_fields
from PIL import Image

from project.smear_project import SmearProject
from project.roi_store import RoiDataset
from project.cells import Cell
from backend.tools.x100_image_infer import run_cell_image_infer
from backend.tools.model_control import warmup_model, ensure_model_loaded
from backend.tools.triton_client import resolve_triton_route, infer_cellularity
from backend.tools.filter_edge_incomplete_cells import um_per_pixel_from_dpi
from algorithms.SelectArea.main_wbc import *
from algorithms.SelectArea.main_meg import *
from algorithms.SelectArea.main_rbc import *
from algorithms.SelectArea.setcover import solve, SetCoverSolverParameter
from algorithms.SelectArea.dedup_cells_across_tiles import dedup_cells_across_tiles_per_type


logger = logging.getLogger(__name__)

_IMAGE_FILE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# 增生分析独立路径：只加载 cell_analysis，走 /147246/infer_ca
_CELLULARITY_MODEL_NAME = "DPI147246_BM_PB_cell_analysis"
_CELLULARITY_RECT_FILES_DIR = "rect0_files"
_CELLULARITY_TILE_SIZE = 2048
_CELLULARITY_ANALYSIS_WIDTH_MM = 10.0
_CELLULARITY_ANALYSIS_HEIGHT_MM = 7.0
_CELLULARITY_MAX_IMAGES = 100
_CELLULARITY_MAX_WORKERS = 6

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
        slide_no = str(task_info.get('slide_no') or '').strip()
        if not slide_no:
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': 'slide_no cannot be empty',
                'reason': 'slide_no cannot be empty',
            }
        ok, err = validate_combo(dpi, smear_type, target_cell_types, allow_empty_types=False)
        if not ok:
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': err,
                'reason': err,
            }
        task_id = uuid.uuid4().hex
        task_info['task_id'] = task_id
        task_info['slide_no'] = slide_no
        task_info['smear_type'] = task_info.get('smear_type', 'BM')
        task_info['task_status'] = RetCode.TASK_RUNNING.value
        task_info['heatmap_orientation'] = int(task_info.get('heatmap_orientation', -1))
        task_info['finished'] = False
        task_info['wbc_pixel_count'] = 0
        task_info['red_pixel_count'] = 0
        os.makedirs(_task_tiles_dir(task_id), exist_ok=True)
        _save_task_info(task_id, task_info)

        # 40 倍平扫：创建时按需 load_models 预热全部 Triton 端点
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

        result = run_cell_image_infer(
            image_bytes,
            dpi,
            smear_type,
            target_cell_types or "",
            filename=filename,
            edge_cell_filter=False,
            ensure_loaded=False,
            allow_dpi_scale=False,
        )
        if not result.get("ok"):
            err = result.get("error") or "infer failed"
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': err,
                'reason': err,
            }
        cell_list = result.get("cell_list") or []
        cells_payload = _cells_to_dicts(result.get("cells") or [])
        payload = {
            'cells': cells_payload,
            'scores': _ensure_json_serializable(result.get("scores", [])),
            'wbc_pixel_count': int(result.get("wbc_pixel_count") or 0),
            'red_pixel_count': int(result.get("red_pixel_count") or 0),
        }
        _write_tile_result(task_id, row_index, col_index, payload)
        response = {
            'ret_code': RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value,
            'cell_list': cell_list,
        }
        if result.get("warning"):
            response['warning'] = result["warning"]
        return response

    def _finish_task_impl(self, task_id: str, project: SmearProject, info: dict) -> None:
        """去重、过滤、落盘为大 JSON + 更新 info（后台可跑，不依赖内存任务表）。"""
        t5 = time.time()
        try:
            dpi = info.get('dpi', 144750)
            tile_w = int(info.get('tile_width', 2448))
            tile_h = int(info.get('tile_height', 2048))
            layer = project.get_layer(dpi)
            tiles = dedup_cells_across_tiles_per_type(
                layer.iter_tiles(),
                tile_w=tile_w,
                tile_h=tile_h,
                iou_thresh=0.2,
                ios_thresh=0.7
                )
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

    def _cellularity_rect_key(self, rect: dict | None) -> tuple[int, int, int, int] | None:
        if not isinstance(rect, dict):
            return None
        try:
            return (
                int(round(float(rect["view_xmin"]))),
                int(round(float(rect["view_ymin"]))),
                int(round(float(rect["view_xmax"]))),
                int(round(float(rect["view_ymax"]))),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _build_cellularity_analysis_rect(
        self,
        largest_task_rect: dict | None,
        dpi: int,
    ) -> tuple[dict | None, dict | None]:
        """以最大外接矩形中心，按 10mm x 7mm 与 dpi 生成分析框。"""
        rect_key = self._cellularity_rect_key(largest_task_rect)
        if rect_key is None:
            return None, {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': 'largest_task_rect is missing or invalid',
                'reason': 'largest_task_rect is missing or invalid',
                'result': {},
            }
        try:
            dpi_val = float(dpi)
        except (TypeError, ValueError):
            return None, {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': f'invalid dpi for cellularity analysis: {dpi}',
                'reason': f'invalid dpi for cellularity analysis: {dpi}',
                'result': {},
            }
        if dpi_val <= 0:
            return None, {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': f'invalid dpi for cellularity analysis: {dpi}',
                'reason': f'invalid dpi for cellularity analysis: {dpi}',
                'result': {},
            }

        xmin, ymin, xmax, ymax = rect_key
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        um_per_px = um_per_pixel_from_dpi(int(dpi_val))
        width_px = (_CELLULARITY_ANALYSIS_WIDTH_MM * 1000.0) / um_per_px
        height_px = (_CELLULARITY_ANALYSIS_HEIGHT_MM * 1000.0) / um_per_px
        half_w = width_px / 2.0
        half_h = height_px / 2.0
        analysis_rect = {
            "view_xmin": int(round(cx - half_w)),
            "view_ymin": int(round(cy - half_h)),
            "view_xmax": int(round(cx + half_w)),
            "view_ymax": int(round(cy + half_h)),
            "width": int(round(width_px)),
            "height": int(round(height_px)),
            "center_x": int(round(cx)),
            "center_y": int(round(cy)),
            "dpi": int(round(dpi_val)),
            "width_mm": _CELLULARITY_ANALYSIS_WIDTH_MM,
            "height_mm": _CELLULARITY_ANALYSIS_HEIGHT_MM,
            "um_per_pixel": um_per_px,
        }
        return analysis_rect, None

    def _cellularity_tile_intersects_rect(self, image_path: str, rect: dict | None) -> bool:
        rect_key = self._cellularity_rect_key(rect)
        if rect_key is None:
            return True
        tile = self._cellularity_tile_bounds(image_path)
        if tile is None:
            return False
        tile_xmin, tile_ymin, tile_xmax, tile_ymax = tile
        rect_xmin, rect_ymin, rect_xmax, rect_ymax = rect_key
        return (
            tile_xmin <= rect_xmax
            and tile_xmax >= rect_xmin
            and tile_ymin <= rect_ymax
            and tile_ymax >= rect_ymin
        )

    def _cellularity_tile_bounds(
        self,
        image_path: str,
    ) -> tuple[int, int, int, int] | None:
        """从文件名 col_row 解析 tile 全局像素范围。"""
        stem = os.path.splitext(os.path.basename(image_path))[0]
        parts = stem.split("_")
        if len(parts) < 2:
            return None
        try:
            col = int(parts[0])
            row = int(parts[1])
        except ValueError:
            return None
        tile_xmin = col * _CELLULARITY_TILE_SIZE
        tile_ymin = row * _CELLULARITY_TILE_SIZE
        return (
            tile_xmin,
            tile_ymin,
            tile_xmin + _CELLULARITY_TILE_SIZE,
            tile_ymin + _CELLULARITY_TILE_SIZE,
        )

    def _select_central_cellularity_images(
        self,
        image_paths: list[str],
        analysis_rect: dict | None,
        max_images: int = _CELLULARITY_MAX_IMAGES,
    ) -> list[str]:
        """按距分析框中心的距离取最中心的 max_images 张；不足则全取。"""
        if len(image_paths) <= max_images:
            return list(image_paths)

        rect_key = self._cellularity_rect_key(analysis_rect)
        if isinstance(analysis_rect, dict):
            try:
                cx = float(analysis_rect.get("center_x"))
                cy = float(analysis_rect.get("center_y"))
            except (TypeError, ValueError):
                cx = cy = None
        else:
            cx = cy = None
        if cx is None or cy is None:
            if rect_key is None:
                return image_paths[:max_images]
            xmin, ymin, xmax, ymax = rect_key
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0

        scored: list[tuple[float, str]] = []
        fallback: list[str] = []
        half = _CELLULARITY_TILE_SIZE / 2.0
        for path in image_paths:
            bounds = self._cellularity_tile_bounds(path)
            if bounds is None:
                fallback.append(path)
                continue
            tile_xmin, tile_ymin, _, _ = bounds
            dx = (tile_xmin + half) - cx
            dy = (tile_ymin + half) - cy
            scored.append((dx * dx + dy * dy, path))

        scored.sort(key=lambda item: (item[0], item[1]))
        selected = [path for _, path in scored[:max_images]]
        if len(selected) < max_images and fallback:
            need = max_images - len(selected)
            selected.extend(fallback[:need])
        return selected

    def _iter_image_paths(self, folder_path: str, largest_task_rect: dict | None = None) -> list[str]:
        image_paths: list[str] = []
        for root, _, files in os.walk(folder_path):
            for name in files:
                if os.path.splitext(name)[1].lower() not in _IMAGE_FILE_EXTS:
                    continue
                image_path = os.path.join(root, name)
                if self._cellularity_tile_intersects_rect(image_path, largest_task_rect):
                    image_paths.append(image_path)
        image_paths.sort()
        return image_paths

    def _resolve_cellularity_folder_path(self, folder_path: str) -> str:
        raw_path = os.path.expanduser(str(folder_path).strip())
        prefix = str(cellularity_file_path_prefix or "").strip()
        if not raw_path or not prefix:
            return os.path.normpath(raw_path)

        normalized_raw = raw_path.replace("\\", "/")
        normalized_prefix = prefix.rstrip("/\\").replace("\\", "/")
        is_windows_abs = (
            len(raw_path) >= 3
            and raw_path[1] == ":"
            and raw_path[2] in ("/", "\\")
        ) or raw_path.startswith("\\\\")
        already_prefixed = (
            normalized_raw == normalized_prefix
            or normalized_raw.startswith(f"{normalized_prefix}/")
        )
        if is_windows_abs or already_prefixed:
            return os.path.normpath(raw_path)
        return os.path.normpath(os.path.join(prefix, raw_path.lstrip("/\\")))

    def _latest_cellularity_folder_path(self, info: dict) -> tuple[str | None, dict | None]:
        slide_no = str(info.get('slide_no') or '').strip().strip("/\\")
        if not slide_no:
            return None, {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': 'slide_no cannot be empty',
                'reason': 'slide_no cannot be empty',
                'result': {},
            }

        base_dir = os.path.normpath(os.path.join(
            str(cellularity_file_path_prefix or "").strip(),
            slide_no,
            _CELLULARITY_RECT_FILES_DIR,
        ))
        if not os.path.isdir(base_dir):
            return None, {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': f'cellularity rect0_files directory not found: {base_dir}',
                'reason': f'cellularity rect0_files directory not found: {base_dir}',
                'result': {},
            }

        numeric_dirs: list[tuple[int, str]] = []
        with os.scandir(base_dir) as entries:
            for entry in entries:
                if entry.is_dir() and entry.name.isdigit():
                    numeric_dirs.append((int(entry.name), entry.path))
        if not numeric_dirs:
            return None, {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': f'No numeric directory found under: {base_dir}',
                'reason': f'No numeric directory found under: {base_dir}',
                'result': {},
            }
        return os.path.normpath(max(numeric_dirs, key=lambda item: item[0])[1]), None

    def _calculate_cellularity_pixels(
        self,
        task_id: str,
        info: dict,
        folder_path: str,
        analysis_rect: dict | None,
    ) -> tuple[dict | None, dict | None, dict]:
        timings = {"pad_ms": 0.0, "infer_ms": 0.0}
        request_folder_path = str(folder_path)
        resolved_folder_path = self._resolve_cellularity_folder_path(request_folder_path)
        if not os.path.isdir(resolved_folder_path):
            return None, {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': f'folder_path not found or not a directory: {resolved_folder_path}',
                'reason': f'folder_path not found or not a directory: {resolved_folder_path}',
                'request_folder_path': request_folder_path,
                'result': {},
            }, timings

        image_paths = self._iter_image_paths(resolved_folder_path, analysis_rect)
        if not image_paths:
            return None, {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': f'No image files matched analysis_rect in folder_path: {resolved_folder_path}',
                'reason': f'No image files matched analysis_rect in folder_path: {resolved_folder_path}',
                'request_folder_path': request_folder_path,
                'analysis_rect': analysis_rect,
                'result': {},
            }, timings
        matched_count = len(image_paths)
        image_paths = self._select_central_cellularity_images(
            image_paths,
            analysis_rect,
            max_images=_CELLULARITY_MAX_IMAGES,
        )

        gpu_id, _ = resolve_triton_route()
        ok, load_err = ensure_model_loaded(_CELLULARITY_MODEL_NAME, gpu_id=gpu_id)
        if not ok:
            return None, {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': load_err,
                'reason': load_err,
                'models': [_CELLULARITY_MODEL_NAME],
                'result': {},
            }, timings

        wbc_pixel_count = 0
        red_pixel_count = 0
        failed_images: list[dict] = []
        success_count = 0
        total_infer_ms = 0.0
        total_pad_ms = 0.0

        def _infer_one(image_path: str) -> tuple[str, dict]:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            result = infer_cellularity(
                image_bytes,
                filename=os.path.basename(image_path),
                gpu_id=gpu_id,
            )
            return image_path, result

        with ThreadPoolExecutor(max_workers=_CELLULARITY_MAX_WORKERS) as executor:
            futures = {
                executor.submit(_infer_one, image_path): image_path
                for image_path in image_paths
            }
            for fut in as_completed(futures):
                image_path = futures[fut]
                try:
                    _, result = fut.result()
                    total_infer_ms += float(result.get("infer_ms") or 0.0)
                    total_pad_ms += float(result.get("pad_ms") or 0.0)
                    if not result.get("ok"):
                        failed_images.append({
                            "image_path": image_path,
                            "reason": result.get("error") or "infer failed",
                        })
                    else:
                        wbc_pixel_count += int(result.get("wbc_pixel_count") or 0)
                        red_pixel_count += int(result.get("red_pixel_count") or 0)
                        success_count += 1
                except Exception as e:
                    logger.exception(
                        "cellularity infer failed task_id=%s image=%s",
                        task_id[:8],
                        image_path,
                    )
                    failed_images.append({"image_path": image_path, "reason": str(e)})

        timings = {"pad_ms": total_pad_ms, "infer_ms": total_infer_ms}
        if success_count <= 0:
            return None, {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': 'All images failed to infer cellularity pixels',
                'reason': 'All images failed to infer cellularity pixels',
                'failed_images': failed_images,
                'result': {},
            }, timings

        info['wbc_pixel_count'] = wbc_pixel_count
        info['red_pixel_count'] = red_pixel_count
        info['cellularity_image_count'] = success_count
        info['cellularity_matched_image_count'] = matched_count
        info['cellularity_failed_count'] = len(failed_images)
        info['cellularity_failed_images'] = failed_images
        info['cellularity_request_folder_path'] = request_folder_path
        info['cellularity_folder_path'] = resolved_folder_path
        info['cellularity_analysis_rect'] = analysis_rect
        info['cellularity_analyzed_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _save_task_info(task_id, info)
        return info, None, timings

    def _has_cached_cellularity_pixels(
        self,
        info: dict,
        folder_path: str,
        analysis_rect: dict | None,
    ) -> bool:
        request_folder_path = str(folder_path)
        resolved_folder_path = self._resolve_cellularity_folder_path(request_folder_path)
        cached_request_path = str(info.get('cellularity_request_folder_path') or "")
        cached_resolved_path = str(info.get('cellularity_folder_path') or "")
        same_path = (
            cached_request_path == request_folder_path
            or (
                cached_resolved_path
                and os.path.normcase(os.path.normpath(cached_resolved_path))
                == os.path.normcase(os.path.normpath(resolved_folder_path))
            )
        )
        same_rect = (
            self._cellularity_rect_key(info.get('cellularity_analysis_rect'))
            == self._cellularity_rect_key(analysis_rect)
        )
        return (
            same_path
            and same_rect
            and 'wbc_pixel_count' in info
            and 'red_pixel_count' in info
            and info.get('cellularity_image_count') is not None
        )

    def _ensure_largest_task_rect(
        self,
        task_id: str,
        info: dict,
    ) -> tuple[dict | None, dict, dict | None, float]:
        largest_task_rect = info.get('largest_task_rect')
        if largest_task_rect:
            return largest_task_rect, info, None, 0.0

        info_task_id = str(info.get('task_id') or task_id).strip()
        t0 = time.perf_counter()
        selection_result = self.get_task_list_x100(
            task_id=info_task_id,
            task_type="WBC",
            user_choice_area=None,
            view_width=384,
            view_height=283,
            kwargs={},
            required_num={"WBC": 30},
        )
        roi_ms = (time.perf_counter() - t0) * 1000.0
        if selection_result.get('ret_code') != RetCode.API_SUCCESS.value:
            err = dict(selection_result)
            err.setdefault('result', {})
            return None, info, err, roi_ms

        refreshed_info = _load_task_info_from_disk(info_task_id) or info
        largest_task_rect = (
            refreshed_info.get('largest_task_rect')
            or selection_result.get('largest_task_rect')
        )
        if largest_task_rect and not refreshed_info.get('largest_task_rect'):
            refreshed_info['largest_task_rect'] = largest_task_rect
            _save_task_info(info_task_id, refreshed_info)
        return largest_task_rect, refreshed_info, None, roi_ms

    def analyze_slide(
        self,
        task_id: str,
        analyze_names: list,
    ) -> dict:
        """
        玻片分析（骨髓玻片增生分析等）。
        cellularity(增生程度) = red_pixel_count / wbc_pixel_count，保留2位小数。
        """
        t0 = time.perf_counter()
        logger.info(
            "analyze_slide start task_id=%s analyze_names=%s",
            task_id[:8],
            analyze_names,
        )
        pad_ms = 0.0
        infer_ms = 0.0
        roi_ms = 0.0

        def _finish(payload: dict) -> dict:
            logger.info(
                "analyze_slide done task_id=%s ret_code=%s total_ms=%.2f "
                "pad_ms=%.2f infer_ms=%.2f roi_ms=%.2f",
                task_id[:8],
                payload.get("ret_code"),
                (time.perf_counter() - t0) * 1000.0,
                pad_ms,
                infer_ms,
                roi_ms,
            )
            return payload

        info, err = _require_task_info(task_id)
        if err:
            err = dict(err)
            err['result'] = {}
            return _finish(err)
        largest_task_rect, info, rect_err, roi_ms = self._ensure_largest_task_rect(task_id, info)
        if rect_err:
            return _finish(rect_err)
        analysis_rect, analysis_err = self._build_cellularity_analysis_rect(
            largest_task_rect,
            info.get('dpi'),
        )
        if analysis_err:
            return _finish(analysis_err)
        cache_hit = False
        cellularity_folder_path = None
        if 'cellularity' in analyze_names:
            cellularity_folder_path, path_err = self._latest_cellularity_folder_path(info)
            if path_err:
                return _finish(path_err)
            if self._has_cached_cellularity_pixels(info, cellularity_folder_path, analysis_rect):
                cache_hit = True
            else:
                info, calc_err, timings = self._calculate_cellularity_pixels(
                    task_id,
                    info,
                    cellularity_folder_path,
                    analysis_rect,
                )
                pad_ms = float(timings.get("pad_ms") or 0.0)
                infer_ms = float(timings.get("infer_ms") or 0.0)
                if calc_err:
                    return _finish(calc_err)

        result = {}
        if 'cellularity' in analyze_names:
            wbc = info.get('wbc_pixel_count', 0) or 0
            red = info.get('red_pixel_count', 0) or 0
            if wbc <= 0:
                result['cellularity'] = None
            else:
                result['cellularity'] = round(red / wbc, 2)
            result['wbc_pixel_count'] = int(wbc)
            result['red_pixel_count'] = int(red)
            result['image_count'] = int(info.get('cellularity_image_count') or 0)
            result['matched_image_count'] = int(info.get('cellularity_matched_image_count') or 0)
            result['failed_count'] = int(info.get('cellularity_failed_count') or 0)
            result['folder_path'] = cellularity_folder_path
            result['cache_hit'] = cache_hit
            result['largest_task_rect'] = largest_task_rect
            result['analysis_rect'] = analysis_rect
        return _finish({
            'ret_code': RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value,
            'result': result,
        })

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

    def _largest_task_rect(self, task_list: list[dict]) -> dict | None:
        xs1: list[float] = []
        ys1: list[float] = []
        xs2: list[float] = []
        ys2: list[float] = []
        for task in task_list or []:
            if not isinstance(task, dict):
                continue
            try:
                if all(k in task for k in ("view_xmin", "view_ymin", "view_xmax", "view_ymax")):
                    x1 = float(task["view_xmin"])
                    y1 = float(task["view_ymin"])
                    x2 = float(task["view_xmax"])
                    y2 = float(task["view_ymax"])
                elif all(k in task for k in ("x", "y", "w", "h")):
                    x1 = float(task["x"])
                    y1 = float(task["y"])
                    x2 = x1 + float(task["w"])
                    y2 = y1 + float(task["h"])
                else:
                    continue
            except (TypeError, ValueError):
                continue

            xs1.append(min(x1, x2))
            ys1.append(min(y1, y2))
            xs2.append(max(x1, x2))
            ys2.append(max(y1, y2))

        if not xs1:
            return None

        x1 = min(xs1)
        y1 = min(ys1)
        x2 = max(xs2)
        y2 = max(ys2)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        return {
            "view_xmin": int(round(x1)),
            "view_ymin": int(round(y1)),
            "view_xmax": int(round(x2)),
            "view_ymax": int(round(y2)),
            "width": int(round(width)),
            "height": int(round(height)),
            "area": int(round(width * height)),
            "task_count": len(xs1),
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
        t_roi0 = time.perf_counter()
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
            pb_cfg = BM40Config(
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
            pipeline = RBCSamplingPipeline(pb_cfg)
            wbc_tasks = pipeline.run(roi=roi)
            final_task_list = [task.to_dict() for task in wbc_tasks]
        else:
            return {
                "ret_code": RetCode.CLIENT_ERROR.value,
                "ret_desc": f"roi_selection not implemented for smear_type={smear_type}, task_type={task_type}",
                "reason": f"roi_selection not implemented for smear_type={smear_type}, task_type={task_type}",
            }
        largest_task_rect = self._largest_task_rect(final_task_list)
        info['largest_task_rect'] = largest_task_rect
        _save_task_info(task_id, info)
        roi_ms = (time.perf_counter() - t_roi0) * 1000.0
        logger.info(
            "roi_selection finished task_id=%s task_list_num=%d largest_task_rect=%s ms=%.2f task_list=%s",
            task_id[:8],
            len(final_task_list),
            largest_task_rect,
            roi_ms,
            str(final_task_list),
        )
        return {
            "ret_code": RetCode.API_SUCCESS.value,
            "ret_desc": RetDesc.API_SUCCESS.value,
            "task_list_num": len(final_task_list),
            "task_list": final_task_list,
            "largest_task_rect": largest_task_rect,
            "roi_selection_ms": round(roi_ms, 2),
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
        filename = getattr(image_file, "filename", None) or "image.jpg"
        result = run_cell_image_infer(
            image_bytes,
            int(dpi),
            smear_type or "BM",
            target_cell_types or "",
            filename=filename,
            edge_cell_filter=_parse_edge_cell_filter_flag(edge_cell_filter),
        )
        if not result.get("ok"):
            err = result.get("error") or "infer failed"
            return {
                "ret_code": RetCode.CLIENT_ERROR.value,
                "ret_desc": err,
                "reason": err,
            }
        cell_list = result.get("cell_list") or []
        response = {
            "ret_code": RetCode.API_SUCCESS.value,
            "ret_desc": RetDesc.API_SUCCESS.value,
            "cell_count": len(cell_list),
            "cell_list": cell_list,
        }
        if result.get("warning"):
            response["warning"] = result["warning"]
        return response
