import os
import threading
import multiprocessing as mp
import time
import uuid

import cv2
import numpy as np
from itertools import chain

from algorithms.x100model import X100ImageModels
from backend.tools.MESSAGE_DICT import RetCode, RetDesc, TaskType, CELL_TYPES_X100
from backend.tools.public_methods import thread_decorator, upload_folder, images_folder
from backend.tools.json_safe_writer import serialize_non_json_fields
from project.smear_project import SmearProject
from project.cells import Cell
from project.inference_queue_manager import TileInferenceQueueManager
from project.tile_queue import TileQueueRouter, TileMsg

# -----------------------------
# B方案：QueueManager 懒加载 + 预热
# -----------------------------

QueueManager = None


def get_queue_manager() -> TileInferenceQueueManager:
    global QueueManager
    if QueueManager is None:
        # Under spawn, child process imports modules too; avoid double-init.
        if mp.current_process().name != "MainProcess":
            raise RuntimeError("QueueManager must be initialized in MainProcess")
        QueueManager = TileInferenceQueueManager(model_num_workers=3)
    return QueueManager


tile_router = TileQueueRouter()

# NOTE: 如果 X100ImageModels 也内部带多进程/GPU，建议也改为懒加载。
dispatcher = X100ImageModels.X100ImageModels(num_workers=1)


def _on_tile_factory(task_service):
    def on_tile(msg: TileMsg):
        task_id = msg.task_id
        image_uid = msg.image_uid
        msg.tile_meta['position_x'] = msg.position_x
        msg.tile_meta['position_y'] = msg.position_y
        # B方案：只提交 bytes，不在 Flask 进程 decode / 不触碰模型
        project = task_service.project[task_id]
        layer = project.list_layers()[0]
        dpi = layer.dpi
        get_queue_manager().submit_tile_bytes(
            project_task_id=task_id,
            dpi=dpi,
            image_uid=image_uid,
            tile_bytes=msg.tile_bytes,
            tile_meta=msg.tile_meta,
        )

    return on_tile


class TaskService:
    def __init__(self):
        self.project = {}
        self.project_info = {}
        self.grids = {}
        self.grids_position = {}
        self.project_lock = {}
        self.X100_results = {}
        self.roi_cache = {}
        self.roi_cache_lock = threading.Lock()

    def load_data(self, task_id):
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, f"{task_id}.smear.pkl")
        if not os.path.exists(file_path):
            return {"ret_code": RetCode.CLIENT_ERROR.value,
                    "ret_desc": RetDesc.CLIENT_ERROR.value,
                    'reason': '任务ID不存在',
                    'msg': f"file for task_id '{task_id}' not found."
                    }
        self.project[task_id] = SmearProject.load_pickle(task_id, upload_folder)

    def create_task(self, task_info: dict) -> dict:
        # B+预热：确保 inference 进程已 READY
        get_queue_manager()
        task_id = uuid.uuid4().hex
        task_info['smear_type'] = task_info.get('smear_type', 'BM')  # bm为骨髓
        task_info['task_status'] = RetCode.TASK_RUNNING.value
        num_rows = task_info.get('num_rows')
        num_cols = task_info.get('num_cols')
        project = SmearProject(smear_type=task_info['smear_type'])
        project.add_layer(task_info['dpi'])
        tile_router.create_task(
            task_id,
            on_tile_callback=_on_tile_factory(self),
            num_workers=4,
            queue_maxsize=256,
            get_timeout_sec=300,
        )

        self.project[task_id] = project
        self.project_lock[task_id] = threading.Lock()
        self.grids[task_id] = np.full((num_rows, num_cols), False, dtype=bool)
        self.grids_position[task_id] = np.full((num_rows, num_cols), False, dtype=bool)
        self.project_info[task_id] = task_info
        print('创建任务成功：', task_id)
        get_queue_manager().register_project(project, task_id)
        get_queue_manager().set_expected_tiles(task_id, num_rows * num_cols)
        return {
            'task_id': task_id,
            'ret_code': RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value
        }

    def update_coordinates(self, task_id, tiles_msg):
        if task_id not in self.project:
            result = self.load_data(task_id)
            if result:
                return result
        project = self.project[task_id]
        layer = project.list_layers()[0]
        project_info = self.project_info[task_id]
        with self.project_lock[task_id]:
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
                tile = layer.add_tile(
                    x=position_x,
                    y=position_y,
                    w=project_info.get('tile_width'),
                    h=project_info.get('tile_height'),
                    image_data=None,
                    extra_meta={
                        'num_rows': project_info.get('num_rows'),
                        'num_cols': project_info.get('num_cols'),
                        'row_index': row_index,
                        'col_index': col_index,
                    }
                )
                matcher[(row_index, col_index)] = tile.image_uid
            else:
                try:
                    tiles = layer.get_tile(image_uid)
                    tiles.x = position_x
                    tiles.y = position_y
                except Exception as e:
                    tile_info['reason'] = str(e)
                    failed_tiles.append(tile_info)
        project.save_json(os.path.join(upload_folder, f"{task_id}.json"))
        return {
            'ret_code': RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value,
            'failed_tiles': failed_tiles
        }

    def upload_image(self, task_id, row_index, col_index, tile_image):
        row_index, col_index = int(row_index), int(col_index)
        with self.project_lock[task_id]:
            matcher = self.project_info[task_id].get('matcher')
            if not matcher:
                matcher = {}
                self.project_info[task_id]['matcher'] = matcher
        image_uid = matcher.get((row_index, col_index))
        project = self.project[task_id]
        layer = project.list_layers()[0]
        # 先创建 tile，确保结果回来能找到 tile
        task_info = self.project_info[task_id]
        image_bytes = tile_image.read()
        with self.project_lock[task_id]:
            grid = self.grids[task_id]
            grid[row_index, col_index] = True
            finished = grid.all()
        if not image_uid:
            tile = layer.add_tile(
                x=None,
                y=None,
                w=task_info['tile_width'],
                h=task_info['tile_height'],
                image_data=None,
                extra_meta={
                    'num_rows': task_info['num_rows'],
                    'num_cols': task_info['num_cols'],
                    'row_index': row_index,
                    'col_index': col_index,
                }
            )
            matcher[(row_index, col_index)] = tile.image_uid
        tile_router.push_tile(
            task_id=task_id,
            image_uid=image_uid,
            position_x=None,
            position_y=None,
            tile_bytes=image_bytes,
            tile_meta={
                'row_index': row_index,
                'col_index': col_index
            }
        )
        # 不要在接口线程里 join（会卡住接口）。把收尾全部放后台线程。
        if finished:
            self._finish_task_async(task_id)
        return {
            'ret_code': RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value,
            'image_uid': image_uid
        }

    @thread_decorator
    def _finish_task_async(self, task_id: str):
        try:
            tile_router.finish_task(task_id)
            tile_router.join_task(task_id)
            project = self.project[task_id]
            get_queue_manager().finish_tile(task_id)
            get_queue_manager().wait_written_all(task_id, timeout=300.0)
            self.project_info[task_id]['task_status'] = RetCode.TASK_FINISHED.value  # 100: 已完成
            project.save_json(os.path.join(upload_folder, f"{task_id}.json"))
            print('任务完成：============================================', task_id)
        except Exception as e:
            self.project_info[task_id]['task_status'] = RetCode.TASK_TIMEOUT.value  # 200: 失败
            print("[finish_task_async ERROR]", task_id, repr(e))

    def check_image(self, task_id: str) -> dict:
        try:
            if task_id not in self.project:
                result = self.load_data(task_id)
                if result:
                    return result
            project = self.project[task_id]
            grid = self.grids.get(task_id)
            if grid is None:
                layer = project.list_layers()[0]
                tiles = layer.iter_tiles()
                tile0 = tiles[0]
                num_rows = tile0.meta.get('num_rows', 0)
                num_cols = tile0.meta.get('num_cols', 0)
                grid = np.full((num_rows, num_cols), False, dtype=bool)
                for tile in tiles:
                    row_index = tile.meta.get('row_index')
                    col_index = tile.meta.get('col_index')
                    grid[row_index, col_index] = True
            rows, cols = np.where(~grid)
            missing_tiles = []
            for r, c in zip(rows, cols):
                missing_tiles.append({
                    'row_index': int(r),
                    'col_index': int(c),
                    'position_missing': True,
                    'image_missing': True
                })
            return {
                'ret_code': RetCode.API_SUCCESS.value,
                'ret_desc': RetDesc.API_SUCCESS.value,
                'missing_tiles': missing_tiles
            }
        except Exception as e:
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': RetDesc.CLIENT_ERROR.value,
                'reason': str(e)
            }

    def task_status(self, task_id: str) -> dict:
        task_info = self.project_info.get(task_id)
        if not task_info:
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': RetDesc.CLIENT_ERROR.value,
                'reason': '任务ID不存在'
            }
        task_status = task_info.get('task_status')
        return {
            'ret_code': RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value,
            'task_status': task_status
        }

    def get_result(self, task_id, roi_xmin, roi_ymin, roi_xmax, roi_ymax, index_offset, request_task_num):
        if task_id not in self.project:
            result = self.load_data(task_id)
            if result:
                return result
        project = self.project[task_id]
        layer = project.list_layers()[0]

        # 统一默认值（和 Layer 保持一致）
        roi_xmin = 0 if roi_xmin is None else int(roi_xmin)
        roi_ymin = 0 if roi_ymin is None else int(roi_ymin)
        roi_xmax = float("inf") if roi_xmax is None else int(roi_xmax)
        roi_ymax = float("inf") if roi_ymax is None else int(roi_ymax)

        offset = max(0, int(index_offset or 0))
        limit = max(0, int(request_task_num or 0))

        roi_key = (roi_xmin, roi_ymin, roi_xmax, roi_ymax)

        # ---- 读缓存 / 构建缓存 ----
        with self.roi_cache_lock:
            task_cache = self.roi_cache.setdefault(task_id, {})
            hit = task_cache.get(roi_key)

        if hit is None:
            # 第一次：遍历一次，拿到全量 Cell（只存对象，分页时再 to_dict）
            cells_all = layer.iter_cells_in_roi(roi_xmin, roi_ymin, roi_xmax, roi_ymax,
                                                is_Cell=True)  # :contentReference[oaicite:5]{index=5}
            total = len(cells_all)
            hit = (total, cells_all)
            with self.roi_cache_lock:
                task_cache[roi_key] = hit
                # 可选：限制每个 task 的 ROI 缓存数量，避免内存爆
                if len(task_cache) > 8:
                    task_cache.pop(next(iter(task_cache)))
        else:
            total, cells_all = hit

        total, cells_all = hit
        page_cells = cells_all[offset: offset + limit]
        page_dicts = [c.to_dict() for c in page_cells]
        has_more = (offset + limit) < total
        if not has_more:
            # 最后一页，释放缓存
            with self.roi_cache_lock:
                task_cache = self.roi_cache.get(task_id)
                if task_cache and roi_key in task_cache:
                    task_cache.pop(roi_key)

        return {
            "ret_code": RetCode.API_SUCCESS.value,
            "ret_desc": RetDesc.API_SUCCESS.value,
            "cell_count": total,  # ✅ 总量必有
            "cell_list": page_dicts,  # ✅ 分页结果
            "index_offset": offset,
            # "request_task_num": limit,
            # "has_more": (offset + limit) < total
        }

    def get_task_list_x100(self, task_id, user_choice_area, view_width, view_height, target_num_WBC,
                           target_num_MEG, index_offset, request_task_num):
        if task_id not in self.project:
            result = self.load_data(task_id)
            if result:
                return result
        project = self.project[task_id]
        layer = project.list_layers()[0]
        if not user_choice_area:
            print('user_choice_area is None, use full area')
            user_choice_area = {
                'x_min': 0,
                'y_min': 0,
                'x_max': layer.num_cols - 1,
                'y_max': layer.num_rows - 1
            }
        x_min, y_min, x_max, y_max = user_choice_area['x_min'], user_choice_area['y_min'], \
            user_choice_area['x_max'], user_choice_area['y_max']
        user_choose_area = {
            'min_row': y_min,
            'max_row': y_max,
            'min_col': x_min,
            'max_col': x_max
        }
        if not self.project_x100.get(task_id):
            infos_40xtile = []
            for row in range(y_min, y_max + 1):
                for col in range(x_min, y_min + 1):
                    tile = layer.get_tile(row, col)
                    global_x, global_y = tile.global_x, tile.global_y
                    cells = tile.cells
                    scores = tile.get_meta_quality_score()
                    new_scores = {
                        '0_0': [scores[0][4], scores[0][5]],
                        '1_0': [scores[1][4], scores[1][5]],
                        '0_1': [scores[2][4], scores[2][5]],
                        '1_1': [scores[3][4], scores[3][5]],
                    }
                    meg_center_pt = []  # 巨核细胞
                    local_cell_rects = []  # 有核细胞
                    global_cell_rects = []
                    # haveCellCenterPoints cell_type == 0: 有核细胞
                    # bigCellRects cell_type == 1: 巨核细胞
                    for one in cells:
                        if one.cell_type == 0:  # 有核细胞
                            local_cell_rects.append([one.x_min, one.y_min, one.x_max, one.y_max, one.class_confidence])
                            global_cell_rects.append([one.x_min + global_x, one.x_max + global_x, one.y_min + global_y,
                                                      one.y_max + global_y])
                        if one.cell_type == 1:  # 巨核细胞
                            meg_center_pt.append([one.x_min, one.y_min, one.x_max, one.y_max, one.class_confidence])
                    infos_40xtile.append({
                        "index_40xtile_x": tile.col_index,
                        'index_40xtile_y': layer.num_rows - tile.row_index - 1,
                        'abs_40xtile_x': global_x,
                        'abs_40xtile_y': global_y,
                        'local_cell_rects': local_cell_rects,
                        'global_cell_rects': global_cell_rects,
                        'meg_rect': meg_center_pt,
                        "scores": new_scores
                    })
            infos_40xtile = dedup_cells_across_tiles(infos_40xtile, iou_thresh=0.2)  # 有核细胞去重
            save_dir = os.path.join(images_folder, task_id)
            os.makedirs(save_dir, exist_ok=True)
            task_list = select_and_generate_bestArea_capture_tasks(infos_40xtile,
                                                                   user_choose_area,
                                                                   2,
                                                                   target_num_WBC,
                                                                   rect_width=view_width,  # 百倍视野的宽
                                                                   rect_height=view_height,  # 百倍视野的高
                                                                   save_flag=True,
                                                                   save_dir=save_dir)
            new_task_list = list(chain(*task_list))
            self.project_x100[task_id] = new_task_list
            pass
        else:
            new_task_list = self.project_x100[task_id]
        return {
            'ret_code': RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value,
            'task_list_num': len(new_task_list),
            'task_list': serialize_non_json_fields(new_task_list[index_offset:index_offset + request_task_num])
        }

    def get_task_result_x100(self, task_id, image_file, algorithm_type, dpi,
                             edge_cell_filter, *args):
        if None in args:
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': RetDesc.CLIENT_ERROR.value
            }
        try:
            task_type = TaskType[algorithm_type].value
        except KeyError:
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': RetDesc.CLIENT_ERROR.value,
            }
        if task_id not in self.project:
            result = self.load_data(task_id)
            if result:
                return result
        project = self.project[task_id]
        layer = project.get_layer(dpi)
        position_xmin, position_ymin, position_xmax, position_ymax = args
        x, y, w, h = position_xmin, position_ymin, position_xmax - position_xmin, position_ymax - position_ymin
        tiles = layer.iter_tiles()
        tile = None
        for one in tiles:
            if one.x == w and one.y == y and one.w == w and one.h == h:
                tile = one
                break
        if tile is None:
            tile = layer.add_tile(
                x=x,
                y=y,
                w=w,
                h=h,
            )
        image_bytes = image_file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR_BGR)
        new_dispatcher = dispatcher
        task_id = new_dispatcher.enqueue_task(image, task_type)
        cell_list = []
        cells = []
        for _ in range(7200000):  # 最多等待2小时
            new_result = new_dispatcher.get_result(task_id)
            if new_result:
                cellRects = new_result['cellRects'].astype(int).tolist()
                cellTypes = new_result['cellTypes'].astype(int).tolist()
                cellRatios = new_result['cellRatios'].astype(float).tolist()
                for i in range(len(cellRects)):
                    x, y, w, h, *o = cellRects[i]
                    cell_type = cellTypes[i][0] + 200000
                    new_cell_type = CELL_TYPES_X100.get(cell_type)
                    if not new_cell_type:
                        cell_type_name = '未知细胞'
                    else:
                        cell_type_name = new_cell_type[1]
                    class_confidence = cellRatios[i][0]
                    new_one_data = {
                        "cell_xmin": x,
                        "cell_ymin": y,
                        'cell_xmax': x + w,
                        'cell_ymax': y + h,
                        'cell_type': cell_type,
                        'cell_type_name': cell_type_name,
                        'class_confidence': class_confidence,
                        'bbox_confidence': 1

                    }
                    cell = Cell(
                        cell_xmin=x,
                        cell_ymin=y,
                        cell_xmax=x + w,
                        cell_ymax=y + h,
                        cell_type=cell_type,
                        cell_type_name=cell_type_name,
                        class_confidence=class_confidence,
                        bbox_confidence=1
                    )
                    cell_list.append(new_one_data)
                    cells.append(cell)
                break
            else:
                time.sleep(0.001)
        tile.add_cells(cells)
        return {
            "ret_code": RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value,
            'cell_count': len(cell_list),
            'cell_list': serialize_non_json_fields(cell_list)
        }
