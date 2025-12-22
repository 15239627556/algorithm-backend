import os
import threading
import multiprocessing as mp
import time
import uuid

import cv2
import numpy as np
from itertools import chain

from algorithms.SelectArea.x40_BoneMarrow_SelectArea import select_and_generate_bestArea_capture_tasks
from algorithms.SelectArea.dedup_cells_across_tiles import dedup_cells_across_tiles
from algorithms.x100model import X100ImageModels
from backend.tools.MESSAGE_DICT import RetCode, RetDesc, TaskType, CELL_TYPES_X100
from backend.tools.public_methods import thread_decorator, upload_folder, images_folder
from backend.tools.json_safe_writer import serialize_non_json_fields
from project.smear_project import SmearProject
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
        project = SmearProject(smear_type=task_info['smear_type'], dpi=task_info['dpi'])
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

    def upload_image(self, task_id, row_index, col_index, position_x, position_y, tile_image):
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
        # 当不存在瓦片数据的时候，只更新位置信息
        if tile_image is None:
            if position_x is None or position_y is None:
                return {
                    'ret_code': RetCode.CLIENT_ERROR.value,
                    'ret_desc': RetDesc.CLIENT_ERROR.value,
                    'reason': 'position_x and position_y are required when tile_image is None'
                }
            with self.project_lock[task_id]:
                self.grids_position[task_id][row_index, col_index] = True
                position_x, position_y = int(position_x), int(position_y)
                flag = 0
                if not image_uid:
                    tiles = layer.iter_tiles()
                    for tile in tiles:
                        r_idx = tile.meta.get('row_index')
                        c_idx = tile.meta.get('col_index')
                        if r_idx == row_index and c_idx == col_index:
                            tile.x = position_x
                            tile.y = position_y
                            flag += 1
                            break
                else:
                    tile = layer.get_tile(image_uid)
                    flag += 1
                    tile.x = position_x
                    tile.y = position_y
                if flag == 0:
                    image_uid = layer.add_tile(
                        x=position_x,
                        y=position_y,
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
                    matcher[(row_index, col_index)] = image_uid
                if self.grids_position[task_id].all():
                    project.save_json(os.path.join(upload_folder, f"{task_id}.json"))
        else:
            image_bytes = tile_image.read()
            with self.project_lock[task_id]:
                grid = self.grids[task_id]
                grid[row_index, col_index] = True
                finished = grid.all()
            if not image_uid:
                image_uid = layer.add_tile(
                    x=position_x,
                    y=position_y,
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
                matcher[(row_index, col_index)] = image_uid
            tile_router.push_tile(
                task_id=task_id,
                image_uid=image_uid,
                position_x=position_x,
                position_y=position_y,
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

    def get_result(self, task_id: str, roi_xmin, roi_ymin, roi_xmax, roi_ymax) -> dict:
        if task_id not in self.project:
            result = self.load_data(task_id)
            if result:
                return result
        project = self.project[task_id]
        layer = project.list_layers()[0]
        cell_list = layer.iter_cells_in_roi(roi_xmin, roi_ymin, roi_xmax, roi_ymax)
        return {
            'ret_code': RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value,
            'cell_count': len(cell_list),
            'cell_list': cell_list
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

    def get_task_result_x100(self, task_id, image_file, smear_type, dpi, task_type,
                             edge_cell_filter, *args):
        if None in args:
            return {
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': RetDesc.CLIENT_ERROR.value
            }
        try:
            task_type = TaskType[task_type].value
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
        # layer = project.get_layer(dpi)
        pass
        # position_xmin, position_ymin, position_xmax, position_ymax = args
        # project.add_tile()
        image_bytes = image_file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR_BGR)
        new_dispatcher = dispatcher
        task_id = new_dispatcher.enqueue_task(image, task_type)
        cell_list = []
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
                    # one_data = {'cell_pos': {'x': cellRects[i][0], 'y': cellRects[i][1], 'width': cellRects[i][2],
                    #                          'height': cellRects[i][3]},
                    #             'cell_types': {
                    #                 f"top{j + 1}": {'type': int(cellTypes[i][j]), 'pecent': float(cellRatios[i][j])}
                    #                 for j in range(len(cellTypes[i]))
                    #             }}
                    cell_list.append(new_one_data)
                break
            else:
                time.sleep(0.001)
        return {
            "ret_code": RetCode.API_SUCCESS.value,
            'ret_desc': RetDesc.API_SUCCESS.value,
            'cell_count': len(cell_list),
            'cell_list': serialize_non_json_fields(cell_list)
        }
