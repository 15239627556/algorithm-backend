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
from backend.tools.MESSAGE_DICT import RET_CODE, RET_DESC
from backend.tools.public_methods import thread_decorator, upload_folder, images_folder
from backend.tools.json_safe_writer import serialize_non_json_fields
from project.smear_project import SmearProject
from project.inference_queue_manager import TileInferenceQueueManager
from project.smear_project import MagnificationLevel, TaskStatus
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
        project = task_service.project[task_id]

        position_x = int(msg.tile_meta["position_x"])
        position_y = int(msg.tile_meta["position_y"])
        image_uid = msg.tile_meta["image_uid"]

        # 先创建 tile，确保结果回来能找到 tile
        project.add_tile(
            magnification=MagnificationLevel.X40,
            row_index=msg.row_index,
            col_index=msg.col_index,
            position_x=position_x,
            position_y=position_y,
            image_uid=image_uid,
        )
        # B方案：只提交 bytes，不在 Flask 进程 decode / 不触碰模型
        get_queue_manager().submit_tile_bytes(
            project_task_id=task_id,
            magnification=MagnificationLevel.X40,
            row_index=msg.row_index,
            col_index=msg.col_index,
            tile_bytes=msg.tile_bytes,
            tile_meta=msg.tile_meta,
            extra=None,
        )

    return on_tile


class TaskService:
    def __init__(self):
        self.project = {}
        self.project_x100 = {}
        self.grids = {}
        self.project_lock = {}
        self.task_id = None

    def load_data(self, task_id):
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, f"{task_id}.smear.pkl")
        if not os.path.exists(file_path):
            return {"ret_code": RET_CODE.CLIENT_ERROR.value,
                    "ret_desc": RET_DESC.CLIENT_ERROR.value,
                    'reason': '任务ID不存在',
                    'msg': f"file for task_id '{task_id}' not found."
                    }
        self.project[task_id] = SmearProject.load_pickle(task_id, upload_folder)

    def create_task(self, task_info: dict) -> dict:
        # B+预热：确保 inference 进程已 READY
        get_queue_manager()

        task_id = uuid.uuid4().hex
        self.task_id = task_id
        task_info['task_id'] = task_id
        task_info['task_status'] = 101  # 101: 进行中, 100: 已完成
        task_info['smear_type'] = task_info.get('smear_type', 'BM')  # bm为骨髓
        num_rows = task_info.get('num_rows')
        num_cols = task_info.get('num_cols')
        tile_width = task_info.get('tile_width', 2448)
        tile_height = task_info.get('pixel_height', 2048)

        project = SmearProject(task_id=task_id, smear_type=task_info['smear_type'],
                               dpi=task_info['dpi'], num_rows=num_rows, num_cols=num_cols,
                               tile_width=tile_width, tile_height=tile_height)

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

        print('创建任务成功：', task_id)
        get_queue_manager().register_project(project)
        get_queue_manager().set_expected_tiles(task_id, num_rows * num_cols)

        return {
            'task_id': task_id,
            'ret_code': RET_CODE.API_SUCCESS.value,
            'ret_desc': RET_DESC.API_SUCCESS.value
        }

    def upload_image(self, task_id, row_index, col_index, position_x, position_y, tile_image):
        if self.task_id is None:
            self.task_id = task_id
        if task_id != self.task_id:
            return {
                'ret_code': RET_CODE.TASK_IN_PROGRESS.value,
                'ret_desc': RET_DESC.TASK_IN_PROGRESS.value,
            }
        image_bytes = tile_image.read()
        if len(image_bytes) == 0:
            return {
                'ret_code': RET_CODE.CLIENT_ERROR.value,
                'ret_desc': RET_DESC.CLIENT_ERROR.value,
            }
        row_index, col_index, position_x, position_y = int(row_index), int(col_index), int(position_x), int(position_y)
        with self.project_lock[task_id]:
            grid = self.grids[task_id]
            if grid[row_index, col_index]:
                return {
                    'ret_code': RET_CODE.IMAGE_ALREADY_UPLOADED.value,
                    'ret_desc': RET_DESC.IMAGE_ALREADY_UPLOADED.value,
                }
            grid[row_index, col_index] = True
            finished = grid.all()
            image_uid = uuid.uuid4().hex
            tile_router.push_tile(
                task_id=task_id,
                row_index=row_index,
                col_index=col_index,
                tile_bytes=image_bytes,
                tile_meta={
                    "position_x": position_x,
                    "position_y": position_y,
                    "image_uid": image_uid
                }
            )
            # 不要在接口线程里 join（会卡住接口）。把收尾全部放后台线程。
            if finished:
                self._finish_task_async(task_id)
            return {
                'ret_code': RET_CODE.API_SUCCESS.value,
                'ret_desc': RET_DESC.API_SUCCESS.value,
                'image_uid': image_uid
            }

    @thread_decorator
    def _finish_task_async(self, task_id: str):
        try:
            tile_router.finish_task(task_id)
            tile_router.join_task(task_id)

            project = self.project[task_id]
            get_queue_manager().finish_tile(task_id, MagnificationLevel.X40)
            print('等待所有图像处理完成：============================================', task_id)
            get_queue_manager().wait_written_all(task_id, timeout=300.0)
            print('所有图像处理完成，开始选区：============================================', task_id)
            project.set_task_status(TaskStatus.COMPLETED)  # 100: 已完成
            print('准备pickle保存：============================================', task_id)
            project.save_pickle(upload_folder)
            print('任务完成：============================================', task_id)
        except Exception as e:
            print("[finish_task_async ERROR]", task_id, repr(e))

    def check_image(self, task_id: str) -> dict:
        if task_id not in self.project:
            result = self.load_data(task_id)
            if result:
                return result
        project = self.project[task_id]
        missing_tiles = project.check_missing_tiles()
        return {
            'ret_code': RET_CODE.API_SUCCESS.value,
            'ret_desc': RET_DESC.API_SUCCESS.value,
            'missing_tiles': missing_tiles
        }

    @staticmethod
    def get_desc(code_value: int) -> str:
        try:
            code_enum = RET_CODE(code_value)  # 反查枚举
            desc_enum = RET_DESC[code_enum.name]  # 名称映射
            return desc_enum.value
        except Exception:
            return "未知状态码"

    def task_status(self, task_id: str) -> dict:
        if task_id not in self.project:
            result = self.load_data(task_id)
            if result:
                return result
        project = self.project[task_id]
        task_status = project.task_status
        return {
            'ret_code': RET_CODE.API_SUCCESS.value,
            'ret_desc': RET_DESC.API_SUCCESS.value,
            'task_status': self.get_desc(task_status.value)
        }

    def get_result(self, task_id: str, roi_xmin, roi_ymin, roi_xmax, roi_ymax) -> dict:
        if task_id not in self.project:
            result = self.load_data(task_id)
            if result:
                return result
        project = self.project[task_id]
        if not roi_xmax or not roi_ymax:
            layer = project.get_layer(MagnificationLevel.X40)
            roi_xmax = layer.num_rows
            roi_ymax = layer.num_cols
        cell_list = project.get_cells_in_roi(MagnificationLevel.X40, roi_xmin, roi_ymin, roi_xmax, roi_ymax)
        return {
            'ret_code': RET_CODE.API_SUCCESS.value,
            'ret_desc': RET_DESC.API_SUCCESS.value,
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
        layer = project.get_layer(MagnificationLevel.X40)
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
        if not self.project_x100[task_id]:
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
                                                                   user_choice_area,
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
            'ret_code': RET_CODE.API_SUCCESS.value,
            'ret_desc': RET_DESC.API_SUCCESS.value,
            'task_list_num': len(new_task_list),
            'task_list': serialize_non_json_fields(new_task_list[index_offset:index_offset + request_task_num])
        }

    @staticmethod
    def get_task_result_x100(task_id, image_file, smear_type, magnification, task_type,
                             camera_type, edge_cell_filter):
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
                    one_data = {'cell_pos': {'x': cellRects[i][0], 'y': cellRects[i][1], 'width': cellRects[i][2],
                                             'height': cellRects[i][3]},
                                'cell_types': {
                                    f"top{j + 1}": {'type': int(cellTypes[i][j]), 'pecent': float(cellRatios[i][j])}
                                    for j in range(len(cellTypes[i]))
                                }}
                    cell_list.append(one_data)
                break
            else:
                time.sleep(0.001)
        return {
            "ret_code": RET_CODE.API_SUCCESS.value,
            'ret_desc': RET_DESC.API_SUCCESS.value,
            'cell_count': len(cell_list),
            'cell_list': serialize_non_json_fields(cell_list)
        }
