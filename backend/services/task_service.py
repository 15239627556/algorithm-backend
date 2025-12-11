import os
import threading
import time
import uuid
from datetime import datetime

import cv2
import numpy as np
import werkzeug.datastructures
from itertools import chain
from functools import wraps
from copy import deepcopy

from algorithms.SelectArea.x40_BoneMarrow_SelectArea import select_and_generate_bestArea_capture_tasks
from algorithms.x100model import X100ImageModels
from backend.tools.MESSAGE_DICT import RET_CODE, RET_DESC
from backend.tools.public_methods import thread_decorator, upload_folder, images_folder
from backend.tools.json_safe_writer import serialize_non_json_fields
from project.smear_project import SmearProject
from project.inference_queue_manager import TileInferenceQueueManager
from project.smear_project import MagnificationLevel

QueueManager = TileInferenceQueueManager()

dispatcher = X100ImageModels.X100ImageModels(num_workers=1)


class TaskService:
    def __init__(self):
        # project
        # {task_id: {"task_info": task_info, "project": ZWXKScanProject, grid: nx.Graph}}
        self.project = {}
        # 当前正在运行的任务id
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
        self.project[task_id] = project
        project.save_pickle(upload_folder)
        print('创建任务成功：', task_id)
        self.task_id = task_id
        QueueManager.register_project(project)
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
        row_index, col_index, position_x, position_y = int(row_index), int(col_index), int(position_x), int(position_y)
        project = self.project[task_id]
        print('上传图片：', task_id, row_index, col_index, position_x, position_y)
        # 判断上传的图片是否为空
        image_bytes = tile_image.read()
        if len(image_bytes) == 0:
            return {
                'ret_code': RET_CODE.CLIENT_ERROR.value,
                'ret_desc': RET_DESC.CLIENT_ERROR.value,
            }
        image_uid = uuid.uuid4().hex
        project.add_tile(
            magnification=MagnificationLevel.X40,
            row_index=row_index,
            col_index=col_index,
            position_x=position_x,
            position_y=position_y,
            image_uid=image_uid,
        )
        print('add_tile done')
        # np_arr = np.frombuffer(image_bytes, dtype=np.uint8).copy()
        # img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        QueueManager.submit_tile(
            project_task_id=task_id,
            magnification=MagnificationLevel.X40,
            row_index=row_index,
            col_index=col_index,
            image=cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        )
        return {
            'ret_code': RET_CODE.API_SUCCESS.value,
            'ret_desc': RET_DESC.API_SUCCESS.value,
            'image_uid': image_uid
        }

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

    def get_result(self, task_id: str, min_row: int, max_row: int, min_col: int, max_col: int) -> dict:
        if task_id not in self.project:
            result = self.load_data(task_id)
            if result:
                return result
        project = self.project[task_id]
        if not max_row or not max_col:
            layer = project.get_layer(MagnificationLevel.X40)
            max_row = layer.num_rows - 1
            max_col = layer.num_cols - 1
        cell_list = project.get_cells_in_roi(MagnificationLevel.X40, min_row, max_row, min_col, max_col)
        return {
            'ret_code': RET_CODE.API_SUCCESS.value,
            'ret_desc': RET_DESC.API_SUCCESS.value,
            'cell_count': len(cell_list),
            'cell_list': cell_list
        }

    def get_node(self, task_id):
        pass

    def get_task_list_x100(self, task_id, user_choice_area, view_width, view_height, target_num_WBC, target_num_MEG,
                           index_offset, request_task_num):
        project = self.project[task_id]
        if project.task_status != 100:
            return {
                'ret_code': RET_CODE.TASK_RUNNING.value,
                'ret_desc': RET_DESC.TASK_RUNNING.value,
            }
        infos_40xtile = []
        layer = project.get_layer(MagnificationLevel.X40)
        if not user_choice_area:
            print('user_choose_area is None, use full area')
            user_choice_area = {
                'min_row': 0,
                'max_row': layer.num_rows - 1,
                'min_col': 0,
                'max_col': layer.num_cols - 1,
            }
        min_row = user_choice_area['min_row']
        max_row = user_choice_area['max_row']
        min_col = user_choice_area['min_col']
        max_col = user_choice_area['max_col']
        if not self.project[task_id].get(f'task_list_x100_{min_row}_{max_row}_{min_col}_{max_col}'):
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    # todo
                    data = project.nodes[(row, col)]
                    position = data['position']
                    meg_center_pt = data['meg_center_pt']
                    global_cell_rects_dedup = data['global_cell_rects_dedup']
                    local_cell_rects = data['wbc_center_pt']
                    scores = data['area_score_info']
                    new_scores = {
                        '0_0': [scores[0][4], scores[0][5]],
                        '1_0': [scores[1][4], scores[1][5]],
                        '0_1': [scores[2][4], scores[2][5]],
                        '1_1': [scores[3][4], scores[3][5]],
                    }
                    if not data:
                        continue
                    abs_40xtile_x = position[0]
                    abs_40xtile_y = position[1]
                    global_cell_rects = [
                        [one[0] + abs_40xtile_x, one[1] + abs_40xtile_y, one[2] + abs_40xtile_x, one[3] + abs_40xtile_y,
                         one[4]] for one in
                        local_cell_rects]
                    infos_40xtile.append({
                        'index_40xtile_x': col,
                        'index_40xtile_y': layer.num_rows - 1 - row,
                        'abs_40xtile_x': abs_40xtile_x,
                        'abs_40xtile_y': abs_40xtile_y,
                        "local_cell_rects": local_cell_rects,
                        "global_cell_rects": global_cell_rects,
                        'global_cell_rects_dedup': global_cell_rects_dedup,
                        "meg_rect": meg_center_pt,
                        'scores': new_scores
                    })
            save_dir = os.path.join(images_folder, task_id)
            os.makedirs(save_dir, exist_ok=True)
            task_list = select_and_generate_bestArea_capture_tasks(infos_40xtile, user_choice_area, 2, request_task_num,
                                                                   save_flag=True, save_dir=save_dir)
            new_task_list = list(chain(*task_list))
            # new_task_list = list(chain.from_iterable(task_list))
            self.project[task_id][f'task_list_x100_{min_row}_{max_row}_{min_col}_{max_col}'] = new_task_list
        else:
            new_task_list = self.project[task_id][f'task_list_x100_{min_row}_{max_row}_{min_col}_{max_col}']
        return {
            'ret_code': RET_CODE.API_SUCCESS.value,
            'ret_desc': RET_DESC.API_SUCCESS.value,
            'task_list_total': len(new_task_list),
            'task_list': new_task_list[index_offset:index_offset + ask_task_num]
        }

    def get_task_result_x40(self, task_id):
        new_project = self.project[task_id]['project']
        match_result = []
        for node in new_project.nodes:
            y, x = node
            data = new_project.nodes[node]
            big_cells = data['meg_center_pt']
            local_cell_centers = data['wbc_center_pt']
            scores = data['area_score_info']
            match_result.append(
                {
                    'x': x,
                    'y': y,
                    'big_cells': big_cells,
                    'local_cell_centers': local_cell_centers,
                    'scores': scores
                }
            )
        return {'ret_code': RET_CODE.API_SUCCESS.value, 'ret_desc': RET_DESC.API_SUCCESS.value,
                'match_result': match_result}

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
