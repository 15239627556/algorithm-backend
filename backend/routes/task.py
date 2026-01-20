from flask import make_response, jsonify, request
from flask_restx import Namespace, Resource, reqparse, fields
from werkzeug.datastructures import FileStorage

from backend.services.task_service import TaskService

taskService = TaskService()

task = Namespace('api/v1/smear_analysis', description='任务相关接口')

get_create_task = reqparse.RequestParser()
get_create_task.add_argument('task_id', type=str, required=True, help='任务ID')

create_task_x40 = task.model('create_task_x40', {
    'smear_type': fields.String(required=True, description="涂片类型:BM, PB, CF", default='BM'),
    'dpi': fields.Integer(required=True, description="扫描倍数", default=40),
    'num_rows': fields.Integer(required=True, description="拼图块行数", default=2),
    'num_cols': fields.Integer(required=True, description="拼图块列数", default=2),
    'tile_width': fields.Integer(required=True, description="拼图块宽度", default=2448),
    'tile_height': fields.Integer(required=True, description="拼图块高度", default=2048)
})


@task.route('/create_task')
class CreateTask(Resource):
    @task.doc(description='X40创建任务')
    @task.expect(create_task_x40)
    def post(self):
        json_data = request.json
        result = taskService.create_task(json_data)
        return make_response(jsonify(result), 200)


upload_tile = task.parser()
upload_tile.add_argument('task_id', type=str, required=True, help='任务ID，由创建任务接口返回', location='form')
upload_tile.add_argument('row_index', type=int, required=True, help='拼图块行索引', location='form')
upload_tile.add_argument('col_index', type=int, required=True, help='拼图块列索引', location='form')
upload_tile.add_argument('tile_image', type=FileStorage, required=True, help='图像文件（.jpg格式）', location='files')


@task.route('/upload_tile')
class UploadImage(Resource):
    @task.doc(description='上传图片')
    @task.expect(upload_tile)
    def post(self):
        args = upload_tile.parse_args()
        task_id = args.get('task_id')
        row_index = args.get('row_index')
        col_index = args.get('col_index')
        tile_image = args.get('tile_image')
        result = taskService.upload_image(task_id, row_index, col_index, tile_image)
        return make_response(jsonify(result), 200)


tiles_msg_model = task.model('tiles_msg', {
    'row_index': fields.Integer(required=True, description='拼图块行索引'),
    'col_index': fields.Integer(required=True, description='拼图块列索引'),
    'position_x': fields.Integer(required=True, description='拼图块在全图中的左上角x坐标'),
    'position_y': fields.Integer(required=True, description='拼图块在全图中的左上角y坐标')
})

coordinates_model = task.model('update_coordinates', {
    'task_id': fields.String(required=True, description='任务ID，由创建任务接口返回'),
    'tiles_msg': fields.List(fields.Nested(tiles_msg_model), required=True, description='拼图块坐标信息列表')
})


@task.route('/update_coordinates')
class UpdateCoordinates(Resource):
    @task.doc(description='更新拼图块坐标信息')
    @task.expect(coordinates_model)
    def post(self):
        json_data = request.json
        task_id = json_data.get('task_id')
        tiles_msg = json_data.get('tiles_msg')
        result = taskService.update_coordinates(task_id, tiles_msg)
        return make_response(jsonify(result), 200)


check_image = task.model('check_image', {
    'task_id': fields.String(required=True, description='任务ID')
})


@task.route('/check_missing_tiles')
class CheckImage(Resource):
    @task.doc(description='检查图片是上传完毕')
    @task.expect(check_image)
    def post(self):
        json_data = request.json
        # 获取参数
        task_id = json_data.get('task_id')
        result = taskService.check_image(task_id)
        return make_response(jsonify(result), 200)


@task.route('/check_task_status')
class TaskStatus(Resource):
    @task.doc(description='获取任务状态')
    @task.expect(check_image)
    def post(self):
        args = request.json
        task_id = args.get('task_id')
        result = taskService.task_status(task_id)
        return make_response(jsonify(result), 200)


get_task_result = task.model('get_task_result', {
    'task_id': fields.String(required=True, description='任务ID'),
    'roi_xmin': fields.Integer(required=False, description="结果区域左上角x坐标，默认为0"),
    'roi_ymin': fields.Integer(required=False, description="结果区域左上角y坐标，默认为0"),
    'roi_xmax': fields.Integer(required=False, description="结果区域右下角x坐标，默认为图像宽度"),
    'roi_ymax': fields.Integer(required=False, description="结果区域右下角y坐标，默认为图像高度"),
    'index_offset': fields.Integer(required=False, description="结果索引偏移，默认为0"),
    'request_task_num': fields.Integer(required=False, description="请求结果数量，默认为100")
})


@task.route('/get_task_result')
class GetResult(Resource):
    @task.doc(description='获取任务结果')
    @task.expect(get_task_result)
    def post(self):
        args = request.json
        task_id = args.get('task_id')
        roi_xmin = args.get('roi_xmin', 0)
        roi_ymin = args.get('roi_ymin', 0)
        roi_xmax = args.get('roi_xmax')
        roi_ymax = args.get('roi_ymax')
        index_offset = args.get('index_offset', 0)
        request_task_num = args.get('request_task_num', 100)
        result = taskService.get_result(task_id, roi_xmin, roi_ymin, roi_xmax, roi_ymax, index_offset, request_task_num)
        return make_response(jsonify(result), 200)


user_choice_area_mod = task.model('user_choice_area', {
    'x_min': fields.Integer(required=False, description='用户框选区域的x最小值'),
    'y_min': fields.Integer(required=False, description='用户框选区域的y最小值'),
    'x_max': fields.Integer(required=False, description='用户框选区域的x最大值'),
    'y_max': fields.Integer(required=False, description='用户框选区域的y最大值'),
})
target_item_mod = task.model('TargetItem', {
    'type': fields.String(required=True, description='目标类型，如 BM_MEG、BM_WBC 等'),
    'count': fields.Integer(required=True, description='该类型目标的数量')
})
get_task_x100 = task.model('get_task_x100', {
    'task_id': fields.String(required=True, description='任务ID'),
    'user_choice_area': fields.Nested(user_choice_area_mod, required=False, description='用户框选的扫描区域'),
    'view_width': fields.Integer(required=True, description='拍摄视图宽度'),
    'view_height': fields.Integer(required=True, description='拍摄视图高度'),
    'target_list': fields.List(fields.Nested(target_item_mod), required=True, description='目标类型及数量列表'),
    'index_offset': fields.Integer(required=False, description='拍摄任务索引偏移，默认为0'),
    'request_task_num': fields.Integer(required=False, description='请求生成的拍摄任务数量，默认为100', default=100),
})


@task.route('/roi_selection')
class GetTaskListX100(Resource):
    @task.doc(description='获取X100任务列表')
    @task.expect(get_task_x100)
    def post(self):
        json_data = request.json
        task_id = json_data.get('task_id')
        user_choice_area = json_data.get('user_choice_area')
        view_width = json_data.get('view_width')
        view_height = json_data.get('view_height')
        target_list = json_data.get('target_list')
        index_offset = json_data.get('index_offset', 0)
        request_task_num = json_data.get('request_task_num', 100)
        result = taskService.get_task_list_x100(task_id, user_choice_area, view_width, view_height, target_list,
                                                index_offset, request_task_num)
        return make_response(jsonify(result), 200)


result_x100 = task.parser()
result_x100.add_argument('task_id', type=str, required=False, help='任务ID，由创建任务接口返回，可不填', location='form')
result_x100.add_argument('position_xmin', type=int, required=False, help='左上角在全图中的x坐标', location='form')
result_x100.add_argument('position_ymin', type=int, required=False, help='左上角在全图中的y坐标', location='form')
result_x100.add_argument('position_xmax', type=int, required=False, help='右下角在全图中的x坐标', location='form')
result_x100.add_argument('position_ymax', type=int, required=False, help='右下角在全图中的y坐标', location='form')
result_x100.add_argument('image_file', type=FileStorage, required=True, help='图像文件（.jpg格式）', location='files')
result_x100.add_argument('dpi', type=int, required=True, help='放大倍数', location='form')
result_x100.add_argument('algorithm_types', type=str, required=True,
                         help='任务类型，取值范围: BM_WBC, BM_MEG, BM_RBC, PB_WBC, PB_RBC, CF_WBC', location='form')
result_x100.add_argument('edge_cell_filter', type=bool, required=False, help='是否过滤边缘细胞，默认为true',
                         location='form', default=True)


@task.route('/analyze_cell_image')
class GetTaskResultX100(Resource):
    @task.doc(description='获取X100任务结果')
    @task.expect(result_x100)
    def post(self):
        args = result_x100.parse_args()
        task_id = args.get('task_id')
        image_file = args.get('image_file')
        dpi = args.get('dpi')
        algorithm_types = args.get('algorithm_types')
        edge_cell_filter = args.get('edge_cell_filter', True)
        position_xmin = args.get('position_xmin', None)
        position_ymin = args.get('position_ymin', None)
        position_xmax = args.get('position_xmax', None)
        position_ymax = args.get('position_ymax', None)
        result = taskService.get_task_result_x100(task_id, image_file, algorithm_types, dpi,
                                                  edge_cell_filter, position_xmin, position_ymin,
                                                  position_xmax, position_ymax)
        return make_response(jsonify(result), 200)
