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
upload_tile.add_argument('position_x', type=int, required=False, help='拼图块在全图中的x坐标', location='form')
upload_tile.add_argument('position_y', type=int, required=False, help='拼图块在全图中的y坐标', location='form')
upload_tile.add_argument('tile_image', type=FileStorage, required=False, help='图像文件（.jpg格式）', location='files')


@task.route('/upload_tile')
class UploadImage(Resource):
    @task.doc(description='上传图片')
    @task.expect(upload_tile)
    def post(self):
        args = upload_tile.parse_args()
        task_id = args.get('task_id')
        row_index = args.get('row_index')
        col_index = args.get('col_index')
        position_x = args.get('position_x')
        position_y = args.get('position_y')
        tile_image = args.get('tile_image')
        result = taskService.upload_image(task_id, row_index, col_index, position_x, position_y, tile_image)
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
    'min_row': fields.Integer(required=True, description="用户框选的扫描区域行号(最小)"),
    'min_col': fields.Integer(required=True, description="用户框选的扫描区域列号(最小)"),
    'max_row': fields.Integer(required=True, description="用户框选的扫描区域行号(最大)"),
    'max_col': fields.Integer(required=True, description="用户框选的扫描区域列号(最大)"),
})


@task.route('/get_task_result')
class GetResult(Resource):
    @task.doc(description='获取任务结果')
    @task.expect(get_task_result)
    def post(self):
        args = request.json
        task_id = args.get('task_id')
        min_row = args.get('min_row', 0)
        min_col = args.get('min_col', 0)
        max_row = args.get('max_row')
        max_col = args.get('max_col')
        result = taskService.get_result(task_id, min_row, max_row, min_col, max_col)
        return make_response(jsonify(result), 200)


user_choice_area = task.model('user_choice_area', {
    'x_min': fields.Integer(required=True, description='用户框选区域的x最小值'),
    'ymin': fields.Integer(required=True, description='用户框选区域的y最小值'),
    'x_max': fields.Integer(required=True, description='用户框选区域的x最大值'),
    'y_max': fields.Integer(required=True, description='用户框选区域的y最大值'),
})
get_task_x100 = task.model('get_task_x100', {
    'task_id': fields.String(required=True, description='任务ID'),
    'user_choice_area': fields.Nested(user_choice_area, required=True, description='用户框选的扫描区域'),
    'target_num_WBC': fields.Integer(required=True, description='目标白细胞数量'),
    'target_num_MEG': fields.Integer(required=True, description='目标巨核细胞数量'),
    'index_offset': fields.Integer(required=True, description='拍摄任务索引偏移，默认为0'),
    'request_task_num': fields.Integer(required=True, description='请求生成的拍摄任务数量，默认为100', default=100),
})


@task.route('/generate_100x_task_list_bm')
class GetTaskListX100(Resource):
    @task.doc(description='获取X100任务列表')
    @task.expect(get_task_x100)
    def post(self):
        json_data = request.json
        task_id = json_data.get('task_id')
        user_choice_area = json_data.get('user_choice_area')
        view_width = json_data.get('view_width')
        view_height = json_data.get('view_height')
        target_num_WBC = json_data.get('target_num_WBC')
        target_num_MEG = json_data.get('target_num_MEG')
        index_offset = json_data.get('index_offset', 0)
        request_task_num = json_data.get('request_task_num', 1000)
        result = taskService.get_task_list_x100(task_id, user_choice_area, view_width, view_height, target_num_WBC,
                                                target_num_MEG, index_offset, request_task_num)
        return make_response(jsonify(result), 200)


result_x100 = task.parser()
result_x100.add_argument('task_id', type=str, required=False, help='任务ID，由创建任务接口返回，可不填', location='form')
result_x100.add_argument('image_file', type=FileStorage, required=True, help='图像文件（.jpg格式）', location='files')
result_x100.add_argument('smear_type', type=str, required=True, help='涂片类型，取值范围：BM, PB, CF', location='form')
result_x100.add_argument('magnification', type=int, required=True, help='放大倍数', location='form')
result_x100.add_argument('task_type', type=str, required=True, help='任务类型，取值范围: WBC, RBC, MEG', location='form')
result_x100.add_argument('camera_type', type=str, required=True, help='相机类型，取值范围：TYPE_A, TYPE_B',
                         location='form')
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
        smear_type = args.get('smear_type')
        magnification = args.get('magnification')
        task_type = args.get('task_type')
        camera_type = args.get('camera_type')
        edge_cell_filter = args.get('edge_cell_filter')
        result = taskService.get_task_result_x100(task_id, image_file, smear_type, magnification, task_type,
                                                  camera_type, edge_cell_filter)
        return make_response(jsonify(result), 200)
