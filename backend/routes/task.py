from flask import make_response, jsonify, request
from flask_restx import Namespace, Resource, reqparse, fields
from werkzeug.datastructures import FileStorage

from backend.services.task_service import TaskService
from backend.tools.MESSAGE_DICT import RetCode, RetDesc

taskService = TaskService()

task = Namespace('api/v1/smear_analysis', description='任务相关接口')

get_create_task = reqparse.RequestParser()
get_create_task.add_argument('task_id', type=str, required=True, help='任务ID')

create_task_x40 = task.model('create_task_x40', {
    'smear_type': fields.String(required=True, description="涂片类型:BM, PB, CF", default='BM'),
    'dpi': fields.Integer(required=True, description="DPI，模型据此选择: 144750/357378/714756", default=144750),
    'tile_width': fields.Integer(required=True, description="拼图块宽度", default=2448),
    'tile_height': fields.Integer(required=True, description="拼图块高度", default=2048),
    'target_cell_types': fields.String(required=False, description="目标细胞类型如 WBC,MEG / WBC,RBC，供任务模式使用", default=''),
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
    @task.doc(description='上传图片（任务模式）：task_id+row_index+col_index必填')
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

wbc_point_mod = task.model('wbc_point', {
    'x': fields.Integer(required=True, description='x 坐标'),
    'y': fields.Integer(required=True, description='y 坐标'),
    'w': fields.Integer(required=True, description='宽度'),
    'h': fields.Integer(required=True, description='高度'),
})

roi_kwargs_mod = task.model('roi_kwargs', {
    'index_offset': fields.Integer(required=False, description='拍摄任务索引偏移，默认为0', default=0),
    'request_task_num': fields.Integer(required=False, description='请求生成的拍摄任务数量，默认为100', default=100),
    'wbc_points': fields.List(fields.Nested(wbc_point_mod), required=False, description='用户选择的有核细胞框列表'),
})

required_num_mod = task.model('required_num', {
    'WBC': fields.Integer(required=False, description='当 task_type=WBC 且 smear_type=BM 时必填'),
    'MEG': fields.Integer(required=False, description='当 task_type=MEG 或 WBC_MEG 且 smear_type=BM 时必填'),
    'RBC': fields.Integer(required=False, description='当 task_type=RBC 且 smear_type=PB 时必填'),
})

roi_selection_model = task.model('roi_selection', {
    'task_id': fields.String(required=True, description='任务ID'),
    'task_type': fields.String(required=True, description='选区类型：WBC/MEG/WBC_MEG/RBC'),
    'user_choice_area': fields.Nested(user_choice_area_mod, required=False, description='用户框选的扫描区域'),
    'view_width': fields.Integer(required=True, description='拍摄视图宽度'),
    'view_height': fields.Integer(required=True, description='拍摄视图高度'),
    'kwargs': fields.Nested(roi_kwargs_mod, required=False, description='选区算法其他参数'),
    'required_num': fields.Nested(required_num_mod, required=False, description='按 task_type 需要的目标数量'),
})


@task.route('/roi_selection')
class GetTaskListX100(Resource):
    @task.doc(description='获取X100任务列表')
    @task.expect(roi_selection_model)
    def post(self):
        json_data = request.json
        task_id = json_data.get('task_id')
        task_type = json_data.get('task_type')
        user_choice_area = json_data.get('user_choice_area')
        view_width = json_data.get('view_width')
        view_height = json_data.get('view_height')
        kwargs = json_data.get('kwargs') or {}
        required_num = json_data.get('required_num') or {}

        result = taskService.get_task_list_x100(
            task_id=task_id,
            task_type=task_type,
            user_choice_area=user_choice_area,
            view_width=view_width,
            view_height=view_height,
            kwargs=kwargs,
            required_num=required_num,
        )
        return make_response(jsonify(result), 200)


result_x100 = task.parser()
result_x100.add_argument('task_id', type=str, required=False, help='任务ID，由创建任务接口返回，可不填', location='form')
result_x100.add_argument('position_xmin', type=int, required=False, help='左上角在全图中的x坐标', location='form')
result_x100.add_argument('position_ymin', type=int, required=False, help='左上角在全图中的y坐标', location='form')
result_x100.add_argument('position_xmax', type=int, required=False, help='右下角在全图中的x坐标', location='form')
result_x100.add_argument('position_ymax', type=int, required=False, help='右下角在全图中的y坐标', location='form')
result_x100.add_argument('image_file', type=FileStorage, required=True, help='图像文件（.jpg格式）', location='files')
result_x100.add_argument('dpi', type=int, required=True, help='DPI，模型据此选择: 144750/357378/714756', location='form')
result_x100.add_argument('target_cell_types', type=str, required=True,
                         help='目标细胞类型如 WBC,MEG / WBC,RBC / MEG 等，见有效组合表', location='form')
result_x100.add_argument('smear_type', type=str, required=False, help='涂片类型BM/PB/CF，单张识别时使用，有task_id时从任务取', location='form')
result_x100.add_argument('edge_cell_filter', type=bool, required=False, help='是否过滤边缘细胞，默认为true',
                         location='form', default=True)


analyze_slide_model = task.model('analyze_slide', {
    'task_id': fields.String(required=True, description="任务ID，用于获取 info 中的 red_pixel_count/wbc_pixel_count"),
    'analyze_names': fields.List(
        fields.String,
        required=True,
        description="分析项列表，目前可选项只有「增生程度」(cellularity)",
        example=['cellularity']
    ),
})

ALLOWED_ANALYZE_NAMES = {'cellularity'}

generate_views_model = task.model('generate_views', {
    'points': fields.List(
        fields.List(fields.Float),
        required=False,
        description='Point centers [[x,y], [x,y], ...]'
    ),
    'view_width': fields.Integer(required=False, description='View box width', default=384),
    'view_height': fields.Integer(required=False, description='View box height', default=283),
    'pad': fields.Integer(required=False, description='Padding around points', default=100),
})


@task.route('/generate_views')
class GenerateViews(Resource):
    @task.doc(description='Generate minimum number of view boxes to cover all points (set cover). Provide points or cells.')
    @task.expect(generate_views_model)
    def post(self):
        json_data = request.json or {}
        points = json_data.get('points')
        view_width = json_data.get('view_width', 384)
        view_height = json_data.get('view_height', 283)
        pad = json_data.get('pad', 100)
        result = taskService.generate_views(
            points=points,
            view_width=view_width, view_height=view_height, pad=pad
        )
        return make_response(jsonify(result), 200)


@task.route('/analyze_slide')
class AnalyzeSlide(Resource):
    @task.doc(description='玻片分析。实际业务：骨髓玻片增生分析；增生程度=red_pixel_count/wbc_pixel_count')
    @task.expect(analyze_slide_model)
    def post(self):
        json_data = request.json or {}
        task_id = json_data.get('task_id')
        analyze_names = json_data.get('analyze_names', [])
        if not task_id:
            return make_response(jsonify({
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': 'task_id cannot be empty',
                'result': {},
            }), 200)
        if not isinstance(analyze_names, list):
            return make_response(jsonify({
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': RetDesc.CLIENT_ERROR.value,
                'result': {},
            }), 200)
        if not analyze_names:
            return make_response(jsonify({
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': 'analyze_names cannot be empty',
                'result': {},
            }), 200)
        invalid = [n for n in analyze_names if n not in ALLOWED_ANALYZE_NAMES]
        if invalid:
            return make_response(jsonify({
                'ret_code': RetCode.CLIENT_ERROR.value,
                'ret_desc': f'Unsupported analyze item: {invalid}, only supported: {list(ALLOWED_ANALYZE_NAMES)}',
                'result': {},
            }), 200)
        result = taskService.analyze_slide(task_id, analyze_names)
        return make_response(jsonify(result), 200)


@task.route('/analyze_cell_image')
class GetTaskResultX100(Resource):
    @task.doc(description='细胞图像分析。任务模式：task_id+position；单张识别：dpi+algorithm_types必填')
    @task.expect(result_x100)
    def post(self):
        args = result_x100.parse_args()
        task_id = args.get('task_id')
        image_file = args.get('image_file')
        dpi = args.get('dpi')
        target_cell_types = args.get('target_cell_types')
        smear_type = args.get('smear_type')
        edge_cell_filter = args.get('edge_cell_filter', True)
        position_xmin = args.get('position_xmin', None)
        position_ymin = args.get('position_ymin', None)
        position_xmax = args.get('position_xmax', None)
        position_ymax = args.get('position_ymax', None)
        result = taskService.get_task_result_x100(task_id, image_file, target_cell_types, dpi,
                                                  edge_cell_filter, smear_type,
                                                  position_xmin, position_ymin, position_xmax, position_ymax)
        return make_response(jsonify(result), 200)
