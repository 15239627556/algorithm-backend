import time

from flask_restx import Namespace, Resource
from werkzeug.datastructures import FileStorage
import sys
import cv2
from io import BytesIO
import numpy as np
from flask import send_file
from algorithms.x40enhance import X40ImageEnhanceModels
from backend.tools import x100EnhancemModule, x40EnhancemModule

dispatcher = X40ImageEnhanceModels.X40ImageEnhanceModels(num_workers=1)

ImgFilter = Namespace('img_filter', description='图片滤镜接口')

post_img = ImgFilter.parser()
post_img.add_argument('image_file', type=FileStorage, required=True, help='图像文件（.jpg格式）',
                      location='files')


@ImgFilter.route('/x100_img_filter')
class UploadImg(Resource):
    @ImgFilter.doc(description='上传图片并应用滤镜')
    @ImgFilter.expect(post_img)
    def post(self):
        """
        上传图片并应用滤镜
        :return: 返回处理后的图片
        """
        args = post_img.parse_args()
        image_file = args.get('image_file')
        if not image_file or not isinstance(image_file, FileStorage):
            return {"message": "无效的图片文件"}, 400
        filename = image_file.filename
        image_bytes = image_file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(np_arr, cv2.COLOR_BGR2RGB)
        out = x100EnhancemModule.x100PicEnhance(img_np)
        _, img_encode = cv2.imencode('.jpg', out)
        new_img_bytes = img_encode.tobytes()
        # 返回处理后的图像
        return send_file(
            BytesIO(new_img_bytes),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=filename
        )


@ImgFilter.route('/x40_img_filter_pt')
class X40ImageFilterPt(Resource):
    @ImgFilter.doc(description='上传图片并应用超分辨率滤镜（普通模式）')
    @ImgFilter.expect(post_img)
    def post(self):
        """
        上传图片并应用滤镜
        :return: 返回处理后的图片
        """
        args = post_img.parse_args()
        image_file = args.get('image_file')
        if not image_file or not isinstance(image_file, FileStorage):
            return {"message": "无效的图片文件"}, 400
        filename = image_file.filename
        image_bytes = image_file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(np_arr, cv2.COLOR_BGR2RGB)
        out = x40EnhancemModule.x40PicEnhance(img_np)
        _, img_encode = cv2.imencode('.jpg', out)
        new_img_bytes = img_encode.tobytes()
        # 返回处理后的图像
        return send_file(
            BytesIO(new_img_bytes),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=filename
        )


@ImgFilter.route('/x40_img_filter')
class X40ImageFilter(Resource):
    @ImgFilter.doc(description='上传图片并应用超分辨率滤镜(深度学习模式)')
    @ImgFilter.expect(post_img)
    def post(self):
        """
        上传图片并应用滤镜
        :return: 返回处理后的图片
        """
        args = post_img.parse_args()
        image_file = args.get('image_file')
        if not image_file or not isinstance(image_file, FileStorage):
            return {"message": "无效的图片文件"}, 400
        filename = image_file.filename
        image_bytes = image_file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR_BGR)
        task_id = dispatcher.enqueue_task(image)
        img = ""
        for _ in range(7200000):  # 最多等待2小时
            new_result = dispatcher.get_result(task_id)
            if new_result.get('enhance_arr') is not None:
                img = new_result["enhance_arr"]
                break
            time.sleep(0.01)
        _, img_encode = cv2.imencode('.jpg', img)
        new_img_bytes = img_encode.tobytes()
        # 返回处理后的图像
        return send_file(
            BytesIO(new_img_bytes),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=filename
        )
