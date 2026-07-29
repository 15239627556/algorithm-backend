from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse, Response

from backend.tools.triton_client import infer_image_enhance, infer_opencv_enhance

ImgFilter = APIRouter(prefix="/img_filter", tags=["图片滤镜接口"])


def _image_filter_response(
    filename: str,
    enhanced_bytes: bytes,
    content_type: str = "image/jpeg",
) -> Response:
    return Response(
        content=enhanced_bytes,
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@ImgFilter.post("/x100_img_filter", summary="x100 滤镜")
def x100_img_filter(image_file: UploadFile = File(..., description="图像文件（.jpg格式）")):
    """上传图片并应用 x100 滤镜（multi_pipeline opencv_enhance）。"""
    filename = image_file.filename or "enhanced.jpg"
    image_bytes = image_file.file.read()
    if not image_bytes:
        return JSONResponse(status_code=400, content={"message": "无效的图片文件"})
    try:
        enhanced_bytes, content_type = infer_opencv_enhance(image_bytes)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Filter inference failed: {e}"})
    return _image_filter_response(filename, enhanced_bytes, content_type)


@ImgFilter.post("/x40_img_filter_pt", summary="x40 超分辨率滤镜（普通模式）")
def x40_img_filter_pt(image_file: UploadFile = File(..., description="图像文件（.jpg格式）")):
    """上传图片并应用 x40 普通滤镜（multi_pipeline opencv_enhance）。"""
    filename = image_file.filename or "enhanced.jpg"
    image_bytes = image_file.file.read()
    if not image_bytes:
        return JSONResponse(status_code=400, content={"message": "无效的图片文件"})
    try:
        enhanced_bytes, content_type = infer_opencv_enhance(image_bytes)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Filter inference failed: {e}"})
    return _image_filter_response(filename, enhanced_bytes, content_type)


@ImgFilter.post("/x40_img_filter", summary="x40 超分辨率滤镜（深度学习模式）")
def x40_img_filter(image_file: UploadFile = File(..., description="图像文件（.jpg格式）")):
    """上传图片并应用 x40 深度学习滤镜（multi_pipeline image_enhance）。"""
    filename = image_file.filename or "enhanced.jpg"
    image_bytes = image_file.file.read()
    if not image_bytes:
        return JSONResponse(status_code=400, content={"message": "无效的图片文件"})
    try:
        enhanced_bytes, content_type = infer_image_enhance(image_bytes)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Filter inference failed: {e}"})
    return _image_filter_response(filename, enhanced_bytes, content_type)
