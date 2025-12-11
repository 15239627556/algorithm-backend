"""
作者：znx
功能描述：40倍骨髓选区
日期：2025-06-04
版本：1.0.0
所需参数：
用户需要的有核细胞数量：target_cell_num
热力图朝向（0头右 1头左）: heatmap_orientation
"user_choose_area": {    // 用户框选的X100有核采集区域(40倍图的行列号)，不指定则给全图
    "min_row": 2,          // 有核采集区域行号（最小）
    "min_col": 2,          // 有核采集区域列号（最小）
    "max_row": 20,         // 有核采集区域行号（最大）
    "max_col": 20          // 有核采集区域列号（最大）
  },

40x平扫图信息：infos_40xtile：[{},{},{}...] 注意：只传入用户框选的40倍图信息
{
    40x 图像 X 索引: index_40xtile_x
    40x 图像 Y 索引: index_40xtile_y
    拼图后40x图像左上角 X 坐标: abs_40xtile_x
    拼图后40x图像左上角 Y 坐标: abs_40xtile_y
    拼图前细胞中心点坐标：local_cell_centers  [[local_cell_centers_x, local_cell_centers_y], ...]
    拼图后细胞中心点坐标：global_cell_centers [[global_cell_centers_x, global_cell_centers_y], ...]
    四宫格评分:scores {0_0:score1, 
                     1_0:score2, 
                     0_1:score3, 
                     1_1:score4}
}
输出参数：task_list
更详细参数说明，参考x40_BoneMarrow_SelectArea_参数说明.txt
"""

import os
import cv2
from typing import List, Tuple, Dict, Optional, Any, Set
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from torchvision.ops import nms as torch_nms
from .utils import *
from .dedup_cells_across_tiles import dedup_cells_across_tiles
from .setcover import SetCoverSolverParameter, solve


def scores_matrix(infos_40xtile, user_choose_area):
    """
    构建得分矩阵。

    参数:
    - infos_40xtile: 40x平扫图信息
    - user_choose_area: tile 用户框选的采集X100有核的区域(40x倍行列号)，不框选则全片

    返回:
    - Scores_Matrix: numpy 矩阵，得分结果
    """
    rows_40xtile = user_choose_area['max_row'] - user_choose_area['min_row'] + 1
    cols_40xtile = user_choose_area['max_col'] - user_choose_area['min_col'] + 1
    height = rows_40xtile * 2
    width = cols_40xtile * 2
    Scores_Matrix = np.full((height, width), -1000.0, dtype=np.float64)
    assert len(infos_40xtile) == rows_40xtile * cols_40xtile, "infos_40xtile 数量与 tile 总数不匹配"

    for tile_info in infos_40xtile:
        x_tile = tile_info['index_40xtile_x'] - user_choose_area['min_col']
        y_tile = tile_info['index_40xtile_y'] - user_choose_area['min_row']
        blocks = tile_info['scores']
        assert len(blocks) == 4, "每个 tile 的 block 数量应为 4"

        for block_key, info in blocks.items():
            dx, dy = map(int, block_key.split('_'))
            x_ = dx + x_tile * 2
            y_ = dy + (rows_40xtile - 1 - y_tile) * 2
            Scores_Matrix[y_, x_] = info[0]
    return Scores_Matrix


def compute_head_crop(Scores_Matrix: np.ndarray, heatmap_orientation: int):
    """
    计算图像头部区域的裁剪宽度和裁剪矩形区域

    参数:
    - Scores_Matrix: 2D numpy 数组，代表得分图（灰度图）
    - heatmap_orientation: int，0 表示头部在右边，1 表示头部在左边，其它值表示无需裁剪

    返回:
    - crop_w: int，裁剪宽度
    - crop_rect: dict，裁剪区域坐标，格式为 {'x': int, 'y': int, 'width': int, 'height': int}
    """
    height, width = Scores_Matrix.shape
    matrix = Scores_Matrix.copy()
    matrix[matrix == -1000] = np.nan  # 无效区域设为 nan
    data_min = np.nanmin(matrix)

    # 每列中非空像素数
    valid_mask = Scores_Matrix > data_min
    cols_pixel_stat = np.sum(valid_mask, axis=0)

    # 判断哪些列是“有效的”
    threshold = height // 10
    valid_cols = np.where(cols_pixel_stat > threshold)[0]

    if valid_cols.size == 0:
        crop_w = 0
    else:
        min_col = valid_cols[0]
        max_col = valid_cols[-1]
        available_w = (max_col - min_col) + 1
        crop_w = (available_w // 5) + min_col

    # 生成裁剪区域
    if heatmap_orientation == 0:
        crop_rect = {
            'x': width - crop_w,
            'y': 0,
            'width': crop_w,
            'height': height
        }
    elif heatmap_orientation == 1:
        crop_rect = {
            'x': 0,
            'y': 0,
            'width': crop_w,
            'height': height
        }
    else:
        crop_rect = {
            'x': 0,
            'y': 0,
            'width': 0,
            'height': 0
        }

    return crop_w, crop_rect


def get_points_in_rotated_rect(lt, lb, rt, rb, width, height):
    """
    留到后续，或许可替换get_points_in_rotated_rect可用于提升运算速度
    获取旋转矩形中的所有整数点（像素坐标），使用掩码加速实现。

    参数:
    - lt, lb, rt, rb: 四个顶点（左上、左下、右上、右下），格式为 (x, y)
    - width: 分值图的宽，限制边界，防止超出
    - height: 分值图的高，限制边界，防止超出

    返回:
    - points: List[List[int, int]] 所有在旋转矩形内的像素点坐标
    """

    # 定义轮廓（顺时针或逆时针都可，只要闭合且不交叉）
    contour = np.array([[lt, rt, rb, lb]], dtype=np.int32)

    # 创建掩码图像
    mask = np.zeros((height, width), dtype=np.uint8)

    # 填充多边形区域
    cv2.fillPoly(mask, contour, 1)

    # 查找所有非零点（即在多边形内的像素坐标）
    nonzero = cv2.findNonZero(mask)

    if nonzero is None:
        return []

    # 返回 List[List[int, int]] 格式
    points = [pt[0].tolist() for pt in nonzero]

    return points


def generate_unique_rects(x40_list):
    """
    生成根据放大倍率和宽高比组合后的所有唯一窗口尺寸（奇数宽高）
    
    参数:
    - x40_list: list[float]，假设需要的40倍平扫图的张数列表
    
    返回:
    - rects: list of (width, height) 元组，去重后的窗口尺寸
    """

    ratio_w = [1.25, 2.0, 3.5, 5.0, 1.0, 1.0, 1.0, 1.0]
    ratio_h = [1.0, 1.0, 1.0, 1.0, 1.25, 2.0, 3.5, 5.0]

    assert len(ratio_w) == len(ratio_h)
    rects = []

    for mag in x40_list:
        for rw, rh in zip(ratio_w, ratio_h):
            step = np.sqrt((mag * 8) / (rw * rh))
            w = int(round(step * rw))
            h = int(round(step * rh))

            # 保证是奇数
            if w % 2 == 0:
                w += 1
            if h % 2 == 0:
                h += 1

            rects.append((w, h))

    # 去重并排序
    rects = sorted(set(rects), key=lambda x: (x[0], x[1]))
    return rects


def count_unique_cells_in_rotated_region(rotated_points, infos_40xtile, user_choose_area):
    """
    使用一次性邻接去重后的 global_cell_rects_dedup 直接计数。
    仍按原逻辑：由 rotated_points 映射出涉及到的 40x tile 集合，然后把这些 tile 的去重点数相加。
    """
    # 建索引
    tile_map = {(t['index_40xtile_x'], t['index_40xtile_y']): t for t in infos_40xtile}

    # 一次性向量化把点映射到 40x tile
    pts = np.asarray(rotated_points, dtype=np.int32)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    min_col = int(user_choose_area['min_col'])
    min_row = int(user_choose_area['min_row'])
    max_row = int(user_choose_area['max_row'])
    row_span = max_row - min_row

    fx = (pts[:, 0] // 2) + min_col
    fy = (row_span - (pts[:, 1] // 2)) + min_row

    # 去重 tile
    fined_x40 = np.unique(np.stack([fx, fy], axis=1), axis=0)  # shape: (K, 2)

    total = 0
    for fx, fy in fined_x40:
        t = tile_map[(fx, fy)]
        pts = t.get('global_cell_rects_dedup')
        if pts is None:
            pts = t.get('global_cell_rects', [])
        total += len(pts)
    return total


def find_optimal_rotated_regions(Scores_Matrix, rects, heatmap_orientation, head_crop_rect, infos_40xtile,
                                 user_choose_area, target_cell_num, all_cell_count):
    """
    对给定得分矩阵应用不同尺寸和角度的旋转核，寻找满足细胞数量目标的最佳采样区域。

    参数：
    - Scores_Matrix: np.ndarray
        得分热图，2D 浮点数组，代表整片切片区域得分
    - rects: List[Tuple[int, int]]
        候选旋转框尺寸列表，每项为 (width, height)
    - heatmap_orientation: int
        图像方向标志，0 表示右头，1 表示左头，其它值表示未知方向
    - head_crop_rect: Dict[str, int]
        头部区域裁剪矩形，格式为 {'x', 'y', 'width', 'height'}
    - infos_40xtile: List[Dict]
        每张 40x 图像的细胞信息列表，包含 tile 坐标与细胞中心点坐标等
    - user_choose_area: Dict
        用户框选区域范围，字段包括 'min_row', 'max_row', 'min_col', 'max_col'
    - target_cell_num: int
        用户期望的细胞数量 * rario (目前ratio默认为3)
    - all_cell_count: int
        图像全局的细胞总数

    返回：
    - Dict[str, List[Dict]]
        返回结果包含两个部分：
            - 'head_results': 极值点落在头部区域的旋转区域信息
            - 'tail_results': 极值点落在非头部区域的旋转区域信息
    """
    print('选区时按 {} 进行选区.'.format(target_cell_num))
    if target_cell_num >= all_cell_count:
        print("用户期望的细胞数量大于或等于全图细胞数量，直接返回全图区域。")
        rows_40xtile = user_choose_area['max_row'] - user_choose_area['min_row'] + 1
        cols_40xtile = user_choose_area['max_col'] - user_choose_area['min_col'] + 1
        height = rows_40xtile * 2
        width = cols_40xtile * 2
        lt = (0, 0)
        lb = (0, height - 1)
        rt = (width - 1, 0)
        rb = (width - 1, height - 1)
        rotated_points = get_points_in_rotated_rect(lt, lb, rt, rb, width, height)
        cx = width // 2
        cy = height // 2
        diff = all_cell_count - target_cell_num
        matix = Scores_Matrix.copy()
        matix[matix == -1000] = np.nan
        Area_score = np.nanmean(matix)
        return {
            'head_results': [{
                'Area_score': Area_score,
                'selectArea_count': all_cell_count,
                'diff': diff,
                'angle': 0,
                'rotated_rect_lb': lb,
                'rotated_rect_lt': lt,
                'rotated_rect_rt': rt,
                'rotated_rect_rb': rb,
                'center': (cx, cy),
                'area_rect': (width, height),
                'outPointsAll': rotated_points
            }],
            'tail_results': []
        }

    height, width = Scores_Matrix.shape
    head_results = []
    tail_results = []

    for (w, h) in rects:
        kernel_size = int(np.sqrt(w * w + h * h)) + 4
        if kernel_size % 2 == 0:
            kernel_size += 1
        # 检查核是否大于图像，避免越界
        if kernel_size > height or kernel_size > width:
            continue
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        x0 = int(kernel_size / 2 - w / 2)
        y0 = int(kernel_size / 2 - h / 2)
        kernel[y0:y0 + h, x0:x0 + w] = 1.0
        # plt.imshow(kernel, cmap='gray')
        # plt.show()

        supImag = np.full((height + kernel_size, width + kernel_size), -1000.0, dtype=np.float64)
        supImag[kernel_size // 2:kernel_size // 2 + height, kernel_size // 2:kernel_size // 2 + width] = Scores_Matrix
        # plt.imshow(supImag, cmap='gray')
        # plt.show()

        for angle in [0, 15, 75]:
            M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), -angle, 1)
            rotated_kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size), flags=cv2.INTER_NEAREST)
            rotated_kernel = np.where(rotated_kernel >= 0.5, 1.0, 0.0).astype(np.float32)
            # plt.imshow(rotated_kernel, cmap='gray')
            # plt.show()

            dst = cv2.filter2D(supImag, -1, rotated_kernel / (w * h), borderType=cv2.BORDER_CONSTANT)
            _, maxVal, _, maxLoc = cv2.minMaxLoc(dst)
            cx = int(maxLoc[0] - kernel_size // 2)
            cy = int(maxLoc[1] - kernel_size // 2)

            box = cv2.boxPoints(((cx, cy), (w, h), float(angle)))
            box = np.clip(box, [0, 0], [width - 1, height - 1])
            # 按原点在左上角，x向右越来越大，y向下越来越大，进行排序
            box = sorted(box, key=lambda p: (p[1], p[0]))  # 排序：先按 y，再按 x
            top2 = sorted(box[:2], key=lambda p: p[0])  # 上面两个点，按 x 排序：左→右
            bottom2 = sorted(box[2:], key=lambda p: p[0])  # 下面两个点，按 x 排序：左→右

            lt = top2[0]
            rt = top2[1]
            lb = bottom2[0]
            rb = bottom2[1]

            # 计算头部极值点
            if heatmap_orientation == 0:
                # 右头：找最大 x 点
                extreme_point = max(box, key=lambda pt: pt[0])
            elif heatmap_orientation == 1:
                # 左头：找最小 x 点
                extreme_point = min(box, key=lambda pt: pt[0])
            else:
                extreme_point = (-1.0, -1.0)

            # 旋转矩形中的所有像素点
            rotated_points = get_points_in_rotated_rect(lt, lb, rt, rb, width, height)
            selectArea_count = count_unique_cells_in_rotated_region(rotated_points, infos_40xtile, user_choose_area)
            diff = selectArea_count - target_cell_num

            # 判断极值点是否落在头部区域内
            if (head_crop_rect['x'] <= extreme_point[0] <= head_crop_rect['x'] + head_crop_rect['width'] and
                    head_crop_rect['y'] <= extreme_point[1] <= head_crop_rect['y'] + head_crop_rect['height']):
                head_results.append({
                    'Area_score': maxVal,
                    'selectArea_count': selectArea_count,
                    'diff': diff,
                    'angle': -angle,
                    'rotated_rect_lb': lb,
                    'rotated_rect_lt': lt,
                    'rotated_rect_rt': rt,
                    'rotated_rect_rb': rb,
                    'center': (cx, cy),
                    'area_rect': (w, h),
                    'outPointsAll': rotated_points
                })

            else:
                tail_results.append({
                    'Area_score': maxVal,
                    'selectArea_count': selectArea_count,
                    'diff': diff,
                    'angle': -angle,
                    'rotated_rect_lb': lb,
                    'rotated_rect_lt': lt,
                    'rotated_rect_rt': rt,
                    'rotated_rect_rb': rb,
                    'center': (cx, cy),
                    'area_rect': (w, h),
                    'outPointsAll': rotated_points
                })

    return {'head_results': head_results, 'tail_results': tail_results}


def filter_closest_to_target_count(results, user_choose_area, target_cell_num, all_cell_count, Scores_Matrix):
    """
    从候选区域中筛选出满足目标细胞数量的区域。

    参数：
    - results: dict
        包含 'tail_results' 和 'head_results' 字段，每项为 List[Dict]
    - user_choose_area: dict
        用户框选区域范围，字段包括 'min_row', 'max_row', 'min_col', 'max_col'
    - target_cell_num: int
        用户期望的细胞数量
    - all_cell_count: int
        图像全局的细胞总数

    返回：
    - List[Dict]: 满足条件的区域列表；若都不满足，则返回最大细胞区域
    """
    candidates = results.get('tail_results', []) or results.get('head_results', [])
    print('筛选选区结果时按 {} 进行筛选.'.format(target_cell_num))

    # 候选为空 → 返回全图区域
    if not candidates:
        print("头部选区结果集合和尾部选区结果集合均为空，返回全图区域。")
        rows_40xtile = user_choose_area['max_row'] - user_choose_area['min_row'] + 1
        cols_40xtile = user_choose_area['max_col'] - user_choose_area['min_col'] + 1
        height = rows_40xtile * 2
        width = cols_40xtile * 2
        lt = (0, 0)
        lb = (0, height - 1)
        rt = (width - 1, 0)
        rb = (width - 1, height - 1)
        rotated_points = get_points_in_rotated_rect(lt, lb, rt, rb, width, height)
        diff = all_cell_count - target_cell_num
        cx = width // 2
        cy = height // 2
        matrix = Scores_Matrix.copy()
        matrix[matrix == -1000] = np.nan
        Area_score = np.nanmean(matrix)
        return [{
            'Area_score': Area_score,
            'selectArea_count': all_cell_count,
            'diff': diff,
            'angle': 0,
            'rotated_rect_lb': lb,
            'rotated_rect_lt': lt,
            'rotated_rect_rt': rt,
            'rotated_rect_rb': rb,
            'center': (cx, cy),
            'area_rect': (width, height),
            'outPointsAll': rotated_points
        }]

    # 模拟 select_ratio *= 2 逻辑
    select_ratio = 0.2
    selected = []

    while not selected:
        upper_limit = target_cell_num * (1 + select_ratio)
        for item in candidates:
            if item['diff'] >= 0 and item['selectArea_count'] <= upper_limit:
                selected.append(item)

        select_ratio *= 2

        # 所有比例都不满足，fallback 条件成立
        if target_cell_num * (1 + select_ratio) > all_cell_count:
            print("所有选区数量不满足需要细胞数量，返回细胞数量最多的选区结果。")
            max_item = max(candidates, key=lambda x: x.get('selectArea_count', 0))
            return [max_item]

    return selected


def cal_cell_uniformity(abs_cell_points, abs_area_pionts, scale=20, sup=100,
                        view_width_px=384, view_height_px=283,
                        debug=False, debug_prefix="debug_uniformity"):
    """
    计算细胞分布均匀性得分，优化性能与稳健性。

    参数：
    - abs_cell_points: List[Tuple[float, float]] 所有细胞绝对坐标
    - abs_out_points: List[Tuple[float, float]] 区域轮廓点（如 rotated_rect_*）
    - scale: int 缩放比例
    - sup: int 额外边界补偿
    - view_width_px, view_height_px: int 每个“视野”在像素中对应的宽高
    - debug: bool 是否保存中间图像
    - debug_prefix: str 保存图像前缀路径

    返回：
    - float：细胞分布均匀性得分（越高越均匀）
    """

    # ---------- Step 1：区域边界（向量化） ----------
    area_pts = np.asarray(abs_area_pionts, dtype=np.int64)
    if area_pts.ndim != 2 or area_pts.shape[1] < 2:
        # 无有效区域，直接返回 0
        return 0.0

    abs_x_min = int(area_pts[:, 0].min())
    abs_x_max = int(area_pts[:, 0].max())
    abs_y_min = int(area_pts[:, 1].min())
    abs_y_max = int(area_pts[:, 1].max())

    abs_width = int((abs_x_max - abs_x_min) // scale + sup)
    abs_height = int((abs_y_max - abs_y_min) // scale + sup)
    if abs_width <= 0 or abs_height <= 0:
        return 0.0

    # ---------- Step 2：构建轮廓掩码（向量化） ----------
    # 将区域四点（或多边形）映射到缩放网格
    poly_map = np.empty((1, area_pts.shape[0], 2), dtype=np.int32)
    poly_map[0, :, 0] = ((area_pts[:, 0] - abs_x_min) // scale) + (sup // 2)
    poly_map[0, :, 1] = ((abs_y_max - area_pts[:, 1]) // scale) + (sup // 2)

    mask_polygon = np.zeros((abs_height, abs_width), dtype=np.uint8)
    cv2.fillPoly(mask_polygon, poly_map, 1)

    # ---------- Step 3：填充细胞点（完全向量化） ----------
    # 映射细胞点；过滤越界；仅保留在 polygon 内的点
    cell_pts = np.asarray(abs_cell_points, dtype=np.int64)
    mask_cells = np.zeros((abs_height, abs_width), dtype=np.uint8)

    if cell_pts.size > 0:
        mx = ((cell_pts[:, 0] - abs_x_min) // scale) + (sup // 2)
        my = ((abs_y_max - cell_pts[:, 1]) // scale) + (sup // 2)

        # 越界过滤
        in_w = (mx >= 0) & (mx < abs_width)
        in_h = (my >= 0) & (my < abs_height)
        ok = in_w & in_h
        if np.any(ok):
            mx_ok = mx[ok]
            my_ok = my[ok]
            # 仅填充落在 polygon 内的点
            inside = mask_polygon[my_ok, mx_ok] > 0
            if np.any(inside):
                # 使用高级索引一次性赋值
                mask_cells[my_ok[inside], mx_ok[inside]] = 255

    # ---------- Step 4：均值卷积（用 boxFilter 更快） ----------
    kernel_w = max(1, int(view_width_px // scale))
    kernel_h = max(1, int(view_height_px // scale))
    # 与原先 ones/(w*h) 的 filter2D 等价
    dst = cv2.boxFilter(mask_cells, ddepth=-1, ksize=(kernel_w, kernel_h),
                        normalize=True, borderType=cv2.BORDER_CONSTANT)

    # ---------- Step 5：计算得分（保持原逻辑） ----------
    not_zero_pixels = cv2.countNonZero(dst)
    # 区域面积用原坐标的多边形面积 / scale^2（与旧实现一致）
    area = max(cv2.contourArea(area_pts.astype(np.float32)) / (scale * scale), 1.0)
    not_zero_scale = float(not_zero_pixels) / float(area)

    # ---------- Step 6：可选调试输出 ----------
    if debug:
        os.makedirs(debug_prefix, exist_ok=True)
        cv2.imwrite(f"{debug_prefix}/mask_cells.png", mask_cells)
        dst_vis = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(f"{debug_prefix}/density_filtered.png", dst_vis)

    return not_zero_scale


def select_best_uniform_region_by_area_score(selected_results, infos_40xtile, user_choose_area):
    """
    将候选区域结果按 Area_score 从大到小排序，并返回前 3 个（若不足则全返回），再计算均匀性评分，输出均匀性评分最高的结果。
    
    参数：
    - results: List[Dict] 符合细胞数量的候选区域结果
    - infos_40xtile: List[Dict]
        每张 40x 图像的细胞信息列表，包含 tile 坐标与细胞中心点坐标等

    返回：
    - Dict: {'Area_score': maxVal,
            'selectArea_count': selectArea_count,
            'diff': diff,
            'angle': -angle,
            'rotated_rect_lb': lb,
            'rotated_rect_lt': lt,
            'rotated_rect_rt': rt,
            'rotated_rect_rb': rb,
            'center': (cx, cy),
            'area_rect': (w, h),
            'outPointsAll': rotated_points'
            'uniformity_score': uniformity_score}
    """
    if not selected_results:
        raise ValueError("selected_results 不能为空")
    # 若只有一个结果，直接返回
    if len(selected_results) == 1:
        return selected_results[0]

    # 预先：tuple 作为 key（你已改好）
    cell_dict = {
        (int(tile["index_40xtile_x"]), int(tile["index_40xtile_y"])): tile
        for tile in infos_40xtile
    }

    # 只取 top-3（大列表更高效的办法是 heapq.nlargest，见后附）
    sorted_results = sorted(selected_results, key=lambda x: x.get('Area_score', float('-inf')), reverse=True)[:3]

    for i, result in enumerate(sorted_results):
        rotated_points = result['outPointsAll']

        # Step 1：一次性向量化把点映射到 40x tile（避免在 for 里不断 int()/加减）
        pts = np.asarray(rotated_points, dtype=np.int32)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        min_col = int(user_choose_area['min_col'])
        min_row = int(user_choose_area['min_row'])
        max_row = int(user_choose_area['max_row'])
        row_span = max_row - min_row

        fx = (pts[:, 0] // 2) + min_col
        fy = (row_span - (pts[:, 1] // 2)) + min_row

        # 去重 tile（不必 set+sorted，unique(axis=0) 更快；若无序要求就不要排序）
        tiles = np.unique(np.stack([fx, fy], axis=1), axis=0)  # shape: (K, 2)

        # Step 2：批量取每个 tile 的 global_cell_rects_dedup，并向量化成中心点
        centers_chunks = []  # 收集为数组块，最后一次性 vstack，再转 list
        for fx_i, fy_i in tiles:
            cell_info = cell_dict.get((int(fx_i), int(fy_i)))
            if not cell_info:
                continue
            rects = cell_info.get("global_cell_rects_dedup", [])
            if not rects:
                continue

            a = np.asarray(rects, dtype=np.float32)
            if a.ndim == 1:
                a = a.reshape(1, -1)
            if a.shape[1] < 4:
                continue  # 容错

            # 只取 x1,y1,x2,y2；忽略 score
            # 中心点：rint 再 astype 能得到和你现在一致的整数
            # centers = np.rint(0.5 * (a[:, 0:2] + a[:, 2:4])).astype(np.int32, copy=False)
            centers = 0.5 * (a[:, 0:2] + a[:, 2:4])
            centers_chunks.append(centers)

        # 最终 all_global_centers 仍然是 list[list[int,int]]（与原先一致）
        if centers_chunks:
            all_global_centers = np.vstack(centers_chunks).tolist()
        else:
            all_global_centers = []

        # Step 3: 获取选区四顶点的绝对坐标
        lt = result['rotated_rect_lt']
        x40_lt = (int(lt[0]) // 2 + user_choose_area['min_col'],
                  (user_choose_area['max_row'] - user_choose_area['min_row']) - int(lt[1]) // 2 + user_choose_area[
                      'min_row'])
        abs_lt = (cell_dict.get(x40_lt)['abs_40xtile_x'], cell_dict.get(x40_lt)['abs_40xtile_y'])
        lb = result['rotated_rect_lb']
        x40_lb = (int(lb[0]) // 2 + user_choose_area['min_col'],
                  (user_choose_area['max_row'] - user_choose_area['min_row']) - int(lb[1]) // 2 + user_choose_area[
                      'min_row'])
        abs_lb = (cell_dict.get(x40_lb)['abs_40xtile_x'], cell_dict.get(x40_lb)['abs_40xtile_y'] + 2048)
        rt = result['rotated_rect_rt']
        x40_rt = (int(rt[0]) // 2 + user_choose_area['min_col'],
                  (user_choose_area['max_row'] - user_choose_area['min_row']) - int(rt[1]) // 2 + user_choose_area[
                      'min_row'])
        abs_rt = (cell_dict.get(x40_rt)['abs_40xtile_x'] + 2448, cell_dict.get(x40_rt)['abs_40xtile_y'])
        rb = result['rotated_rect_rb']
        x40_rb = (int(rb[0]) // 2 + user_choose_area['min_col'],
                  (user_choose_area['max_row'] - user_choose_area['min_row']) - int(rb[1]) // 2 + user_choose_area[
                      'min_row'])
        abs_rb = (cell_dict.get(x40_rb)['abs_40xtile_x'] + 2448, cell_dict.get(x40_rb)['abs_40xtile_y'] + 2048)
        abs_area_pionts = [abs_lt, abs_rt, abs_rb, abs_lb]  # 注意顺序：左上、右上、右下、左下 （顺时针）

        # Step 4: 调用均匀性评分函数
        uniformity_score = cal_cell_uniformity(abs_cell_points=all_global_centers, abs_area_pionts=abs_area_pionts)
        #    debug=True,
        #    debug_prefix=os.path.join(save_dir, str(i)))

        # Step 4: 存入结果字典
        result['uniformity_score'] = uniformity_score

    sorted_results = sorted(sorted_results, key=lambda x: x.get('uniformity_score', float('-inf')), reverse=True)

    return sorted_results[0]


def choose_area_points_addition(outPointsAll, Threshold_Matrix, subset_rect):
    """
    计算补拍区域。
    
    参数：
    - outPointsAll: List[List] 选区中剩余未拍摄的点坐标
    - Threshold_Matrix: np.ndarray 初始框选的二值化矩阵
    - subset_rect: 选区已拍区域 (x, y, w, h)

    返回：
    - AdditionRect：补拍区域 (x, y, w, h)
    - updated_rect：更新后的选区矩形 (x, y, w, h)
    """

    matH, matW = Threshold_Matrix.shape
    rx, ry, rw, rh = subset_rect

    rects = [
        (rx, ry - 1, rw, 1),  # top
        (rx, ry + rh, rw, 1),  # bottom
        (rx - 1, ry, 1, rh),  # left
        (rx + rw, ry, 1, rh)  # right
    ]

    def count_rect_hits(rect):
        x, y, w, h = rect
        if x < 0 or y < 0 or x + w > matW or y + h > matH:
            return 0  # skip out-of-bound
        roi = Threshold_Matrix[y:y + h, x:x + w]
        return np.count_nonzero(roi > 0)

    def count_points_in_rects(points, rects):
        points = np.array(points)
        xs, ys = points[:, 0], points[:, 1]
        counts = []
        for x, y, w, h in rects:
            in_rect = (xs >= x) & (xs < x + w) & (ys >= y) & (ys < y + h)
            counts.append(np.count_nonzero(in_rect))
        return counts

    counts = [count_rect_hits(rect) for rect in rects]
    if max(counts) == 0:
        counts = count_points_in_rects(outPointsAll, rects)
        if max(counts) == 0:
            return None, None

    best_idx = counts.index(max(counts))
    AdditionRect = rects[best_idx]

    x1 = min(rx, AdditionRect[0])
    y1 = min(ry, AdditionRect[1])
    x2 = max(rx + rw, AdditionRect[0] + AdditionRect[2])
    y2 = max(ry + rh, AdditionRect[1] + AdditionRect[3])
    updated_rect = (x1, y1, x2 - x1, y2 - y1)

    return AdditionRect, updated_rect


def generate_capture_tasks_from_region_filtered_BMP(selectArea_result, infos_40xtile, target_cell_num,
                                                    Scores_Matrix, user_choose_area, select_ratio=0.3):
    outPointsAll = selectArea_result['outPointsAll']

    # -------------- tile 索引与 label 查询 --------------
    rows_40xtile = user_choose_area['max_row'] - user_choose_area['min_row'] + 1
    cols_40xtile = user_choose_area['max_col'] - user_choose_area['min_col'] + 1
    tile_map = {(t['index_40xtile_x'], t['index_40xtile_y']): t for t in infos_40xtile}

    def pt_to_label(pt):
        x_m, y_m = int(pt[0]), int(pt[1])
        local_tx, dx = divmod(x_m, 2)
        rev_ty, dy = divmod(y_m, 2)
        local_ty = (rows_40xtile - 1) - rev_ty
        ix = user_choose_area['min_col'] + local_tx
        iy = user_choose_area['min_row'] + local_ty

        tinfo = tile_map.get((ix, iy))
        if not tinfo:
            return None
        blk_key = f"{dx}_{dy}"
        val = tinfo['scores'].get(blk_key)
        # print(f"Point {pt} maps to 第四象限 tile ({ix}, {rev_ty}) / 第一象限 tile ({ix}, {iy}) with offset ({dx}, {dy}) and block key '{blk_key}': {val}")
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            return val[1]
        if isinstance(val, dict):
            return val.get('label')
        return None

    # -------------- 一次性过滤掉 label==5 的点（全局预过滤） --------------
    outPointsAll = [p for p in outPointsAll if (pt_to_label(p) not in (None, 5))]

    # -------------- 只做“落在矩形内”的向量化几何筛选 --------------
    def points_in_rect(points, rect):
        """返回 rect 内的点（不再做 label 判定）。"""
        if not points:
            return []
        pts = np.asarray(points, dtype=np.int32)  # (N,2)
        x, y, w, h = rect
        m = (pts[:, 0] >= x) & (pts[:, 0] < x + w) & (pts[:, 1] >= y) & (pts[:, 1] < y + h)
        return pts[m].tolist()

    # --------- 构建旋转区域 mask ----------
    rotated_points = [selectArea_result['rotated_rect_lt'],
                      selectArea_result['rotated_rect_rt'],
                      selectArea_result['rotated_rect_rb'],
                      selectArea_result['rotated_rect_lb']]
    mask = np.zeros_like(Scores_Matrix, dtype=np.uint8)
    cv2.fillPoly(mask, np.array([rotated_points], dtype=np.int32), 1)

    max_thresh = selectArea_result['Area_score'] - 7
    min_thresh = 0.0
    index_result = 0
    subset_outPoints, subset_rects = [], []

    # 兜底：记录最近一次的二值图与连通域矩形
    fallback_rect = None
    last_th = None

    while abs(max_thresh - min_thresh) > 1e-5 and index_result == 0:
        thresh = (max_thresh + min_thresh) / 2
        _, th = cv2.threshold(Scores_Matrix, thresh, 255, cv2.THRESH_BINARY)
        th = np.where(mask == 1, th, 0).astype(np.uint8)
        last_th = th  # 保存最后一次阈值图

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(th)
        if num_labels <= 1:
            min_thresh = thresh
            continue

        areas = stats[1:, cv2.CC_STAT_AREA]
        max_idx = 1 + np.argmax(areas)
        x, y, w, h = (stats[max_idx, cv2.CC_STAT_LEFT],
                      stats[max_idx, cv2.CC_STAT_TOP],
                      stats[max_idx, cv2.CC_STAT_WIDTH],
                      stats[max_idx, cv2.CC_STAT_HEIGHT])

        fallback_rect = (x, y, w, h)

        # 仅用“几何筛选”的有效点评估
        subset_outPoint = points_in_rect(outPointsAll, (x, y, w, h))
        subset_cell_num = count_unique_cells_in_rotated_region(
            subset_outPoint, infos_40xtile, user_choose_area
        )

        if subset_cell_num >= target_cell_num:
            if subset_cell_num <= target_cell_num * (1 + select_ratio):
                index_result = 1
                subset_outPoints.append(subset_outPoint)
                subset_rects.append((x, y, w, h))
            else:
                max_thresh = thresh
        else:
            if subset_cell_num >= target_cell_num * (1 - select_ratio):
                index_result = 1
                subset_outPoints.append(subset_outPoint)
                subset_rects.append((x, y, w, h))
            else:
                min_thresh = thresh

    # 阈值搜索未命中，fallback
    if not subset_outPoints:
        if fallback_rect is None or not outPointsAll:
            return [], []
        subset_outPoint = points_in_rect(outPointsAll, fallback_rect)
        subset_outPoints.append(subset_outPoint)
        subset_rects.append(fallback_rect)

    # ---------- 补拍 ----------
    # 移除已选点
    taken = set(map(tuple, subset_outPoints[0]))
    outPointsAll = [pt for pt in outPointsAll if tuple(pt) not in taken]

    subset_rect = subset_rects[0]
    add_mask = last_th if last_th is not None else mask  # 兜底

    MAX_ITER, iter_count = 1000, 0
    while outPointsAll and iter_count < MAX_ITER:
        result = choose_area_points_addition(outPointsAll, add_mask, subset_rect)
        if result[0] is None:
            break

        AdditionRect, updated_rect = result
        x, y, w, h = AdditionRect

        subset_outPoint = points_in_rect(outPointsAll, (x, y, w, h))
        if not subset_outPoint:
            subset_rect = updated_rect
            iter_count += 1
            continue

        subset_outPoints.append(subset_outPoint)
        subset_rects.append((x, y, w, h))

        taken = set(map(tuple, subset_outPoint))
        outPointsAll = [pt for pt in outPointsAll if tuple(pt) not in taken]

        subset_rect = updated_rect
        iter_count += 1

    print("补拍次数：{}".format(iter_count))
    return subset_outPoints, subset_rects


def find_point_nearest_40x_index(x, y, abs_40xtile, fined_x40_list):
    """
    找到与点(x, y) 最近的 40x 图像索引。

    参数:
    - x, y: 百倍任务框中心点坐标
    - abs_40xtile: List[List[x, y]] 点列表
    - fined_x40: set[(x40_x, x40_y)] 40x 图像索引列表

    返回:
    - 40x 图像 X 索引, 40x 图像 Y 索引, 拼图后40x图像左上角点 X 坐标, 拼图后40x图像左上角 Y 坐标
    """
    tile_w, tile_h = 2448, 2048
    points = np.array(abs_40xtile, dtype=np.float32).reshape(-1, 2)
    center_points = points + np.array([tile_w / 2, tile_h / 2])

    target = np.array([x, y])
    distances = np.linalg.norm(center_points - target, axis=1)
    nearest_index = np.argmin(distances)
    fx, fy = fined_x40_list[nearest_index]
    abs_x, abs_y = abs_40xtile[nearest_index]
    return fx, fy, abs_x, abs_y


def clamp_roi_to_tile(x, y, w, h, abs_x, abs_y, tw=2448, th=2048):
    # 仅钳制左上角，使 视野框 尽量落在 tile 内；w/h 不改动
    vx = int(np.clip(x - abs_x, 0, max(0, tw - w)))
    vy = int(np.clip(y - abs_y, 0, max(0, th - h)))
    return vx, vy


def to_xy_array(points):
    """
    将各种点表示统一为形状 (N,2) 的 float 数组:
    - [[x,y], [x,y], ...]
    - 扁平 [x1,y1,x2,y2,...]  (长度为偶数)
    - [{'x':..., 'y':...}, {'cell_x':..., 'cell_y':...}, ...]
    为空则返回 shape=(0,2)。
    """
    if points is None:
        return np.empty((0, 2), dtype=float)

    arr = np.asarray(points, dtype=float)

    # 字典列表
    if arr.dtype == object and len(points) > 0 and isinstance(points[0], dict):
        xy = []
        for p in points:
            x = p.get('x', p.get('cx', p.get('cell_x', None)))
            y = p.get('y', p.get('cy', p.get('cell_y', None)))
            if x is None or y is None:
                continue
            xy.append([float(x), float(y)])
        return np.array(xy, dtype=float)

    # 标准/扁平数组
    if arr.ndim == 0:
        return np.empty((0, 2), dtype=float)

    if arr.ndim == 1:
        if arr.size == 0:
            return np.empty((0, 2), dtype=float)
        if arr.size == 2:
            return arr.reshape(1, 2)
        if arr.size % 2 == 0:
            return arr.reshape(-1, 2)
        # 其他奇怪情况
        return np.empty((0, 2), dtype=float)

    # (N, M) 取前2列
    if arr.shape[1] >= 2:
        return arr[:, :2].astype(float)

    return np.empty((0, 2), dtype=float)


def _points_in_union_rects(P, rects):
    """
    判断点是否落入若干矩形的并集（半开区间 [x, x+w) × [y, y+h)）。

    Parameters
    ----------
    P : array-like of shape (N, 2)
        点坐标 [[x, y], ...]。若已是 np.ndarray 且形状为 (N,2)，不会额外拷贝。
    rects : array-like of shape (M, 4)
        矩形列表/数组 [[x, y, w, h], ...]。半开区间。

    Returns
    -------
    keep : np.ndarray of bool, shape (N,)
        对于每个点，是否落入任一矩形。
    """
    # ---- Fast path for P ----
    if isinstance(P, np.ndarray) and P.ndim == 2 and P.shape[1] == 2:
        P_arr = P
        if P_arr.dtype != np.float32:
            P_arr = P_arr.astype(np.float32, copy=False)
    else:
        P_arr = np.asarray(P, dtype=np.float32).reshape(-1, 2)

    N = P_arr.shape[0]
    if N == 0 or not rects:
        return np.zeros(N, dtype=bool)

    # ---- Fast path for rects ----
    if isinstance(rects, np.ndarray):
        R = rects
        if R.ndim == 1:
            R = R.reshape(-1, 4)
        elif R.ndim == 2 and R.shape[1] >= 4:
            R = R[:, :4]
        else:
            R = np.asarray(rects, dtype=np.float32).reshape(-1, 4)
    else:
        R = np.asarray(rects, dtype=np.float32).reshape(-1, 4)

    if R.size == 0:
        return np.zeros(N, dtype=bool)
    if R.dtype != np.float32:
        R = R.astype(np.float32, copy=False)

    # ---- Broadcast test ----
    X = P_arr[:, 0:1]  # (N,1)
    Y = P_arr[:, 1:2]  # (N,1)
    rx = R[:, 0][None, :]  # (1,M)
    ry = R[:, 1][None, :]
    rw = R[:, 2][None, :]
    rh = R[:, 3][None, :]

    inside = (X >= rx) & (X < rx + rw) & (Y >= ry) & (Y < ry + rh)  # (N,M)
    return inside.any(axis=1)  # (N,)


def wbc_generate_task(
        best_result,
        infos_40xtile,
        user_choose_area,
        pad=100,
        params=None,
        tile_w: int = 2448,
        tile_h: int = 2048,
        halo_tiles: int = 1,  # 邻域补片
        halo_px: int = 0,  # 对“微矩形”做轻量像素外扩
):
    """
    只用【被选中的网格点】对应的微矩形作为 ROI（而非整块 tile），
    并保留 halo_tiles 对候选 tile 的外扩以修复跨 tile 边界。
    """
    if not best_result or 'outPointsAll' not in best_result:
        raise ValueError("best_result 不能为空，且必须包含 'outPointsAll' 键")
    if params is None:
        params = SetCoverSolverParameter()

    # 1) 建 tile 索引
    tile_map = {(int(t['index_40xtile_x']), int(t['index_40xtile_y'])): t
                for t in infos_40xtile}

    # 2) 由网格点求“被选中 tile 集合”（用于候选 tile 外扩）
    rotated_points = best_result['outPointsAll']
    pts = np.asarray(rotated_points, dtype=np.int32)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    min_col = int(user_choose_area['min_col'])
    min_row = int(user_choose_area['min_row'])
    max_row = int(user_choose_area['max_row'])
    row_span = max_row - min_row

    fx = (pts[:, 0] // 2) + min_col
    fy = (row_span - (pts[:, 1] // 2)) + min_row

    # 去重 tile
    sel_tiles = np.unique(np.stack([fx, fy], axis=1), axis=0)  # shape: (K, 2)
    if sel_tiles.size == 0:
        return [], []

    sel_tiles_set = set(map(tuple, sel_tiles))  # {(x,y), ...}

    # 预备偏移（去掉 (0,0)）
    offs = [(dx, dy)
            for dx in range(-halo_tiles, halo_tiles + 1)
            for dy in range(-halo_tiles, halo_tiles + 1)
            if (dx != 0 or dy != 0)]

    # 生成 halo 邻居，并与现有 tile_map 取交集（只保留真实存在的 tile）
    tile_keys = set(tile_map.keys())  # {(col,row), ...}

    neighbors = {
        (x + dx, y + dy)
        for (x, y) in sel_tiles_set
        for (dx, dy) in offs
    }

    cand_tiles = (sel_tiles_set | neighbors) & tile_keys  # 最终候选集合

    # 4) ROI = 【每个被选中网格点】对应的“微矩形(1×1)”像素区域，并可 halo_px 外扩
    roi_rects = []
    for pt in best_result['outPointsAll']:
        # 将一个网格点 pt=[gx,gy] 转为绝对像素矩形 (x,y,w,h)，等价于把 1x1 的网格矩形
        gx, gy = int(pt[0]), int(pt[1])
        x, y, w, h = grid_rect_to_abs_rect((gx, gy, 1, 1), infos_40xtile, user_choose_area)
        if halo_px and halo_px > 0:
            x -= halo_px;
            y -= halo_px;
            w += 2 * halo_px;
            h += 2 * halo_px
        roi_rects.append((x, y, w, h))

    # 5) 只收集落入 ROI 并集内的点（从候选 tile 里取，能补边界但不会放大到整块 tile）
    all_pts = []  # centers (N,2)
    all_sizes = []  # widths & heights (N,2)

    for fx, fy in cand_tiles:
        t = tile_map.get((fx, fy))
        if not t:
            continue
        rects = t.get("global_cell_rects_dedup", [])
        if not rects:
            continue

        a = np.asarray(rects, dtype=np.float32)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if a.shape[1] < 4:
            continue  # 容错

        # 只取 x1,y1,x2,y2；忽略 score
        # 中心点（若你希望与原逻辑整数对齐，可用 rint 再 astype）
        centers = 0.5 * (a[:, 0:2] + a[:, 2:4])

        # 尺寸：按半开区间使用 w = x2 - x1 + 1, h = y2 - y1 + 1
        sizes = (a[:, 2:4] - a[:, 0:2] + 1.0)
        sizes = np.clip(sizes, 1.0, None)  # 防御性：至少为 1

        keep = _points_in_union_rects(centers, roi_rects)
        if np.any(keep):
            all_pts.append(centers[keep])
            all_sizes.append(sizes[keep])

    if not all_pts:
        return [], [], []

    points = np.vstack(all_pts)  # (N,2) float
    sizes = np.vstack(all_sizes)  # (N,2) float 或 int

    # 6) set-cover 求 100× 视野框
    pad = int(pad)
    x_min = int(np.floor(points[:, 0].min())) - pad
    x_max = int(np.ceil(points[:, 0].max())) + pad
    y_min = int(np.floor(points[:, 1].min())) - pad
    y_max = int(np.ceil(points[:, 1].max())) + pad
    bounding_rect = np.array(
        [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1], dtype=np.int32
    )

    rects_x100 = solve(points, bounding_rect, params, rec_depth=0)

    # 7) 逐矩形匹配点（半开区间、互斥）
    used = np.zeros(len(points), dtype=bool)
    matched_points = []  # 每个矩形内的点中心 (Mi,2)
    matched_sizes = []  # 每个矩形内对应的 (w,h) (Mi,2)

    for rx, ry, rw, rh in rects_x100:
        in_x = (points[:, 0] >= rx) & (points[:, 0] < rx + rw)
        in_y = (points[:, 1] >= ry) & (points[:, 1] < ry + rh)
        mask = in_x & in_y & (~used)

        matched_points.append(points[mask].tolist())
        matched_sizes.append(sizes[mask].tolist())  # 同步保存 w,h
        used[mask] = True

    return rects_x100, matched_points, matched_sizes


def meg_generate_task_filtered_BMP(
        wbs_first_rect,
        infos_40xtile,
        user_choose_area,
        Adition_shooting_flag,
        view_type="meg",
        pad=100,
        save_flag=False,
        save_dir=None,
        params: SetCoverSolverParameter = SetCoverSolverParameter(),
        iou_thr: float = 0.90,
        iom_thr: float = 0.90,  # 保留签名兼容；最终去重采用 NMS 的 IoU
        bad_iom_thr: float = 0.50,  # 过滤坏区用：候选框与坏区的 IoM 阈值（交/小框面积）
        bad_center_hit: bool = True,  # 过滤坏区用：是否用中心点命中坏区
        eps: float = 1e-9,  # 数值稳定
        debug: bool = False
):
    """
    获取巨核细胞坐标，先将所有 tile 的 label==5 OFFSETS 块转换为“全局坏区”，
    再用全局坏区过滤候选框（中心命中或与坏区 IoM>=bad_iom_thr），然后做全局去重，
    最后生成巨核百倍拍摄任务。

    更新点：
    - 过滤阶段改为“全局坏区判定”，解决跨 tile 边界时 (which_block==None) 漏滤的问题。
    - 去重仍用 IoU / IoM；当撞车时，保留面积更大者。

    参数：
    - wbs_first_rect: Tuple[int, int, int, int] 初始有核选区（绝对像素）
    - infos_40xtile: List[Dict] 每张 40x 图像的细胞信息
    - Adition_shooting_flag: bool 是否为补拍区域
    - params: SetCoverSolverParameter 矩形生成参数
    - iou_thr, iom_thr: 去重阈值
    - bad_iom_thr, bad_center_hit: 与坏区过滤的策略
    - eps: 数值稳定
    - debug: 打印调试信息

    返回：巨核百倍拍摄任务列表（结构同原函数）
    """

    # ---------- 常量 ----------
    OFFSETS = {
        (0, 0): [0, 352, 896, 896],
        (1, 0): [848, 352, 896, 896],
        (0, 1): [0, 1152, 896, 896],
        (1, 1): [848, 1152, 896, 896],
    }

    # ---------- 小工具 ----------
    def _read_block_label(tile: dict, blk_key: str):
        scores = tile.get("scores", {})
        entry = scores.get(blk_key)
        if entry is None:
            return None
        if isinstance(entry, (list, tuple)):
            return entry[1] if len(entry) >= 2 else None
        if isinstance(entry, dict):
            for k in ("label", "cls", "class", "typeid"):
                if k in entry:
                    return entry[k]
        return None

    # ---------- 1) 收集所有候选 meg 矩形（全局坐标、元信息） ----------
    all_gx = []
    all_gy = []
    all_gw = []
    all_gh = []
    all_area = []
    all_cols = []
    all_rows = []
    all_absx = []
    all_absy = []

    # 外层 tile 遍历不可避免，但单 tile 内全部向量化
    for t in infos_40xtile:
        rects = t.get("meg_rect", [])
        if not rects:
            continue

        col_index = t.get("index_40xtile_x")
        row_index = t.get("index_40xtile_y")
        abs_x = t.get("abs_40xtile_x")
        abs_y = t.get("abs_40xtile_y")
        if None in (col_index, row_index, abs_x, abs_y):
            continue

        col_index = int(col_index)
        row_index = int(row_index)
        abs_x, abs_y = int(abs_x), int(abs_y)

        r = np.asarray(rects, dtype=np.int32)
        if r.ndim == 1:
            r = r.reshape(1, -1)
        if r.size == 0:
            continue
        r = r[:, :4]  # [x, y, w, h]（局部）

        gxy = r[:, :2] + np.array([abs_x, abs_y], dtype=np.int32)  # (N,2)
        gw_h = r[:, 2:4]  # (N,2)

        n = r.shape[0]
        all_gx.append(gxy[:, 0].astype(np.float32))
        all_gy.append(gxy[:, 1].astype(np.float32))
        all_gw.append(gw_h[:, 0].astype(np.float32))
        all_gh.append(gw_h[:, 1].astype(np.float32))
        all_area.append((gw_h[:, 0] * gw_h[:, 1]).astype(np.float32))
        all_cols.append(np.full((n,), col_index, dtype=np.int32))
        all_rows.append(np.full((n,), row_index, dtype=np.int32))
        all_absx.append(np.full((n,), abs_x, dtype=np.int32))
        all_absy.append(np.full((n,), abs_y, dtype=np.int32))

    if len(all_gx) == 0:
        return []

    gx = np.concatenate(all_gx)
    gy = np.concatenate(all_gy)
    gw = np.concatenate(all_gw)
    gh = np.concatenate(all_gh)
    g_area = np.concatenate(all_area)
    cols = np.concatenate(all_cols)
    rows = np.concatenate(all_rows)
    absxs = np.concatenate(all_absx)
    absys = np.concatenate(all_absy)

    # xyxy（供 NMS）
    x1 = gx
    y1 = gy
    x2 = gx + gw
    y2 = gy + gh
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    boxes_xywh = np.stack([gx, gy, gw, gh], axis=1).astype(np.float32)
    if debug:
        print(f"[debug] meg candidates : {boxes_xyxy.shape[0]}")

    # ---------- 2) 构建“全局坏区” ----------
    bad_rects_xywh = []
    for t in infos_40xtile:
        ax = t.get("abs_40xtile_x");
        ay = t.get("abs_40xtile_y")
        if ax is None or ay is None:
            continue
        ax, ay = int(ax), int(ay)
        for (dx, dy), (ox, oy, ow, oh) in OFFSETS.items():
            lab = _read_block_label(t, f"{dx}_{dy}")
            try:
                lab = int(lab) if lab is not None else None
            except Exception:
                lab = None
            if lab == 5:
                bad_rects_xywh.append([ax + ox, ay + oy, ow, oh])
    bad_rects_xywh = np.asarray(bad_rects_xywh, dtype=np.float32)

    # ---------- 3) “全局坏区”过滤（中心命中 + IoM>=阈值），整体广播 ----------
    keep_mask = np.ones(gx.shape[0], dtype=bool)
    if bad_rects_xywh.size > 0:
        bx, by, bw, bh = (bad_rects_xywh[:, 0], bad_rects_xywh[:, 1],
                          bad_rects_xywh[:, 2], bad_rects_xywh[:, 3])

        # 3.1 中心命中坏区
        if bad_center_hit:
            cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5
            cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) * 0.5
            center_hit_any = ((cx[:, None] >= bx[None, :]) & (cx[:, None] < (bx + bw)[None, :]) &
                              (cy[:, None] >= by[None, :]) & (cy[:, None] < (by + bh)[None, :])).any(axis=1)
        else:
            center_hit_any = np.zeros(gx.shape[0], dtype=bool)

        # 3.2 IoM（交 / 小框面积）
        bx1 = bx
        by1_ = by
        bx2 = bx1 + bw
        by2_ = by1_ + bh
        b_area = (bw * bh).astype(np.float32)

        iw = np.maximum(0.0, np.minimum(x2[:, None], bx2[None, :]) - np.maximum(x1[:, None], bx1[None, :]))
        ih = np.maximum(0.0, np.minimum(y2[:, None], by2_[None, :]) - np.maximum(y1[:, None], by1_[None, :]))
        inter = iw * ih
        denom = np.minimum(g_area[:, None], b_area[None, :]) + eps
        iom = inter / denom
        iom_hit_any = (iom >= float(bad_iom_thr)).any(axis=1)

        remove = center_hit_any | iom_hit_any
        keep_mask = ~remove

    if not keep_mask.any():
        if debug:
            print("[debug] all candidates filtered by global bad masks.")
        return []

    # 过滤后的数组
    boxes_xyxy_f = boxes_xyxy[keep_mask]
    boxes_xywh_f = boxes_xywh[keep_mask]
    gx_f, gy_f, gw_f, gh_f = gx[keep_mask], gy[keep_mask], gw[keep_mask], gh[keep_mask]
    area_f = g_area[keep_mask]
    cols_f, rows_f = cols[keep_mask], rows[keep_mask]
    absxs_f, absys_f = absxs[keep_mask], absys[keep_mask]

    if debug:
        print(f"[debug] meg candidates after bad-mask filter: {boxes_xyxy_f.shape[0]}")

    if boxes_xyxy_f.shape[0] == 0:
        return []

    # ---------- 4) NMS 前 exact-duplicate 去重（降规模，不改变结果） ----------
    if boxes_xyxy_f.shape[0] > 1:
        view = np.ascontiguousarray(boxes_xyxy_f).view([('', boxes_xyxy_f.dtype)] * 4)
        _, uniq_idx = np.unique(view, return_index=True, axis=0)
        if uniq_idx.size < boxes_xyxy_f.shape[0]:
            uniq_idx = np.sort(uniq_idx)
            boxes_xyxy_f = boxes_xyxy_f[uniq_idx]
            boxes_xywh_f = boxes_xywh_f[uniq_idx]
            area_f = area_f[uniq_idx]
            cols_f = cols_f[uniq_idx]
            rows_f = rows_f[uniq_idx]
            absxs_f = absxs_f[uniq_idx]
            absys_f = absys_f[uniq_idx]

    # ---------- 5) NMS（torchvision；score=面积） ----------
    boxes_t = torch.as_tensor(np.ascontiguousarray(boxes_xyxy_f), dtype=torch.float32)
    scores_t = torch.as_tensor(np.ascontiguousarray(area_f), dtype=torch.float32)
    keep_idx = torch_nms(boxes_t, scores_t, float(iou_thr))
    if keep_idx.numel() == 0:
        return []
    keep_idx_np = keep_idx.detach().cpu().numpy()

    # 保留并提取中心点/wh/元数据
    gx_k, gy_k, gw_k, gh_k = gx_f[keep_idx_np], gy_f[keep_idx_np], gw_f[keep_idx_np], gh_f[keep_idx_np]
    cols_k, rows_k = cols_f[keep_idx_np], rows_f[keep_idx_np]
    absxs_k, absys_k = absxs_f[keep_idx_np], absys_f[keep_idx_np]
    pts_k = np.stack([gx_k + gw_k * 0.5, gy_k + gh_k * 0.5], axis=1).astype(np.float32)
    wh_k = np.stack([gw_k, gh_k], axis=1).astype(np.int32)

    # ---------- 6) used tiles（唯一 (col,row) 及其 abs 原点） ----------
    cr = np.stack([cols_k, rows_k], axis=1)
    uniq_cr, uniq_idx_cr = np.unique(cr, axis=0, return_index=True)
    fined_x40_list = uniq_cr.astype(np.int32).tolist()
    abs_40xtile = np.stack([absxs_k[uniq_idx_cr], absys_k[uniq_idx_cr]], axis=1).astype(np.int32).tolist()

    # ---------- 7) 计算最小外接矩形 + padding ----------
    x_min = int(math.floor(pts_k[:, 0].min())) - pad
    x_max = int(math.ceil(pts_k[:, 0].max())) + pad
    y_min = int(math.floor(pts_k[:, 1].min())) - pad
    y_max = int(math.ceil(pts_k[:, 1].max())) + pad
    bounding_rect = np.array([x_min, y_min, x_max - x_min + 1, y_max - y_min + 1], dtype=np.int32)

    # ---------- 8) set-cover 求百倍任务矩形 ----------
    rects_x100 = solve(pts_k, bounding_rect, params, rec_depth=0)
    if rects_x100 is None:
        return []
    rects_x100 = np.asarray(rects_x100, dtype=np.int32).reshape(-1, 4)
    if rects_x100.size == 0:
        return []

    # ---------- 9) 按“离首个有核框中心近”排序 ----------
    wx, wy, ww, hh = map(float, wbs_first_rect)
    cx0, cy0 = wx + ww * 0.5, wy + hh * 0.5
    rcx = rects_x100[:, 0] + rects_x100[:, 2] * 0.5
    rcy = rects_x100[:, 1] + rects_x100[:, 3] * 0.5
    order = np.argsort((rcx - cx0) ** 2 + (rcy - cy0) ** 2)
    rects_sorted = rects_x100[order]

    if save_flag:
        visualize_meg_view(x_min, y_min, x_max, y_max, wbs_first_rect, pts_k, rects_sorted, save_dir=save_dir)

    # ---------- 10) 将点分配到“第一个覆盖它的任务矩形”（R×P 广播） ----------
    R = rects_sorted.shape[0]
    P = pts_k.shape[0]
    if P == 0:
        return []

    rx = rects_sorted[:, 0].astype(np.float32)[:, None]  # (R,1)
    ry = rects_sorted[:, 1].astype(np.float32)[:, None]
    rw = rects_sorted[:, 2].astype(np.float32)[:, None]
    rh = rects_sorted[:, 3].astype(np.float32)[:, None]
    px = pts_k[:, 0].astype(np.float32)[None, :]  # (1,P)
    py = pts_k[:, 1].astype(np.float32)[None, :]

    cover = (px >= rx) & (px < rx + rw) & (py >= ry) & (py < ry + rh)  # (R,P)
    # “第一个覆盖它的矩形”：把 True 映射为 (row_index+1)，否则 0；取列最小非零-1
    idx_mat = np.where(cover, (np.arange(R, dtype=np.int32)[:, None] + 1), 0)
    idx_mat_nozero = np.where(idx_mat == 0, R + 1, idx_mat)
    assigned_idx = idx_mat_nozero.min(axis=0) - 1  # [-1..R-1]
    assigned_idx = np.where(assigned_idx >= R, -1, assigned_idx)  # 未覆盖 -> -1

    valid_mask = assigned_idx >= 0
    if not valid_mask.any():
        return []

    pts_valid = pts_k[valid_mask]
    wh_valid = wh_k[valid_mask]
    rect_id_v = assigned_idx[valid_mask]

    sort_o = np.argsort(rect_id_v)
    pts_sorted_by_rect = pts_valid[sort_o]
    wh_sorted_by_rect = wh_valid[sort_o]
    rect_id_sorted = rect_id_v[sort_o]

    counts = np.bincount(rect_id_sorted, minlength=R)
    offsets = np.concatenate([[0], np.cumsum(counts[:-1])])

    # ---------- 11) 构建任务列表（I/O 构造，小循环） ----------
    result = []
    row_min = int(user_choose_area['min_row'])
    row_max = int(user_choose_area['max_row'])
    row_span = (row_max - row_min)

    for i in range(R):
        cnt = int(counts[i])
        x, y, w, h = map(int, rects_sorted[i].tolist())

        if cnt > 0:
            s = int(offsets[i])
            e = s + cnt
            pts_i = pts_sorted_by_rect[s:e]  # (cnt,2)
            wh_i = wh_sorted_by_rect[s:e]  # (cnt,2)
        else:
            pts_i = np.empty((0, 2), dtype=np.float32)
            wh_i = np.empty((0, 2), dtype=np.int32)

        # 任务中心用于选最近 tile
        cx = x + w // 2
        cy = y + h // 2
        col_index, row_index, abs_40xtile_x, abs_40xtile_y = find_point_nearest_40x_index(
            cx, cy, abs_40xtile, fined_x40_list
        )

        # 视野位置裁剪到 tile 内
        view_pos_x, view_pos_y = clamp_roi_to_tile(x, y, w, h, abs_40xtile_x, abs_40xtile_y)

        # 将全局点转为 tile 局部 cell 框（向量化计算后再打包为 dict 列表）
        if pts_i.shape[0] > 0:
            gx_cell = pts_i[:, 0]
            gy_cell = pts_i[:, 1]
            cw = wh_i[:, 0].astype(np.int32)
            ch = wh_i[:, 1].astype(np.int32)
            cell_x = (gx_cell - abs_40xtile_x - cw * 0.5).astype(np.int32)
            cell_y = (gy_cell - abs_40xtile_y - ch * 0.5).astype(np.int32)

            cell_list = [
                {"cell_x": int(cell_x[k]), "cell_y": int(cell_y[k]), "cell_width": int(cw[k]),
                 "cell_height": int(ch[k])}
                for k in range(pts_i.shape[0])
            ]
        else:
            cell_list = []

        result.append({
            "sort_id": int(i + 1),
            "Adition_shooting_flag": str(Adition_shooting_flag),
            "row_index": row_span - int(row_index) + row_min,  # 第四象限映射
            "col_index": int(col_index),
            "view_pos_x": int(view_pos_x),
            "view_pos_y": int(view_pos_y),
            "view_width": int(w),
            "view_height": int(h),
            "view_type": str(view_type),
            "cell_list": cell_list,
        })

    return result


def heatmap_rect_to_abs_rect(wbs_first_rect, user_choose_area, infos_40xtile):
    """
    将 热力图坐标系 的 wbs_first_rect 转换为 绝对像素坐标系。
    参数：
    - wbs_first_rect: Tuple[int, int, int, int] 有核初始选区矩形-热力图坐标系 每个矩形框为 (x, y, w, h) 格式
    """

    # === 常量：每个 40x tile 的子块设置（2×2 网格，窗口大小与步长）===
    PATCH_W, PATCH_H = 896, 896
    STRIDE_X = 848
    ROW0_Y, ROW1_Y = 352, 1152  # 两行子块在 tile 内的 y 起点
    # offset = {(0,0):[0, 352], (1,0):[848, 352], (0,1):[0, 1152], (1,1):[848, 1152]}
    offset = {
        (0, 0): (0, ROW0_Y),  # 左上
        (1, 0): (STRIDE_X, ROW0_Y),  # 右上
        (0, 1): (0, ROW1_Y),  # 左下
        (1, 1): (STRIDE_X, ROW1_Y),  # 右下
    }

    # 1) 计算所在的 tile 索引（列不翻转，行需要翻转）
    wbs_first_x, wbs_first_y = int(wbs_first_rect[0]), int(wbs_first_rect[1])

    wbs_first_indexID_x = wbs_first_x // 2 + user_choose_area['min_col']
    wbs_first_indexID_y = (user_choose_area['max_row'] - user_choose_area['min_row']) - (wbs_first_y // 2) + \
                          user_choose_area['min_row']

    off_x = wbs_first_x % 2
    off_y = wbs_first_y % 2
    if off_x not in (0, 1) or off_y not in (0, 1):
        raise ValueError(f"sub-block index out of range: ({off_x}, {off_y})")

    # 2) 找到对应 tile 的绝对左上角
    cell_info = None
    for info in infos_40xtile:
        if (info.get("index_40xtile_x") == wbs_first_indexID_x and
                info.get("index_40xtile_y") == wbs_first_indexID_y):
            cell_info = info
            break

    if cell_info is None:
        raise ValueError("未找到第一个有核任务对应的 x40 图块信息，请检查逻辑。")

    tile_abs_x = int(cell_info.get("abs_40xtile_x", 0))
    tile_abs_y = int(cell_info.get("abs_40xtile_y", 0))

    # 3) tile 内部的子块偏移（像素）
    off_px_x, off_px_y = offset[(off_x, off_y)]

    # 4) 计算绝对位置与尺寸
    wbs_first_abs_x = tile_abs_x + off_px_x
    wbs_first_abs_y = tile_abs_y + off_px_y

    # 注意：只有当 wbs_first_rect 的 w/h 是“子块个数”时才乘 PATCH_W/H
    w_cells = int(wbs_first_rect[2])
    h_cells = int(wbs_first_rect[3])
    wbs_first_abs_w = w_cells * PATCH_W
    wbs_first_abs_h = h_cells * PATCH_H

    wbs_first_abs_rect = (wbs_first_abs_x, wbs_first_abs_y, wbs_first_abs_w, wbs_first_abs_h)

    return wbs_first_abs_rect


def grid_rect_to_abs_rect(grid_rect, infos_40xtile, user_choose_area):
    PATCH_W, PATCH_H = 1224, 1024
    # offset = {(0,0):[0, 0], (1,0):[1224, 0], (0,1):[0, 1024], (1,1):[1224, 1024]}
    offset = {
        (0, 0): (0, 0),  # 左上
        (1, 0): (PATCH_W, 0),  # 右上
        (0, 1): (0, PATCH_H),  # 左下
        (1, 1): (PATCH_W, PATCH_H),  # 右下
    }

    # 1) 计算所在的 tile 索引（列不翻转，行需要翻转）
    _x, _y = int(grid_rect[0]), int(grid_rect[1])

    _indexID_x = _x // 2 + user_choose_area['min_col']
    _indexID_y = (user_choose_area['max_row'] - user_choose_area['min_row']) - (_y // 2) + user_choose_area['min_row']

    off_x = _x % 2
    off_y = _y % 2
    if off_x not in (0, 1) or off_y not in (0, 1):
        raise ValueError(f"sub-block index out of range: ({off_x}, {off_y})")

    # 2) 找到对应 tile 的绝对左上角
    cell_info = None
    for info in infos_40xtile:
        if (info.get("index_40xtile_x") == _indexID_x and
                info.get("index_40xtile_y") == _indexID_y):
            cell_info = info
            break

    if cell_info is None:
        raise ValueError("未找到第一个有核任务对应的 x40 图块信息，请检查逻辑。")

    tile_abs_x = int(cell_info.get("abs_40xtile_x", 0))
    tile_abs_y = int(cell_info.get("abs_40xtile_y", 0))

    # 3) tile 内部的子块偏移（像素）
    off_px_x, off_px_y = offset[(off_x, off_y)]

    # 4) 计算绝对位置与尺寸
    abs_x = tile_abs_x + off_px_x
    abs_y = tile_abs_y + off_px_y

    # 注意：只有当 _rect 的 w/h 是“子块个数”时才乘 PATCH_W/H
    w_cells = int(grid_rect[2])
    h_cells = int(grid_rect[3])
    abs_w = w_cells * PATCH_W
    abs_h = h_cells * PATCH_H

    abs_rect = (abs_x, abs_y, abs_w, abs_h)

    return abs_rect


def _find_tile_for_point(px, py, infos_40xtile, TILE_W, TILE_H):
    """根据绝对坐标点找到其所在 40× tile；若无则返回 None。"""
    for t in infos_40xtile:
        tx = int(t.get('abs_40xtile_x', 0))
        ty = int(t.get('abs_40xtile_y', 0))
        if (px >= tx) and (px < tx + TILE_W) and (py >= ty) and (py < ty + TILE_H):
            return t
    return None


def _rect_intersection_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih


def _point_in_rect_abs(px: float, py: float, r) -> bool:
    x, y, w, h = r
    return (px >= x) and (px < x + w) and (py >= y) and (py < y + h)


# ------- 把“网格点/网格矩形”转换为绝对像素 ROI -------
def _points_group_to_abs_rects(points_group: List[List[int]],
                               infos_40xtile,
                               user_choose_area,
                               halo_px: int = 0) -> List[Tuple[int, int, int, int]]:
    """
    points_group: [[gx,gy], ...] （一个小区域的网格点集）
    返回该小区域的 ROI：由多个微矩形(1x1) 组成的像素矩形列表
    """
    rects = []
    for gx, gy in points_group:
        x, y, w, h = grid_rect_to_abs_rect((int(gx), int(gy), 1, 1),
                                           infos_40xtile, user_choose_area)
        if halo_px and halo_px > 0:
            x -= halo_px;
            y -= halo_px;
            w += 2 * halo_px;
            h += 2 * halo_px
        rects.append((x, y, w, h))
    return rects


# ------- 计算与“并集 ROI”的关系（中心点与总相交面积） -------
def _center_in_union(cx: float, cy: float, roi_rects_abs: List[Tuple[int, int, int, int]]) -> bool:
    for r in roi_rects_abs:
        if _point_in_rect_abs(cx, cy, r):
            return True
    return False


def _total_intersection_area(r_xywh, roi_rects_abs: List[Tuple[int, int, int, int]]) -> float:
    s = 0.0
    for rr in roi_rects_abs:
        s += _rect_intersection_xywh(r_xywh, rr)
    return s


def match_x100_to_task_regions(
        rects_x100: List[Tuple[float, float, float, float]],
        user_choose_area: Dict,
        infos_40xtile: List[Dict],
        task_points_list: Optional[List[List[List[int]]]] = None,  # [[[gx,gy],...],  ...]
        halo_px: int = 0
):
    """
    返回:
      assignment: List[List[int]]，第 i 个“小区域”（以 points_group 为单位）包含的 x100 索引
      unassigned: List[int]，未分配的 x100 索引
    说明：
      - 若传入 task_points_list：每一组 points 是一个小区域（与你生成任务时的分组一致）
      - 若未提供 points，则退回用 task_rects_list（兼容旧逻辑）
    """

    if task_points_list is None:
        raise ValueError("提供 task_points_list")

    # 1) 构建每个“小区域”的 ROI（像素坐标、由多个矩形构成）
    rois_by_region: List[List[Tuple[int, int, int, int]]] = []
    # 用“真实选择的网格点”定义小区域（更精确）
    for points_group in task_points_list:
        rois_by_region.append(
            _points_group_to_abs_rects(points_group, infos_40xtile, user_choose_area, halo_px=halo_px)
        )

    M = len(rois_by_region)
    assignment = [[] for _ in range(M)]
    unassigned: List[int] = []

    # 2) 逐个 x100 进行分配
    for i, r in enumerate(rects_x100):
        cx = r[0] + 0.5 * r[2]
        cy = r[1] + 0.5 * r[3]

        # 2.1 先看中心点落在哪个“小区域”的 ROI 并集里
        chosen = -1
        for gi in range(M):
            if _center_in_union(cx, cy, rois_by_region[gi]):
                chosen = gi
                break

        # 2.2 兜底：按与 ROI 并集的“总相交面积”最大分配（>0 即可）
        if chosen < 0:
            best_area, best_idx = 0.0, -1
            for gi in range(M):
                inter_sum = _total_intersection_area(r, rois_by_region[gi])
                if inter_sum > best_area:
                    best_area, best_idx = inter_sum, gi
            if best_area > 0:
                chosen = best_idx

        if chosen >= 0:
            assignment[chosen].append(i)
        else:
            unassigned.append(i)

    return assignment, unassigned


def build_grouped_results_from_assignment(
        assignment,
        rects_x100,
        matched_points,
        matched_sizes,
        infos_40xtile,
        user_choose_area,
        view_type="wbc",
        TILE_W=2448,
        TILE_H=2048
):
    """
    把分配结果转成最终输出格式：
    输出: [result1, result2, ...]，其中 resulti 是该小区域的任务列表（list of dict）
    规则：
      - result[0] 内的任务项 "Adition_shooting_flag"="false"，其余为 "true"
      - 每个区域内部 sort_id 从 1 开始
      - sort_id 排序规则：按 x 再 y 升序
    """
    grouped_results = []

    assert len(matched_points) == len(matched_sizes) == len(
        rects_x100), "警告：matched_points/matched_sizes 与 rects_x100 数量不一致。"

    for gi, rect_indices in enumerate(assignment):
        Adition_flag_str = "False" if gi == 0 else "True"
        result = []

        # 按 x 再 y 对该组的 rect 进行排序（升序）
        rect_indices_sorted = sorted(
            rect_indices,
            key=lambda i: (rects_x100[i][0], rects_x100[i][1])  # (rx, ry)
        )

        for sort_id, idx in enumerate(rect_indices_sorted, start=1):
            rx, ry, rw, rh = rects_x100[idx]
            pts = matched_points[idx]
            sizes = matched_sizes[idx]

            # 用 x100 框中心点定位 tile
            cx, cy = rx + 0.5 * rw, ry + 0.5 * rh
            tile = _find_tile_for_point(cx, cy, infos_40xtile, TILE_W, TILE_H)
            if tile is None:
                continue

            abs_40xtile_x = int(tile.get('abs_40xtile_x', 0))
            abs_40xtile_y = int(tile.get('abs_40xtile_y', 0))
            col_index = int(tile.get('index_40xtile_x', 0))
            tile_y_idx = int(tile.get('index_40xtile_y', 0))
            row_index = user_choose_area['max_row'] - user_choose_area['min_row'] - int(tile_y_idx) + user_choose_area[
                'min_row']

            # 视野框在 tile 内的局部坐标
            view_pos_x = int(rx - abs_40xtile_x)
            view_pos_y = int(ry - abs_40xtile_y)
            w, h = int(rw), int(rh)

            # 收集该矩形中的细胞（限定在该 tile 且在 x100 框内，半开区间）
            cell_list = []
            if pts:
                for (cell_x, cell_y), (sw, sh) in zip(pts, sizes):
                    cell_x_local = int(cell_x) - abs_40xtile_x
                    cell_y_local = int(cell_y) - abs_40xtile_y
                    cell_w = int(round(sw))
                    cell_h = int(round(sh))

                    cell_list.append({
                        "cell_x": cell_x_local - cell_w // 2,
                        "cell_y": cell_y_local - cell_h // 2,
                        "cell_width": cell_w,
                        "cell_height": cell_h
                    })

            result.append({
                "sort_id": sort_id,  # 从 1 开始
                "Adition_shooting_flag": Adition_flag_str,  # "true" 或 "false"
                "row_index": int(row_index),  # 按第四象限换算后的行索引
                "col_index": int(col_index),  # 40× 列索引
                "view_pos_x": int(view_pos_x),  # 在 40× 内的像素 X
                "view_pos_y": int(view_pos_y),  # 在 40× 内的像素 Y
                "view_width": int(w),  # 视野宽
                "view_height": int(h),  # 视野高
                "view_type": str(view_type),  # "wbc"
                "cell_list": cell_list,  # 任务内细胞列表
            })

        grouped_results.append(result)

    return grouped_results


def select_and_generate_bestArea_capture_tasks(
        infos_40xtile,
        user_choose_area,
        heatmap_orientation,
        target_cell_num,
        save_flag=False,
        save_dir: str | None = None,
):
    """
    从中选择最优区域。

    参数：
    - heatmap_orientation: int
        图像方向标志，0 表示右头，1 表示左头，其它值表示未知方向
    - head_crop_rect: Dict[str, int]
        头部区域裁剪矩形，格式为 {'x', 'y', 'width', 'height'}
    - infos_40xtile: List[Dict]
        每张 40x 图像的细胞信息列表，包含 tile 坐标与细胞中心点坐标等
    - user_choose_area: Dict
        用户框选区域范围，字段包括 'min_row', 'max_row', 'min_col', 'max_col'
    - target_cell_num: int
        用户期望的细胞数量
    - save_flag: bool 是否保存中间图像
    - save_dir: str 保存图像前缀路径

    返回：
    - task_list: List[List[Dict]] 百倍任务列表（包含巨核百倍任务）格式如下：
    [
        [
            {
                "sort_id": 1,                // 任务点的顺序编号
                "row_index": 0,              // 任务点所在X40图布图的行索引
                "col_index": 0,              // 任务点所在X40图布图的列索引
                "view_pos_x": 0,             // 任务点在X40图里的像素X坐标
                "view_pos_y": 0,             // 任务点在X40图里的像素Y坐标
                "view_width": 20,            // 任务点在X40图里的图像宽度（像素）
                "view_height": 20,           // 任务点在X40图里的图像高度（像素）
                "cell_list":                 // 任务点内的细胞列表
                [               
                    {
                    "cell_x": 0,             // 细胞在X40图里的像素X坐标
                    "cell_y": 0,             // 细胞在X40图里的像素Y坐标
                    "cell_width": 20,        // 细胞在X40图里的图像宽度（像素）
                    "cell_height": 20        // 细胞在X40图里的图像高度（像素）
                    }
                ]
            }
            .
            .
            .
        ]
        [补拍1任务列表]
        [补拍2任务列表]
        .
        .
        .
        [补拍n任务列表]
    ]
    """

    if target_cell_num <= 0:
        raise ValueError("目标细胞数量必须大于 0")

    # 1) 创建分值热力图
    Scores_Matrix = scores_matrix(infos_40xtile, user_choose_area)

    # 2) 计算图像头部区域的裁剪宽度和裁剪矩形区域
    head_crop_x, head_crop_rect = compute_head_crop(Scores_Matrix, heatmap_orientation=heatmap_orientation)
    print("头部区域裁剪 x 坐标： {}".format(head_crop_x))
    print("头部区域裁剪 Rect： {}".format(head_crop_rect))

    # 3) 计算旋转框尺寸
    rects = generate_unique_rects(
        [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 40, 60, 80, 100, 120, 160, 200, 300,
         400])
    print("生成的旋转框尺寸： {}".format(len(rects)))

    # 4) 计算每个旋转框的得分和细胞数量（目标= target_cell_num*3）
    all_cell_count = sum(len(t.get('global_cell_rects_dedup', [])) for t in infos_40xtile)
    print("项目全部有核细胞数量（全局一次性去重后）：", all_cell_count)
    results = find_optimal_rotated_regions(
        Scores_Matrix, rects, heatmap_orientation, head_crop_rect,
        infos_40xtile, user_choose_area, target_cell_num * 3, all_cell_count
    )
    print("头部选区结果集合有： {}".format(len(results.get('head_results', []))))
    print("尾部选区结果集合有： {}".format(len(results.get('tail_results', []))))

    # 5) 筛选出最接近目标细胞数量的选区
    filter_cell_results = filter_closest_to_target_count(
        results, user_choose_area, target_cell_num * 3, all_cell_count, Scores_Matrix
    )
    print("筛选后符合细胞数量的选区结果有： {}".format(len(filter_cell_results)))

    # 6) 计算每个选区的细胞分布均匀性得分，取最佳
    best_result = select_best_uniform_region_by_area_score(
        filter_cell_results, infos_40xtile, user_choose_area
    )
    print("最佳选区结果： {}".format(best_result))
    if save_flag:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        best_result_vis_path = os.path.join(save_dir, "best_result.png")
        visualize_selection(best_result, Scores_Matrix, best_result_vis_path)

    # 7) 根据 best_result 生成过滤骨髓小粒的初始/补拍的任务点和矩形（热力图坐标）
    task_points_list, task_rects_list = generate_capture_tasks_from_region_filtered_BMP(
        best_result, infos_40xtile, target_cell_num, Scores_Matrix, user_choose_area
    )
    if len(task_points_list) != len(task_rects_list):
        raise ValueError("选区点结果列表与 rect 结果列表数量不符，请检查生成任务的逻辑。")
    print("任务点列表： {}".format(task_points_list))
    print("任务矩形列表： {}".format(task_rects_list))

    if save_flag:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        img_points_vis_path = os.path.join(save_dir, "task_points.png")
        visualize_capture_tasks(Scores_Matrix, task_points_list, task_rects_list, img_points_vis_path)

    # 8) 一次性生成所有有核拍摄任务（100×）
    # 虽然该步骤根据best_result[outPointsAll]生成的x100框，但在步骤9 x100框分配时是按task_points_list进行匹配的，task_points_list过滤了骨髓小粒
    rects_x100, matched_points, matched_sizes = wbc_generate_task(best_result, infos_40xtile, user_choose_area)
    print("生成的百倍任务矩形列表： {}".format(len(rects_x100)))

    if len(rects_x100) != len(matched_points):
        raise ValueError("生成的百倍任务列表长度与选区结果数量不符，请检查生成任务的逻辑。")

    if save_flag:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        rects_points_vis_path = os.path.join(save_dir, "wbc_rects_points.png")
        visualize_wbc_rects_points(
            rects_x100=rects_x100,
            matched_points=matched_points,
            save_path=rects_points_vis_path,
            factor=10.0,
            max_canvas_side=4096,
            pad=80
        )

    # 9) 将全局的 x100 框分配到各个 task 小区域，并构建最终百倍任务列表
    assignment, unassigned = match_x100_to_task_regions(
        rects_x100,
        user_choose_area,
        infos_40xtile,
        task_points_list=task_points_list,
    )
    x100_results = build_grouped_results_from_assignment(
        assignment=assignment,
        rects_x100=rects_x100,
        matched_points=matched_points,
        matched_sizes=matched_sizes,
        infos_40xtile=infos_40xtile,
        user_choose_area=user_choose_area,
        view_type="wbc"
    )

    if len(unassigned) > 0:
        print(
            f"提示：有 {len(unassigned)} 个百倍任务框未分配到任何小区域，边缘或骨髓小粒周边百倍任务框未分配属于正常情况。")
        print("未分配的 x100 任务框索引：", unassigned)
        print("未分配的 x100 任务框坐标：", [rects_x100[i] for i in unassigned])

    if len(x100_results) != len(task_points_list):
        raise ValueError("生成的百倍任务列表长度与选区结果数量不符，请检查生成任务的逻辑。")

    # 10) 生成巨核拍摄任务
    wbs_first_rect = task_rects_list[0]  # 热力图坐标 (x, y, w, h)
    wbs_first_abs_rect = heatmap_rect_to_abs_rect(wbs_first_rect, user_choose_area, infos_40xtile)  # 绝对像素坐标
    meg_task = meg_generate_task_filtered_BMP(wbs_first_abs_rect, infos_40xtile, user_choose_area,
                                              Adition_shooting_flag='False',
                                              save_flag=save_flag, save_dir=save_dir,
                                              view_type="meg", debug=True, bad_iom_thr=0.2)

    x100_results.append(meg_task)  # 巨核任务放在最后

    return x100_results


if __name__ == "__main__":

    import time

    json_dir = "/home/ubuntu/VScodeProjects/megLoc_heatmap/data_20250929/"
    project_dir = "/home/ubuntu/VScodeProjects/megLoc_heatmap/image/"
    # project_list = ['847304e1494f4a9da1747b341bfed4a7']
    # project_list = ['data2025063008.json']
    project_list = os.listdir(json_dir)

    for project_name in project_list:
        json_path = os.path.join(json_dir, project_name)
        save_dir = '/home/ubuntu/VScodeProjects/megLoc_heatmap/new_output_8/' + Path(json_path).stem
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
        save_result_json_path = os.path.join(save_dir, "x100_task_list.json")
        # if os.path.exists(save_result_json_path):continue

        infos_40xtile, col_num, row_num = parse_json_file(json_path)
        s_t = time.time()
        infos_40xtile = dedup_cells_across_tiles(infos_40xtile, iou_thresh=0.2)  # 有核细胞去重
        e_t = time.time()
        print(f"有核细胞去重成功，耗时：{e_t - s_t:.2f} 秒")

        user_choose_area = {
            "min_row": 0,  # 有核采集区域行号（最小）
            "min_col": 0,  # 有核采集区域列号（最小）
            "max_row": row_num - 1,  # 有核采集区域行号（最大）
            "max_col": col_num - 1}  # 有核采集区域列号（最大）
        print(user_choose_area)
        heatmap_orientation = 1
        target_cell_num = 500
        start_time = time.time()
        task_list = select_and_generate_bestArea_capture_tasks(infos_40xtile, user_choose_area, heatmap_orientation,
                                                               target_cell_num,
                                                               save_flag=True, save_dir=save_dir)
        end_time = time.time()
        print(f"百倍任务列表生成成功，耗时：{end_time - start_time:.2f} 秒，共 {len(task_list)} 个任务块")

        with open(save_result_json_path, 'w') as f:
            json.dump(task_list, f, indent=4)
        print(f"百倍任务列表保存成功：{Path(json_path).stem}")
