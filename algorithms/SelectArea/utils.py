import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import math
import os
import json
from pathlib import Path


def parse_cell_file(file_path):
    tile_info = {}

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue  # 忽略格式错误行

            # 解析字段
            index_x = int(parts[0])
            index_y = int(parts[1])
            abs_x = int(parts[2]) - 1224
            abs_y = int(parts[3]) - 1024
            cell_num = int(parts[4])  # 虽然有这个字段，但每行实际只有一个细胞

            local_x = float(parts[5])
            local_y = float(parts[6])
            global_x = float(parts[7])
            global_y = float(parts[8])

            key = (index_x, index_y)

            # 初始化 tile_info 中对应图像的结构
            if key not in tile_info:
                tile_info[key] = {
                    "index_40xtile_x": index_x,
                    "index_40xtile_y": index_y,
                    "abs_40xtile_x": abs_x,
                    "abs_40xtile_y": abs_y,
                    "local_cell_centers": [],
                    "global_cell_centers": []
                }

            # 追加当前细胞中心点坐标
            tile_info[key]["local_cell_centers"].append([local_x, local_y])
            tile_info[key]["global_cell_centers"].append([global_x, global_y])

    return tile_info



def read_json(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)  # dict 或 list




def parse_block_file(file_path):
    block_scores = {}

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue  # 跳过格式错误行

            index_x = int(parts[0])
            index_y = int(parts[1])
            pos_x = int(parts[2])  # 左上角坐标 x
            pos_y = int(parts[3])  # 左上角坐标 y
            # parts[4] 是标志位，可能是用于标记有效性
            score = float(parts[5])

            # 判断子块属于哪一个四宫格位置
            dx = 0 if pos_x == 0 else 1
            dy = 0 if pos_y == 352 else 1  # 从示例中可见判断阈值
            pos_key = f"{dx}_{dy}"

            key = (index_x, index_y)

            if key not in block_scores:
                block_scores[key] = {}
            block_scores[key][pos_key] = score

    return block_scores




def gen_meg_rect_np(n, tile_w=2448, tile_h=2048, seed=None, margin=0):
    rng = np.random.default_rng(seed)
    xs = rng.integers(margin, tile_w  - margin, size=n)
    ys = rng.integers(margin, tile_h - margin, size=n)
    ws = rng.integers(100, 400, size=n)
    hs = rng.integers(100, 400, size=n)
    return np.stack([xs, ys, ws, hs], axis=1).tolist()




def build_infos(cells_path, block_path):
    cell_data = parse_cell_file(cells_path)
    block_data = parse_block_file(block_path)

    infos_40xtile = []
    for key, scores in block_data.items():
        # 获取细胞数据（可能为空）
        data = cell_data.get(key, {
            "index_40xtile_x": key[0],
            "index_40xtile_y": key[1],
            "abs_40xtile_x": 0,
            "abs_40xtile_y": 0,
            "local_cell_centers": [],
            "global_cell_centers": []
        })

        # 添加 scores 信息
        data["scores"] = scores

        if 68 <= key[0] <= 78 and 68 <= key[1] <= 78:
            if random.random() < 0.3:
                meg_rect = gen_meg_rect_np(2, margin=10)
            else:
                meg_rect = []    
        else:
            meg_rect = []

        data["meg_rect"] = meg_rect

        infos_40xtile.append(data)

    return infos_40xtile




def parse_json_file(file_path):
    infos_40xtile = []
    data = read_json(file_path)
    
    nodes = data['nodes']
    for info in nodes:
        index_40xtile_x = int(info['col'])
        index_40xtile_y = int(info['row'])

        block_scores = {}
        area_scores = info['data']['area_score_info']
        assert len(area_scores) == 4, f"area_score_info error: 期望 4 个，实际 {len(area_scores)}，数据={area_scores}"
        for area_score in area_scores:
            pos_x = int(area_score[0])  # 左上角坐标 x
            pos_y = int(area_score[1])  # 左上角坐标 y
            # 判断子块属于哪一个四宫格位置
            dx = 0 if pos_x == 0 else 1
            dy = 0 if pos_y == 88 else 1  # 从示例中可见判断阈值
            pos_key = f"{dx}_{dy}"
            score = [float(area_score[4]), int(area_score[5])]
            block_scores[pos_key] = score

        position = info['data']['position']  
        abs_40xtile_x = position[0]
        abs_40xtile_y = position[1]

        local_cell_rects = info['data']['wbc_center_pt']
        global_cell_rects = [[wbc_rect[0] + abs_40xtile_x, wbc_rect[1] + abs_40xtile_y,
                                wbc_rect[2] + abs_40xtile_x, wbc_rect[3] + abs_40xtile_y,
                                wbc_rect[4]] for wbc_rect in local_cell_rects]
        meg_rect = info['data']['meg_center_pt']
        infos_40xtile.append({
            "index_40xtile_x": index_40xtile_x,
            "index_40xtile_y": data['task_info']['row_num'] - 1 - index_40xtile_y,
            "abs_40xtile_x": abs_40xtile_x,
            "abs_40xtile_y": abs_40xtile_y,
            "local_cell_rects": local_cell_rects,
            "global_cell_rects": global_cell_rects,
            "meg_rect": meg_rect,
            "scores": block_scores
        })

    return infos_40xtile, data['task_info']['col_num'], data['task_info']['row_num']








def filter_infos_by_user_area(infos_40xtile, user_choose_area):
    """
    根据用户选区范围筛选 infos_40xtile。

    参数：
    - infos_40xtile: List[Dict]，原始的 tile 数据列表
    - user_choose_area: Dict，包括 min_row, max_row, min_col, max_col

    返回：
    - filtered_infos: List[Dict]，仅包含选区内的 tile 数据
    """
    min_row = user_choose_area["min_row"]
    max_row = user_choose_area["max_row"]
    min_col = user_choose_area["min_col"]
    max_col = user_choose_area["max_col"]

    filtered_infos = [
        info for info in infos_40xtile
        if min_row <= info["index_40xtile_y"] <= max_row and
           min_col <= info["index_40xtile_x"] <= max_col
    ]

    return filtered_infos



def save_scores_as_clean_grayscale(Scores_Matrix, save_path="scores_clean_gray.png"):
    # 可选归一化为 0-255 灰度图
    matrix = Scores_Matrix.copy()
    matrix[matrix == -1000] = np.nan  # 将默认无效值变为空
    min_val = np.nanmin(matrix)
    max_val = np.nanmax(matrix)

    # 归一化到 0-255 灰度
    norm = (matrix - min_val) / (max_val - min_val + 1e-5)
    gray_img = (norm * 255).astype(np.uint8)
    gray_img[np.isnan(norm)] = 0  # 将 NaN 填为黑色

    # 保存为图像
    plt.imsave(save_path, gray_img, cmap='gray', format='png')
    print(f"保存成功：{save_path}")


def visualize_selection(dict_info, Scores_Matrix, best_result_vis_path):
    """
    在 Scores_Matrix 上画出旋转框和所有点。

    参数：
    - dict_info: 包含旋转矩形信息和所有点的字典
    - Scores_Matrix: 原始图像（单通道或三通道），将复制用于可视化
    - best_result_vis_path： 保存可视化结果的路径
    """

    # 复制图像（转换为 BGR，便于彩色绘制）
    vis_img = Scores_Matrix.copy()
    if vis_img.dtype == np.float64:
        vis_img = vis_img - np.nanmin(vis_img)  # 平移到起点
        vis_img = vis_img / (np.nanmax(vis_img) + 1e-5)
        vis_img = (vis_img * 255).clip(0, 255).astype(np.uint8)

    if len(vis_img.shape) == 2:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

    # 获取旋转框四个顶点
    lt = tuple(map(int, dict_info['rotated_rect_lt']))
    rt = tuple(map(int, dict_info['rotated_rect_rt']))
    rb = tuple(map(int, dict_info['rotated_rect_rb']))
    lb = tuple(map(int, dict_info['rotated_rect_lb']))

    # 依次连接四个角画框（颜色红色）
    cv2.line(vis_img, lt, rt, (0, 0, 255), 1)
    cv2.line(vis_img, rt, rb, (0, 0, 255), 1)
    cv2.line(vis_img, rb, lb, (0, 0, 255), 1)
    cv2.line(vis_img, lb, lt, (0, 0, 255), 1)

    # 绘制所有 outPointsAll 中的点（颜色绿色）
    for pt in dict_info['outPointsAll']:
        pt_int = tuple(map(int, pt))
        cv2.circle(vis_img, pt_int, 1, (0, 255, 0), -1)  # filled circle

    plt.imsave(best_result_vis_path, vis_img, format='png')
    print(f"保存成功：{best_result_vis_path}")








def visualize_capture_tasks(Scores_Matrix, subset_outPoints, subset_rects, img_points_vis_path):
    """
    在 Scores_Matrix 上分别可视化所有任务点和矩形框。

    参数：
    - Scores_Matrix: 原始灰度图或得分图
    - subset_outPoints: List[List[Tuple[int, int]]]
    - subset_rects: List[Tuple[int, int, int, int]]
    - img_points_vis_path: 保存点可视化图像的路径
    
    返回：
    """
    if Scores_Matrix.dtype == np.float64:
        Scores_Matrix = Scores_Matrix - np.nanmin(Scores_Matrix)
        Scores_Matrix = Scores_Matrix / (np.nanmax(Scores_Matrix) + 1e-5)
        Scores_Matrix = (Scores_Matrix * 255).clip(0, 255).astype(np.uint8)

    # 转为彩色图像
    img_points = Scores_Matrix.copy()
    img_rects = Scores_Matrix.copy()

    if len(img_points.shape) == 2:
        img_points = cv2.cvtColor(img_points, cv2.COLOR_GRAY2BGR)
        img_rects = cv2.cvtColor(img_rects, cv2.COLOR_GRAY2BGR)

    N = len(subset_outPoints)

    for i in range(N):
        # 第一组为红色，其他为绿色
        if i == 0:
            color = (255, 0, 0)  # 红色 BGR
        else:
            color = (0, 255, 0)  # 绿色 BGR

        # 绘制点
        for pt in subset_outPoints[i]:
            pt = tuple(map(int, pt))
            cv2.circle(img_points, pt, 1, color, -1)

    plt.imsave(img_points_vis_path, img_points, format='png')
    print(f"保存成功：{img_points_vis_path}")






def get_points_in_rotated_rect1(lt, lb, rt, rb, width, height):
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



def get_points_in_rotated_rect(lt, lb, rt, rb, width, height):
    """
    获取旋转矩形中的所有整数点（像素坐标）

    参数:
    - lt, lb, rt, rb: 四个顶点（左上、左下、右上、右下），格式为 (x, y)
    - width: 分值图的宽，限制边界，防止超出
    - height: 分值图的高，限制边界，防止超出

    返回:
    - points: np.ndarray of shape (N, 2)，每行一个(x, y)
    """

    # 定义旋转矩形顶点顺序（必须是闭合多边形）
    contour = np.array([lb, lt, rt, rb], dtype=np.float32)

    # 计算包围盒（最小外接矩形的边界）
    x_min = max(int(np.floor(np.min(contour[:, 0]))), 0)
    x_max = min(int(np.ceil(np.max(contour[:, 0]))), width - 1)

    y_min = max(int(np.floor(np.min(contour[:, 1]))), 0)
    y_max = min(int(np.ceil(np.max(contour[:, 1]))), height - 1)


    # 构建所有候选点
    xs, ys = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
    all_points = np.stack([xs.ravel(), ys.ravel()], axis=-1)

    # 使用 pointPolygonTest 判断点是否在旋转多边形内
    contour = contour.reshape((-1, 1, 2))
    inside_mask = np.array([cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) >= 0 for pt in all_points])
    valid_points = all_points[inside_mask].tolist()

    return valid_points







def visualize_meg_view(x_min, y_min, x_max, y_max,
                       wbs_first_rect,
                       points,                # ndarray (N,2), 绝对坐标
                       rects_x100_sorted,     # [(x,y,w,h), ...] 绝对坐标
                       factor=20.0,
                       max_canvas_side=4096,
                       include_wbs=True,
                       save_dir: str | None = None):
    """
    可视化 meg 视野范围、所有点、x100 矩形框（缩小 20 倍）

    参数：
    - x_min, y_min, x_max, y_max: meg 细胞点的边界范围 绝对坐标（所有 meg 细胞点最小外接矩形 + padding）
    - wbs_first_rect: wbc视野的初始拍摄框 绝对坐标和尺寸
    - points: 所有细胞点的绝对坐标 ndarray，shape (N, 2)
    - rects_x100_sorted: 所有 x100 矩形框的绝对坐标列表，格式 [(x, y, w, h), ...]
    """

    # 1) 统一画布范围（meg bbox ∪ wbs）
    bx1, by1, bx2, by2 = x_min, y_min, x_max, y_max
    if include_wbs and wbs_first_rect is not None:
        wx, wy, ww, wh = wbs_first_rect
        bx1 = min(bx1, wx) - 100
        by1 = min(by1, wy) - 100
        bx2 = max(bx2, wx + ww) + 100
        by2 = max(by2, wy + wh) + 100

    W_full = int(bx2 - bx1 + 1)
    H_full = int(by2 - by1 + 1)

    s_fixed = 1.0 / float(factor)
    s_side  = min(max_canvas_side / max(W_full, 1), max_canvas_side / max(H_full, 1))
    s       = min(s_fixed, s_side)
    s       = min(s, 1.0)
    if s <= 0:
        s = s_fixed

    W_small = max(1, int(math.ceil(W_full * s)))
    H_small = max(1, int(math.ceil(H_full * s)))
    canvas  = np.zeros((H_small, W_small, 3), dtype=np.uint8)

    def to_small_xy(x_abs, y_abs):
        x_local = (x_abs - bx1) * s
        y_local = (y_abs - by1) * s
        xi = int(round(x_local))
        yi = int(round(y_local))
        xi = min(max(xi, 0), W_small - 1)
        yi = min(max(yi, 0), H_small - 1)
        return xi, yi

    def to_small_rect_with_min(x_abs, y_abs, w, h, min_wh=2):
        x1, y1 = to_small_xy(x_abs, y_abs)
        x2, y2 = to_small_xy(x_abs + w, y_abs + h)
        # 至少保证 min_wh 像素的可见尺寸
        if x2 <= x1:
            x2 = min(x1 + min_wh, W_small - 1)
            if x2 == x1 and x1 > 0:
                x1 -= 1
        if y2 <= y1:
            y2 = min(y1 + min_wh, H_small - 1)
            if y2 == y1 and y1 > 0:
                y1 -= 1
        return x1, y1, x2, y2

    print(f"[meg] union_bbox: ({bx1}, {by1}, {bx2}, {by2}), "
          f"orig_wh=({W_full},{H_full}), scale={s:.4f}, small_wh=({W_small},{H_small})")
    print(f"[meg] wbs: {wbs_first_rect}, meg_bbox: ({x_min},{y_min},{x_max},{y_max})")

    # ---- 先画点（白） ----
    if points is not None and len(points) > 0:
        shifted = (points - np.array([bx1, by1], dtype=np.float32)) * s
        sp = np.rint(shifted).astype(int)
        sp[:, 0] = np.clip(sp[:, 0], 0, W_small - 1)
        sp[:, 1] = np.clip(sp[:, 1], 0, H_small - 1)
        for x, y in sp:
            canvas[y, x] = (255, 255, 255)

    # ---- 再画 x100 矩形（绿） ----
    for i, (rx, ry, rw, rh) in enumerate(rects_x100_sorted):
        x1, y1, x2, y2 = to_small_rect_with_min(rx, ry, rw, rh, min_wh=2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(canvas, str(i), (x1, max(y1 - 3, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    # ---- 最后画 WBS（红，置顶）----
    if wbs_first_rect is not None:
        wx, wy, ww, wh = wbs_first_rect
        x1, y1, x2, y2 = to_small_rect_with_min(wx, wy, ww, wh, min_wh=2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)  # BGR 红色
        # 画中心十字，便于小框可见
        cx, cy = to_small_xy(wx + ww / 2.0, wy + wh / 2.0)
        cv2.drawMarker(canvas, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                       markerSize=8, thickness=2)
        cv2.putText(canvas, "WBS", (x1, max(y1 - 4, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir) 
    save_path = os.path.join(save_dir, "meg_rects_x100.png")
    cv2.imwrite(save_path, canvas)
    print(f"[meg] saved: {save_path}")






def visualize_wbc_rects_points(
    rects_x100,            # [(x,y,w,h), ...] 绝对坐标
    matched_points,        # [[ [x,y], ... ], ...] 与 rects_x100 一一对应
    save_path,             # 输出路径（含文件名）
    factor=20.0,           # 缩小倍数（越大越小）
    max_canvas_side=4096,  # 画布最长边上限
    pad=50,                # 边界留白（像素，绝对坐标系）
    min_rect_vis_px=2      # 缩小后矩形最小可视像素
):
    assert len(rects_x100) == len(matched_points), "rects_x100 与 matched_points 数量不一致"

    # ---- 1) 计算整体边界（矩形 ∪ 点），顺便加 pad ----
    xs, ys = [], []
    for (x,y,w,h) in rects_x100:
        xs += [x, x+w]
        ys += [y, y+h]
    for pts in matched_points:
        if pts:
            pts_arr = np.asarray(pts, dtype=float)
            xs += pts_arr[:,0].tolist()
            ys += pts_arr[:,1].tolist()

    if not xs or not ys:
        raise ValueError("没有可视化的数据：没有矩形也没有点")

    bx1 = int(np.floor(min(xs))) - int(pad)
    by1 = int(np.floor(min(ys))) - int(pad)
    bx2 = int(np.ceil (max(xs))) + int(pad)
    by2 = int(np.ceil (max(ys))) + int(pad)

    W_full = max(1, bx2 - bx1 + 1)
    H_full = max(1, by2 - by1 + 1)

    # ---- 2) 计算缩放比例 ----
    s_fixed = 1.0 / float(factor)
    s_side  = min(max_canvas_side / float(W_full), max_canvas_side / float(H_full))
    s = min(s_fixed, s_side, 1.0)
    if s <= 0: s = s_fixed

    W_small = max(1, int(math.ceil(W_full * s)))
    H_small = max(1, int(math.ceil(H_full * s)))
    canvas  = np.zeros((H_small, W_small, 3), dtype=np.uint8)  # 黑底，如要白底改成 255

    def to_small_xy(x_abs, y_abs):
        xi = int(round((x_abs - bx1) * s))
        yi = int(round((y_abs - by1) * s))
        xi = min(max(xi, 0), W_small - 1)
        yi = min(max(yi, 0), H_small - 1)
        return xi, yi

    def to_small_rect_with_min(x_abs, y_abs, w, h, min_wh=min_rect_vis_px):
        x1, y1 = to_small_xy(x_abs, y_abs)
        x2, y2 = to_small_xy(x_abs + w, y_abs + h)
        # 确保至少有 min_wh 像素可见
        if x2 <= x1:
            x2 = min(x1 + min_wh, W_small - 1)
            if x2 == x1 and x1 > 0: x1 -= 1
        if y2 <= y1:
            y2 = min(y1 + min_wh, H_small - 1)
            if y2 == y1 and y1 > 0: y1 -= 1
        return x1, y1, x2, y2

    # ---- 3) 颜色生成（BGR）：HSV 均匀取色，数量够多也不怕 ----
    def color_k(k, n):
        # 生成鲜艳颜色；OpenCV 用 BGR
        h = (k % n) / float(n)
        s_v = 200  # 饱和度/明度固定较高
        v_v = 255
        hsv = np.uint8([[[int(h*179), s_v, v_v]]])   # OpenCV HSV: H[0..179]
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0].tolist()
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    # ---- 4) 绘制：每个矩形及其点用同色 ----
    for i, ((rx,ry,rw,rh), pts) in enumerate(zip(rects_x100, matched_points)):
        col = color_k(i, max(1, len(rects_x100)))

        # 画矩形（置底）
        x1, y1, x2, y2 = to_small_rect_with_min(rx, ry, rw, rh)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{i}", (x1, max(0, y1-3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

        # 画该矩形对应的点（同色，实心小圆）
        if pts:
            pts_arr = np.asarray(pts, dtype=float)
            shifted = (pts_arr - np.array([bx1, by1], dtype=float)) * s
            sp = np.rint(shifted).astype(int)
            sp[:,0] = np.clip(sp[:,0], 0, W_small-1)
            sp[:,1] = np.clip(sp[:,1], 0, H_small-1)
            for (px, py) in sp:
                cv2.circle(canvas, (int(px), int(py)), 2, col, -1, cv2.LINE_AA)

    # ---- 5) 保存 ----
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, canvas)
    print(f"[wbc] saved: {save_path} | bbox=({bx1},{by1},{bx2},{by2}) scale={s:.4f} size=({W_small},{H_small})")



from sklearn.neighbors import KDTree

def _to_xy_array(points):
    if points is None: return np.empty((0,2), np.float32)
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size == 0: return np.empty((0,2), np.float32)
        if arr.size == 2: return arr.reshape(1,2)
        if arr.size % 2 == 0: return arr.reshape(-1,2)
        return np.empty((0,2), np.float32)
    return arr[:, :2].astype(np.float32)

def visualize_dedup_cells_once(
    infos_40xtile,
    user_choose_area,
    save_dir,
    tile_w=2448,
    tile_h=2048,
    img_dir=None,
    red_bgr=(0, 0, 255),
    green_bgr=(0, 255, 0),
    raw_radius=9,            # 原始点画大圈，突出
    dedup_radius=4,          # 去重点画小点
    thickness=2,
    legend=True,
    match_tol_px=6,          # raw 与 dedup 的匹配容差（像素）
    show_removed=True,       # 高亮被删重复点
    removed_bgr=(0, 215, 255)# 被删点用“金黄”
):
    """
    在每张 40× tile 上对比绘制：
      - raw：global_cell_centers（红色空心大圆 + 半透明）
      - dedup：global_cell_centers_dedup（绿色实心小点 + 黑描边）
      - removed（仅 raw 中存在、与任何 dedup 距离>match_tol_px）：黄色叉叉
    """
    os.makedirs(save_dir, exist_ok=True)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for t in infos_40xtile:
        ix = int(t.get('index_40xtile_x', 0))
        # 为了和你磁盘命名一致，这里仍按“翻转 y”生成文件名
        iy = user_choose_area["max_row"] - int(t.get('index_40xtile_y', 0)) - user_choose_area["min_row"]
        ax = int(t.get('abs_40xtile_x', 0))
        ay = int(t.get('abs_40xtile_y', 0))

        # 1) 取图；若无图则画灰底
        img = None
        if img_dir:
            p = os.path.join(img_dir, f"Pos[{ix}][{iy}].jpg")
            img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            img = np.full((tile_h, tile_w, 3), 40, np.uint8)
            cv2.putText(img, "no image", (20, 40), font, 1.0, (200,200,200), 2, cv2.LINE_AA)

        H, W = img.shape[:2]
        # 若实际尺寸与 tile_w/h 不同，做坐标缩放因子
        sx = W / float(tile_w)
        sy = H / float(tile_h)

        # 2) 全局→局部（在期望尺寸坐标系）
        raw_g   = _to_xy_array(t.get('global_cell_centers', []))
        dedup_g = _to_xy_array(t.get('global_cell_centers_dedup', []))

        def to_local_expected(gpts):
            if gpts.size == 0: return np.empty((0,2), np.float32)
            L = np.empty_like(gpts, np.float32)
            L[:,0] = gpts[:,0] - ax
            L[:,1] = gpts[:,1] - ay
            return L

        raw_loc  = to_local_expected(raw_g)
        ded_loc  = to_local_expected(dedup_g)

        # 3) 缩放到“实际图片像素坐标”
        def to_px(loc):
            if loc.size == 0: return np.empty((0,2), np.int32)
            P = np.empty_like(loc, np.int32)
            P[:,0] = np.clip(np.round(loc[:,0]*sx), 0, W-1).astype(np.int32)
            P[:,1] = np.clip(np.round(loc[:,1]*sy), 0, H-1).astype(np.int32)
            return P

        raw_px  = to_px(raw_loc)
        ded_px  = to_px(ded_loc)

        # 4) 先画 raw 空心大圆（用透明叠加，避免挡住结构）
        overlay = img.copy()
        for x, y in raw_px:
            cv2.circle(overlay, (int(x), int(y)), raw_radius, red_bgr, thickness, cv2.LINE_AA)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)  # 半透明叠加

        # 5) 高亮被删掉的重复点（raw 中与任何 dedup 距离>阈值者）
        removed_cnt = 0
        if show_removed and raw_px.shape[0] > 0 and ded_px.shape[0] > 0:
            tree = KDTree(ded_px.astype(np.float32), leaf_size=40)
            dists, _ = tree.query(raw_px.astype(np.float32), k=1)
            dists = dists.reshape(-1)
            removed_mask = dists > float(match_tol_px)
            removed_pts = raw_px[removed_mask]
            removed_cnt = int(removed_pts.shape[0])
            # 画黄色叉叉
            for x, y in removed_pts:
                x, y = int(x), int(y)
                l = max(6, dedup_radius+3)
                cv2.line(img, (x-l, y-l), (x+l, y+l), removed_bgr, 2, cv2.LINE_AA)
                cv2.line(img, (x-l, y+l), (x+l, y-l), removed_bgr, 2, cv2.LINE_AA)

        # 6) 再画 dedup（黑描边 + 绿色填充），置于最上层
        for x, y in ded_px:
            cv2.circle(img, (int(x), int(y)), dedup_radius+2, (0,0,0), -1, cv2.LINE_AA)   # 黑色外圈
            cv2.circle(img, (int(x), int(y)), dedup_radius,     green_bgr, -1, cv2.LINE_AA) # 绿色实心

        # 7) 文本标注
        cv2.putText(img, f"raw: {raw_px.shape[0]}",   (20, H-72), font, 0.9, red_bgr, 2, cv2.LINE_AA)
        cv2.putText(img, f"dedup: {ded_px.shape[0]}", (20, H-44), font, 0.9, green_bgr, 2, cv2.LINE_AA)
        if show_removed:
            cv2.putText(img, f"removed: {removed_cnt}", (20, H-16), font, 0.9, removed_bgr, 2, cv2.LINE_AA)
        cv2.putText(img, f"(x{ix}, y{iy})", (20, 80), font, 1.0, (255,255,255), 2, cv2.LINE_AA)

        if legend:
            x0, y0, w, h = W - 290, 20, 270, 110
            x0 = max(10, x0)
            pad = 6
            box = img.copy()
            cv2.rectangle(box, (x0, y0), (x0+w, y0+h), (0,0,0), -1)
            img = cv2.addWeighted(box, 0.35, img, 0.65, 0)
            # raw 圈
            cx = x0+24; cy = y0+28
            cv2.circle(img, (cx, cy), raw_radius, red_bgr, thickness, cv2.LINE_AA)
            cv2.putText(img, "raw (hollow)", (cx+20, cy+6), font, 0.6, red_bgr, 2, cv2.LINE_AA)
            # dedup 点
            cx = x0+24; cy = y0+60
            cv2.circle(img, (cx, cy), dedup_radius+2, (0,0,0), -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), dedup_radius, green_bgr, -1, cv2.LINE_AA)
            cv2.putText(img, "dedup (solid+outline)", (cx+20, cy+6), font, 0.6, green_bgr, 2, cv2.LINE_AA)
            # removed 叉
            if show_removed:
                cx = x0+24; cy = y0+92
                l = max(6, dedup_radius+3)
                cv2.line(img, (cx-l, cy-l), (cx+l, cy+l), removed_bgr, 2, cv2.LINE_AA)
                cv2.line(img, (cx-l, cy+l), (cx+l, cy-l), removed_bgr, 2, cv2.LINE_AA)
            cv2.putText(img, "removed duplicates", (cx+20, y0+92+6), font, 0.6, removed_bgr, 2, cv2.LINE_AA)

        # 8) 保存
        out_name = f"tile_x{ix}_y{iy}_raw{raw_px.shape[0]}_dedup{ded_px.shape[0]}.png"
        out_path = os.path.join(save_dir, out_name)
        cv2.imencode('.png', img)[1].tofile(out_path)
