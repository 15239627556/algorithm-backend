# -*- coding: utf-8 -*-
import os
import json
from collections import defaultdict

import cv2
import numpy as np



# ====== 工具函数 ======
def _to_int(x, default=None):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default

def _to_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

def _sort_key_for_view(t):
    BIG = 10 ** 9
    return (
        _to_int(t.get("sort_id"), BIG),
        _to_float(t.get("view_pos_y"), 0.0),
        _to_float(t.get("view_pos_x"), 0.0),
    )

def draw_label(img, text, org, color=(255, 255, 255), font_scale=0.35, thickness=1,
               outline=True, lineType=cv2.LINE_AA):
    x, y = int(org[0]), int(org[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    if outline and thickness >= 1:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    cv2.putText(img, text, (x + dx, y + dy), font, font_scale,
                                (0, 0, 0), thickness + 1, lineType)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, lineType)

def _to_xy_array(points):
    """
    统一为 (N,2) float 数组；支持：
    - [[x,y], ...]
    - 扁平 [x1,y1,x2,y2,...]
    - [{'x':..,'y':..}] / [{'cx':..,'cy':..}] / [{'cell_x':..,'cell_y':..}]
    """
    if points is None:
        return np.empty((0, 2), dtype=float)

    arr_obj = np.asarray(points, dtype=object)
    if arr_obj.dtype == object and arr_obj.size and isinstance(arr_obj.flat[0], dict):
        out = []
        for p in points:
            x = p.get("x", p.get("cx", p.get("cell_x", None)))
            y = p.get("y", p.get("cy", p.get("cell_y", None)))
            if x is None or y is None:
                continue
            out.append([float(x), float(y)])
        return np.asarray(out, dtype=float)

    arr = np.asarray(points, dtype=float)

    # 处理一维输入：允许 (x, y) 或 (x1, y1, x2, y2[, score])
    if arr.ndim == 1:
        if arr.size == 2:
            return arr.reshape(1, 2)
        elif arr.size >= 4:
            arr = arr.reshape(1, -1)
        else:
            return np.empty((0, 2), dtype=float)

    # 二维输入
    if arr.ndim == 2:
        if arr.shape[1] >= 4:
            centers = 0.5 * (arr[:, 0:2] + arr[:, 2:4])   # (N,2)
            # centers = np.rint(centers).astype(np.int32, copy=False)
            return centers.astype(float, copy=False)
        elif arr.shape[1] >= 2:
            return arr[:, :2].astype(float, copy=False)

    # 其他情况
    return np.empty((0, 2), dtype=float)



# ====== 主流程 ======
# json_dir = "/home/ubuntu/VScodeProjects/megLoc_heatmap/data/"
json_dir = "/home/ubuntu/VScodeProjects/megLoc_heatmap/data_20250929/"
project_dir = "/home/ubuntu/VScodeProjects/megLoc_heatmap/image/"
# project_list = ['data2025063003']
# project_list = ['847304e1494f4a9da1747b341bfed4a7']
project_list = os.listdir(json_dir)

    
for project_name in project_list:
    # data_json = f"/home/ubuntu/VScodeProjects/megLoc_heatmap/data/{project_name}.json"
    project_name = project_name.split('.')[0] if project_name.endswith('.json') else project_name
    data_json = f"/home/ubuntu/VScodeProjects/megLoc_heatmap/data_20250929/{project_name}.json"
    task_json = f"/home/ubuntu/VScodeProjects/megLoc_heatmap/new_output_8/{project_name}/x100_task_list.json"
    if not (os.path.exists(data_json) and os.path.exists(task_json)):
        print(f"[SKIP] missing {data_json} or {task_json}")
        

    with open(data_json, "r") as f:
        data = json.load(f)
    with open(task_json, "r") as f:
        x100_tasks = json.load(f)

    # 40× 图块信息
    images_info = []
    TILE_W, TILE_H = 2448, 2048
    for one in data["nodes"]:
        row = int(one["row"])
        col = int(one["col"])
        images_info.append(
            {
                "path": f"/home/ubuntu/VScodeProjects/megLoc_heatmap/image/{project_name}/Images/Pos[{col}][{row}].jpg",
                # "path": f"/home/ubuntu/VScodeProjects/megLoc_heatmap/debug_data/{project_name}/Images/Pos[{col}][{row}].jpg",
                
                "top_left": (
                    float(one["data"]["position"][0]),
                    float(one["data"]["position"][1]),
                ),
                "node": (row, col),
                "component_id": int(one["data"]["component_id"]),
                # 绿色圈将基于这些 tile 级别点：
                "wbc_center_pt": one["data"].get("wbc_center_pt", []),
                "meg_center_pt": one["data"].get("meg_center_pt", []),
                "area_score_info": one["data"].get("area_score_info", []),
            }
        )


    # wbc_tasks = sorted(x100_tasks[0], key=_sort_key_for_view)
    wbc_groups = [
        [t for t in group if isinstance(t, dict)]
        for group in x100_tasks[:-1]
    ]

    # —— 按 tile 聚合任务 & 统计索引范围（来自所有组的并集） ——
    tasks_by_tile_groups = []               # [defaultdict(list), ...] 与 wbc_groups 对应
    tiles_seen = set()                      # 用于决定拼图范围（所有组的并集）

    for group in wbc_groups:
        d = defaultdict(list)
        for t in group:
            r = _to_int(t.get("row_index"))
            c = _to_int(t.get("col_index"))
            if r is None or c is None:
                continue
            d[(r, c)].append(t)
            tiles_seen.add((r, c))
        tasks_by_tile_groups.append(d)

    if not tiles_seen:
        print("[WARN]: 没有命中的 WBC tile，跳过")
        # 可根据你的流程 return/exit

        

    # —— 拼图范围向外各扩 1 张（仅纳入确实存在的 tile） ——
    min_row = min(r for r, _ in tiles_seen) - 1
    max_row = max(r for r, _ in tiles_seen) + 1
    min_col = min(c for _, c in tiles_seen) - 1
    max_col = max(c for _, c in tiles_seen) + 1
    num_rows = max_row - min_row + 1
    num_cols = max_col - min_col + 1
    print(
        f"[INFO] 40×(WBC) 拼图索引范围(扩1): row {min_row}..{max_row}, col {min_col}..{max_col} "
        f"（共 {num_rows}×{num_cols} 张，可能存在缺片）"
    )

    info_by_node = {info["node"]: info for info in images_info}
    infos_in_rect = []
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            if (r, c) in info_by_node:
                infos_in_rect.append(info_by_node[(r, c)])

    if not infos_in_rect:
        print(f"[WARN]: 范围内找不到任何实际图片，跳过")
        

    # —— 绝对坐标包围盒 + 缩放 ——
    min_x_abs = min(info["top_left"][0] for info in infos_in_rect)
    min_y_abs = min(info["top_left"][1] for info in infos_in_rect)
    max_x_abs = max(info["top_left"][0] + TILE_W for info in infos_in_rect)
    max_y_abs = max(info["top_left"][1] + TILE_H for info in infos_in_rect)

    full_w = int(np.ceil(max_x_abs - min_x_abs))
    full_h = int(np.ceil(max_y_abs - min_y_abs))
    if full_w <= 0 or full_h <= 0:
        print(f"[WARN]: 无效拼图尺寸，跳过")
        

    max_long = 4000
    scale = min(1.0, max_long / max(full_w, full_h))
    out_w = int(np.ceil(full_w * scale))
    out_h = int(np.ceil(full_h * scale))
    print(
        f"[INFO] : 绝对坐标拼图 {full_w}×{full_h}，缩放 {scale:.3f} → 输出 {out_w}×{out_h}"
    )

    # —— 输出路径 ——

    OUT_DIR = f"/home/ubuntu/VScodeProjects/megLoc_heatmap/new_output_8/{project_name}"
    os.makedirs(OUT_DIR, exist_ok=True)
    MOS_OUT = os.path.join(OUT_DIR, "mosaic_40x_WBC_absRange_all_x100task_absCoor.png")
    mosaic = np.zeros((out_h, out_w, 3), dtype=np.uint8)



    # ========== 仅用于 tile 小图上叠加（行列号 / 四宫格类型） ==========
    def _draw_overlays_on_tile_scaled(tile_img_small, row, col, info):
        font_scale_box = 0.35
        font_thick = 1
        h, w = tile_img_small.shape[:2]
        half_w = w // 2
        half_h = h // 2

        draw_label(
            tile_img_small,
            f"r{int(row)},c{int(col)}",
            (4, 14),
            color=(255, 255, 255),
            font_scale=font_scale_box,
            thickness=font_thick,
            outline=True,
        )

        for i, item in enumerate(info.get("area_score_info", [])):
            try:
                typeid_ = (
                    item.get("typeid")
                    if isinstance(item, dict)
                    else (item[5] if len(item) >= 6 else None)
                )
                if typeid_ is None:
                    continue
                if i == 0:
                    tx, ty = 0, 0
                elif i == 1:
                    tx, ty = half_w, 0
                elif i == 2:
                    tx, ty = 0, half_h
                else:
                    tx, ty = half_w, half_h
                draw_label(
                    tile_img_small,
                    f"{int(typeid_)}",
                    (int(tx), int(ty) + 24),
                    color=(0, 255, 255),
                    font_scale=font_scale_box,
                    thickness=font_thick,
                    outline=True,
                )
            except Exception:
                pass
        return tile_img_small

    # ========== 先全部贴图 & 同时收集“绿色点”的 mosaic 坐标 ==========
    green_pts_all = []  # list of (N_i,2) arrays in mosaic coord
    green_abs_pts = []
    out_tile_w = max(1, int(round(TILE_W * scale)))
    out_tile_h = max(1, int(round(TILE_H * scale)))

    for info in infos_in_rect:
        row, col = info["node"]
        img = cv2.imread(info["path"])
        if img is None:
            continue

        # tile 在 mosaic 中左上角位置（以 min_x_abs/min_y_abs 为原点）
        sx = int(round((info["top_left"][0] - min_x_abs) * scale))
        sy = int(round((info["top_left"][1] - min_y_abs) * scale))

        # 缩放
        img_small = cv2.resize(img, (out_tile_w, out_tile_h), interpolation=cv2.INTER_AREA)

        # 仅在小图上叠加行列号/四宫格类型
        img_small = _draw_overlays_on_tile_scaled(img_small, row, col, info)

        # 贴到 mosaic（越界则裁剪）
        y1, x1 = sy, sx
        y2, x2 = sy + img_small.shape[0], sx + img_small.shape[1]
        if y1 >= out_h or x1 >= out_w or y2 <= 0 or x2 <= 0:
            continue
        yy1, yy2 = max(0, y1), min(out_h, y2)
        xx1, xx2 = max(0, x1), min(out_w, x2)
        mosaic[yy1:yy2, xx1:xx2] = img_small[(yy1 - y1):(yy2 - y1), (xx1 - x1):(xx2 - x1)]

        # —— 收集该 tile 的绿色点（转换为 mosaic 坐标；此时先不画，避免被后续贴图覆盖） ——
        pts_local = _to_xy_array(info.get("wbc_center_pt", []))
        if pts_local.size:
            tlx, tly = info["top_left"]
            # 局部 → 绝对
            pts_abs = pts_local + np.array([tlx, tly], dtype=float)
            green_abs_pts.append(pts_abs)

    # ========== 全部贴图完成后，再统一画“绿色圈”（空心，不遮挡黄色） ==========
    def S_abs(ax, ay):
        mx = int(round((float(ax) - float(min_x_abs)) * float(scale)))
        my = int(round((float(ay) - float(min_y_abs)) * float(scale)))
        return mx, my

    if green_abs_pts:
        H, W = mosaic.shape[:2]
        green_abs_all = np.vstack(green_abs_pts)
        for ax, ay in green_abs_all:
            px, py = S_abs(ax, ay)   # 统一过同一个 S_abs
            if 0 <= px < W and 0 <= py < H:
                cv2.circle(mosaic, (px, py), 3, (0,255,0), 1, cv2.LINE_AA)

    

    def draw_wbc_groups_on_mosaic_absolute(
        mosaic,
        tasks_by_tile_groups,   # <<< 新：每个组一个 defaultdict
        info_by_node,
        min_x_abs, min_y_abs,
        scale
    ):
        H, W = mosaic.shape[:2]

        def S_abs(ax, ay):
            mx = int(round((float(ax) - float(min_x_abs)) * float(scale)))
            my = int(round((float(ay) - float(min_y_abs)) * float(scale)))
            return mx, my

        def clip_rect(x1, y1, x2, y2):
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W - 1, x2))
            y2 = max(0, min(H - 1, y2))
            return x1, y1, x2, y2

        # OpenCV 为 BGR
        palette = [
            (0,   0, 255),   # 红
            (0, 255, 255),   # 黄
            (255,255,  0),   # 青
            (255,  0,255),   # 粉/洋红
            (0, 255,  0),    # 绿
            (255,165,  0),   # 橙（近似）
            (255,  0,  0),   # 蓝
        ]

        legend_x, legend_y = 8, 18
        for gi, tasks_by_tile in enumerate(tasks_by_tile_groups):
            color = palette[gi % len(palette)]
            for (row, col), task_list in tasks_by_tile.items():
                info = info_by_node.get((row, col))
                if info is None:
                    continue
                tlx, tly = info["top_left"]  # tile 在全局 40× 绝对坐标
                # 画图例
                cv2.rectangle(mosaic, (legend_x, legend_y-10), (legend_x+12, legend_y+2), color, -1)
                draw_label(mosaic, f"G{gi+1}", (legend_x+18, legend_y), color=color, font_scale=0.5, thickness=1)
                legend_y += 16
                # 每组内按原排序规则画
                for idx, t in enumerate(sorted(task_list, key=_sort_key_for_view), start=1):
                    vx = _to_float(t.get("view_pos_x"))
                    vy = _to_float(t.get("view_pos_y"))
                    vw = _to_float(t.get("view_width"))
                    vh = _to_float(t.get("view_height"))

                    # 绝对坐标（未缩放）
                    ax1 = tlx + vx
                    ay1 = tly + vy
                    ax2 = ax1 + vw
                    ay2 = ay1 + vh

                    # 画矩形（组颜色）
                    mx1, my1 = S_abs(ax1, ay1)
                    mx2, my2 = S_abs(ax2, ay2)
                    if mx2 <= mx1 or my2 <= my1:
                        continue
                    mx1, my1, mx2, my2 = clip_rect(mx1, my1, mx2, my2)
                    cv2.rectangle(mosaic, (mx1, my1), (mx2, my2), color, 2)

                    # 标号：组号-顺序（例如 2-15）
                    order = _to_int(t.get("sort_id"), idx)
                    y_text = max(10, my1 - 4)
                    draw_label(
                        mosaic,
                        f"{gi+1}-{order}",
                        (mx1, y_text),
                        color=color,
                        font_scale=0.5,
                        thickness=1,
                        outline=True,
                    )

                    # 任务内细胞中心点（用同组颜色）
                    cells = t.get("cell_list", [])
                    for c in cells:
                        cx_local = _to_float(c.get("cell_x"))
                        cy_local = _to_float(c.get("cell_y"))
                        cw = _to_float(c.get("cell_width"))
                        ch = _to_float(c.get("cell_height"))
                        cx_abs = tlx + cx_local + cw * 0.5
                        cy_abs = tly + cy_local + ch * 0.5
                        px, py = S_abs(cx_abs, cy_abs)
                        if 0 <= px < W and 0 <= py < H:
                            cv2.circle(mosaic, (px, py), radius=2, color=color, thickness=-1)




    draw_wbc_groups_on_mosaic_absolute(
        mosaic=mosaic,
        tasks_by_tile_groups=tasks_by_tile_groups,   # <<< 新参数
        info_by_node=info_by_node,
        min_x_abs=min_x_abs,
        min_y_abs=min_y_abs,
        scale=scale,
    )




    ok = cv2.imwrite(MOS_OUT, mosaic)
    print(f"[{'OK' if ok else 'FAIL'}] 保存拼图: {MOS_OUT}")
