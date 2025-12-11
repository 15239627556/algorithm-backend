import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import os
from collections import OrderedDict, defaultdict

project_dir = "/home/ubuntu/VScodeProjects/megLoc_heatmap/image/"
# project_list = os.listdir(project_dir)
project_list = ['data2025063021']
# project_list = ['data2025070304','data2025063004','data2025063003','data2025063002','data2025063001']

# ----------------- 工具函数 -----------------
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
    BIG = 10**9
    return (_to_int(t.get('sort_id'), BIG),
            _to_float(t.get('view_pos_y'), 0.0),
            _to_float(t.get('view_pos_x'), 0.0))

def _norm_vtype(v):
    v = str(v or "").strip().lower()
    if "wbc" in v: return "wbc"
    if "meg" in v: return "meg"
    return None

def draw_label(img, text, org, color=(255,255,255), font_scale=0.35, thickness=1,
               outline=True, lineType=cv2.LINE_AA):
    """仅描边的小号文字，避免遮挡"""
    x, y = int(org[0]), int(org[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    if outline and thickness >= 1:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    cv2.putText(img, text, (x+dx, y+dy), font, font_scale,
                                (0,0,0), thickness+1, lineType)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, lineType)

# ----------------- 主流程 -----------------
for project_name in project_list:
    # 仅处理目录
    if not os.path.isdir(os.path.join(project_dir, project_name)):
        continue

    data_json = f"/home/ubuntu/VScodeProjects/megLoc_heatmap/data/{project_name}.json"
    task_json = f"/home/ubuntu/VScodeProjects/megLoc_heatmap/output3/{project_name}/x100_task_list.json"
    if not (os.path.exists(data_json) and os.path.exists(task_json)):
        print(f"[SKIP] missing {data_json} or {task_json}")
        continue

    with open(data_json, 'r') as f:
        data = json.load(f)
    with open(task_json, "r") as f:
        x100_tasks = json.load(f)

    # 40× 图块信息
    images_info = []
    TILE_W, TILE_H = 2448, 2048
    for one in data['nodes']:
        row = int(one['row'])
        col = int(one['col'])
        images_info.append({
            'path': f"/home/ubuntu/VScodeProjects/megLoc_heatmap/image/{project_name}/Images/Pos[{col}][{row}].jpg",
            'top_left': (float(one['data']['position'][0]), float(one['data']['position'][1])),  # 绝对坐标
            'node': (row, col),
            "component_id": int(one['data']['component_id']),
            'wbc_center_pt': one['data'].get('wbc_center_pt', []),
            'meg_center_pt': one['data'].get('meg_center_pt', []),
            'area_score_info': one['data'].get('area_score_info', []),
        })

    # 选取视野（WBC 全部 + MEG 前 N）
    N_TOP = 50
    top0 = sorted(x100_tasks[0], key=_sort_key_for_view)                  # WBC 全部
    top1 = sorted(x100_tasks[-1], key=_sort_key_for_view)[:N_TOP]         # MEG 前 N

    # 将任务按 tile 聚合 + 统计“索引范围”
    tasks_by_tile_top = defaultdict(lambda: {"wbc": [], "meg": []})
    tiles_seen = set()
    for t in (top0 + top1):
        r = _to_int(t.get("row_index"))
        c = _to_int(t.get("col_index"))
        if r is None or c is None:
            continue
        vtype = _norm_vtype(t.get("view_type"))
        if vtype is None:
            continue
        tasks_by_tile_top[(r, c)][vtype].append(t)
        tiles_seen.add((r, c))

    if not tiles_seen:
        print(f"[WARN] {project_name}: 没有命中的 tile，跳过")
        continue

    # —— 关键修改处：用“索引范围内的所有 tile”来拼接（而不是只放有任务的）——
    min_row = min(r for r, _ in tiles_seen)
    max_row = max(r for r, _ in tiles_seen)
    min_col = min(c for _, c in tiles_seen)
    max_col = max(c for _, c in tiles_seen)
    num_rows = max_row - min_row + 1
    num_cols = max_col - min_col + 1
    print(f"[INFO] 40× 拼图索引范围: row {min_row}..{max_row}, col {min_col}..{max_col} （共 {num_rows}×{num_cols} 张）")


    # 从 images_info 中筛出“索引范围内”的所有 tile（即使没有任务也要贴上）
    info_by_node = {info['node']: info for info in images_info}
    infos_in_rect = []
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            if (r, c) in info_by_node:
                infos_in_rect.append(info_by_node[(r, c)])
            else:
                # 缺图的 tile 不加入（对应位置会保持为黑）
                pass

    if not infos_in_rect:
        print(f"[WARN] {project_name}: 范围内找不到任何实际图片，跳过")
        continue

    # 用“范围内的所有 tile”的 top_left 绝对坐标计算包围盒和缩放
    min_x_abs = min(info['top_left'][0] for info in infos_in_rect)
    min_y_abs = min(info['top_left'][1] for info in infos_in_rect)
    max_x_abs = max(info['top_left'][0] + TILE_W for info in infos_in_rect)
    max_y_abs = max(info['top_left'][1] + TILE_H for info in infos_in_rect)

    full_w = int(np.ceil(max_x_abs - min_x_abs))
    full_h = int(np.ceil(max_y_abs - min_y_abs))
    if full_w <= 0 or full_h <= 0:
        print(f"[WARN] {project_name}: 无效拼图尺寸，跳过")
        continue

    max_long = 4000
    scale = min(1.0, max_long / max(full_w, full_h))
    out_w = int(np.ceil(full_w * scale))
    out_h = int(np.ceil(full_h * scale))
    print(f"[INFO] {project_name}: 绝对坐标拼图 {full_w}×{full_h}，缩放 {scale:.3f} → 输出 {out_w}×{out_h}")

    # 输出
    OUT_DIR = f"/home/ubuntu/VScodeProjects/megLoc_heatmap/output4/{project_name}"
    os.makedirs(OUT_DIR, exist_ok=True)
    MOS_OUT = os.path.join(OUT_DIR, f"mosaic_40x_top{N_TOP*2}_absRange_absCoor.png")

    mosaic = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # 在缩小后的 tile 上叠加 WBC/MEG 任务框与编号（红/绿，小号字）
    def _draw_overlays_on_tile_scaled(tile_img_small, row, col, scale_for_coord, info):
    
        font_scale_box = 0.35
        font_thick = 1

        h, w = tile_img_small.shape[:2]
        half_w = w // 2
        half_h = h // 2

        # 左上角写 row/col（可选）
        draw_label(tile_img_small, f"r{int(row)},c{int(col)}", (4, 14),
                color=(255, 255, 255), font_scale=font_scale_box, thickness=font_thick, outline=True)

        # 仅画四宫格 typeid（可选）
        for i, item in enumerate(info.get('area_score_info', [])):
            try:
                typeid_ = (item.get('typeid') if isinstance(item, dict)
                        else (item[5] if len(item) >= 6 else None))
                if typeid_ is None:
                    continue
                if i == 0:      tx, ty = 0, 0
                elif i == 1:    tx, ty = half_w, 0
                elif i == 2:    tx, ty = 0, half_h
                else:           tx, ty = half_w, half_h
                draw_label(tile_img_small, f"{int(typeid_)}", (int(tx), int(ty)+24),
                        color=(0,255,255), font_scale=font_scale_box, thickness=font_thick, outline=True)
            except Exception:
                pass

        # 不在 tile 上画 WBC/MEG 框，避免被覆盖
        return tile_img_small


    def _draw_tasks_on_mosaic_absolute(
        mosaic,                      # 已拼好的整张缩放后大图 (H×W×3)
        tasks_by_tile_top,           # 你的聚合任务 dict: {(row,col): {"wbc":[...], "meg":[...]}}
        info_by_node,                # {(row,col): info}, 其中 info['top_left'] 为 tile 的绝对坐标(未缩放)
        min_x_abs, min_y_abs,        # 本次拼图的绝对坐标原点（左上角）
        scale                        # 绝对坐标→mosaic像素的缩放
    ):
        H, W = mosaic.shape[:2]

        def S_abs(ax, ay):
            """绝对坐标 → mosaic 像素坐标"""
            mx = int(round((float(ax) - float(min_x_abs)) * float(scale)))
            my = int(round((float(ay) - float(min_y_abs)) * float(scale)))
            return mx, my

        def clip_rect(x1, y1, x2, y2):
            """裁剪到图内，保证 OpenCV 不越界"""
            x1 = max(0, min(W-1, x1))
            y1 = max(0, min(H-1, y1))
            x2 = max(0, min(W-1, x2))
            y2 = max(0, min(H-1, y2))
            return x1, y1, x2, y2

        def _key(t):
            BIG = 10**9
            return (_to_int(t.get('sort_id'), BIG),
                    _to_float(t.get('view_pos_y'), 0.0),
                    _to_float(t.get('view_pos_x'), 0.0))

        # 统一在 mosaic 上画，后画的不会被 tile 覆盖了
        for (row, col), groups in tasks_by_tile_top.items():
            info = info_by_node.get((row, col))
            if info is None:
                continue
            tlx, tly = info['top_left']  # 该 tile 在原始 40× 绝对坐标系中的左上角

            for vtype, color in (("wbc", (0, 0, 255)), ("meg", (0, 255, 0))):
                for idx, t in enumerate(sorted(groups.get(vtype, []), key=_key), start=1):
                    vx = _to_float(t.get("view_pos_x"));  vy = _to_float(t.get("view_pos_y"))
                    vw = _to_float(t.get("view_width"));  vh = _to_float(t.get("view_height"))

                    # —— 绝对坐标（未缩放）——
                    ax1 = tlx + vx
                    ay1 = tly + vy
                    ax2 = ax1 + vw
                    ay2 = ay1 + vh

                    # —— 映射到 mosaic 像素 —— 
                    mx1, my1 = S_abs(ax1, ay1)
                    mx2, my2 = S_abs(ax2, ay2)
                    # 防止舍入导致的反转/退化
                    if mx2 <= mx1 or my2 <= my1:
                        continue
                    mx1, my1, mx2, my2 = clip_rect(mx1, my1, mx2, my2)

                    # 画框+编号
                    cv2.rectangle(mosaic, (mx1, my1), (mx2, my2), color, 2)
                    order = _to_int(t.get("sort_id"), idx)
                    y_text = max(10, my1 - 4)
                    draw_label(mosaic, f"{order}", (mx1, y_text),color=color, font_scale=0.5, thickness=1, outline=True)



    # 逐 tile：按 top_left 绝对坐标摆放（范围内所有 tile），先缩放再叠加，再贴到 mosaic
    out_tile_w = max(1, int(round(TILE_W * scale)))
    out_tile_h = max(1, int(round(TILE_H * scale)))

    for info in infos_in_rect:
        row, col = info['node']
        img = cv2.imread(info['path'])
        if img is None:
            continue

        # tile 在整幅中的位置（以 min_x_abs/min_y_abs 为原点）
        sx = int(round((info['top_left'][0] - min_x_abs) * scale))
        sy = int(round((info['top_left'][1] - min_y_abs) * scale))

        # 缩放
        img_small = cv2.resize(img, (out_tile_w, out_tile_h), interpolation=cv2.INTER_AREA)

        # 叠加（只有有任务的 tile 会画框；没任务的 tile 只贴图）
        img_small = _draw_overlays_on_tile_scaled(img_small, row, col, scale, info)

        # 贴到 mosaic（越界则裁剪）
        y1, x1 = sy, sx
        y2, x2 = sy + img_small.shape[0], sx + img_small.shape[1]
        if y1 >= out_h or x1 >= out_w or y2 <= 0 or x2 <= 0:
            continue
        yy1, yy2 = max(0, y1), min(out_h, y2)
        xx1, xx2 = max(0, x1), min(out_w, x2)
        mosaic[yy1:yy2, xx1:xx2] = img_small[(yy1 - y1):(yy2 - y1), (xx1 - x1):(xx2 - x1)]
    
    _draw_tasks_on_mosaic_absolute(
        mosaic=mosaic,
        tasks_by_tile_top=tasks_by_tile_top,
        info_by_node=info_by_node,
        min_x_abs=min_x_abs,
        min_y_abs=min_y_abs,
        scale=scale
    )
    ok = cv2.imwrite(MOS_OUT, mosaic)
    print(f"[{'OK' if ok else 'FAIL'}] 保存拼图: {MOS_OUT}")
