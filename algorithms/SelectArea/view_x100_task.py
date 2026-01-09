# import cv2
# import os
# import json
# from collections import OrderedDict, defaultdict  


# # ================== 读 JSON ==================
# with open("/home/ubuntu/VScodeProjects/megLoc_heatmap/data/data2025063009.json", 'r') as f:
#     data = json.load(f)


# # ===== 读取 x100 百倍任务（首个 WBC 初始框、最后一个或若干个 MEG 框）=====
# TASK_JSON = "/home/ubuntu/VScodeProjects/megLoc_heatmap/output5/data2025063009/x100_task_list.json"  
# with open(TASK_JSON, "r") as f:
#     x100_tasks = json.load(f)  



# # ==== 工具函数 ====
# def _to_int(x, default=None):
#     try:
#         return int(x)
#     except (TypeError, ValueError):
#         return default

# def _to_float(x, default=0.0):
#     try:
#         return float(x)
#     except (TypeError, ValueError):
#         return default

# # ==== 1) 重新构建 tasks_by_tile：规范化 view_type，并为缺失 sort_id 补回退值 ====
# tasks_by_tile = defaultdict(lambda: {"wbc": [], "meg": []})

# for j, task_list in enumerate([x100_tasks[0], x100_tasks[-1]]):  # WBC 初始框 + MEG 最终框
#     for i, t in enumerate(task_list):
#         if j == 1 and i > 50: break  # MEG 最多取前 30 个
#         if not isinstance(t, dict):
#             continue
#         r = _to_int(t.get("row_index"))
#         c = _to_int(t.get("col_index"))
#         if r is None or c is None:
#             continue

#         # 规范化 view_type 到 {"wbc","meg"}
#         vtype = str(t.get("view_type", "")).strip().lower()
#         if vtype not in ("wbc", "meg"):
#             if "wbc" in vtype:
#                 vtype = "wbc"
#             elif "meg" in vtype:
#                 vtype = "meg"
#             else:
#                 continue  # 其他类型跳过

#         # 复制任务字典并保证有整数 sort_id（没有就用本列表枚举序 i+1）
#         t2 = dict(t)
#         sid = _to_int(t2.get("sort_id"))
#         t2["sort_id"] = sid if sid is not None else (i + 1)

#         tasks_by_tile[(r, c)][vtype].append(t2)

# # ==== 2) 排序函数：优先按 sort_id，其次 (y, x) 做稳定次序 ====
# def _sorted_tasks(tasks):
#     BIG = 10**9
#     return sorted(
#         tasks or [],
#         key=lambda task: (
#             _to_int(task.get('sort_id'), BIG),
#             _to_float(task.get('view_pos_y'), 0.0),
#             _to_float(task.get('view_pos_x'), 0.0),
#         )
#     )

# # ==== 3) 遍历每个 tile，载入 40 倍图，绘制并保存 ====
# IMAGES_DIR_40X = "/home/ubuntu/VScodeProjects/megLoc_heatmap/image/data2025063009/Images"
# OUT_DIR = "/home/ubuntu/VScodeProjects/megLoc_heatmap/output5/data2025063009/tiles_40x_annot"
# os.makedirs(OUT_DIR, exist_ok=True)

# for (row, col), groups in tasks_by_tile.items():
#     img_path = f"{IMAGES_DIR_40X}/Pos[{col}][{row}].jpg"
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"[WARN] 载入失败，跳过: {img_path}")
#         continue
    

#     # --- data: area_score_info（四宫格 typeid 到四象限角落） ---
#     for i, item in enumerate(groups.get('area_score_info', [])):
#         try:
#             typeid_ = (item.get('typeid') if isinstance(item, dict)
#                     else (item[5] if len(item) >= 6 else None))
#             if typeid_ is None:
#                 continue
#             if i == 0:
#                 tx, ty = 0, 0
#             elif i == 1:
#                 tx, ty = 1224, 0
#             elif i == 2:
#                 tx, ty = 0, 1024
#             else:
#                 tx, ty = 1224, 1024

#             cv2.putText(img, f"{int(typeid_)}", (tx, ty + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#         except Exception:
#             pass

    
#     # ---- 画 WBC（红色）----
#     wbc_list = _sorted_tasks(groups.get("wbc", []))
#     for idx, t in enumerate(wbc_list, start=1):
#         vx = int(_to_float(t.get("view_pos_x")))
#         vy = int(_to_float(t.get("view_pos_y")))
#         vw = int(_to_float(t.get("view_width")))
#         vh = int(_to_float(t.get("view_height")))
#         cv2.rectangle(img, (vx, vy), (vx + vw, vy + vh), (0, 0, 255), 2)

#         # 顺序号：优先 sort_id，否则用 idx
#         order = _to_int(t.get("sort_id"), idx)
#         cv2.putText(img, f"{order}", (vx, vy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#         # 细胞（红点，或改成小矩形）
#         for cell in t.get("cell_list", []):
#             cx = int(_to_float(cell.get("cell_x")))
#             cy = int(_to_float(cell.get("cell_y")))
#             # cw = int(_to_float(cell.get("cell_width"))); ch = int(_to_float(cell.get("cell_height")))
#             cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
#             # 若要框：cv2.rectangle(img, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 1)

#     # ---- 画 MEG（绿色）----
#     meg_list = _sorted_tasks(groups.get("meg", []))
#     for idx, t in enumerate(meg_list, start=1):
#         vx = int(_to_float(t.get("view_pos_x")))
#         vy = int(_to_float(t.get("view_pos_y")))
#         vw = int(_to_float(t.get("view_width")))
#         vh = int(_to_float(t.get("view_height")))
#         cv2.rectangle(img, (vx, vy), (vx + vw, vy + vh), (0, 255, 0), 2)

#         order = _to_int(t.get("sort_id"), idx)
#         cv2.putText(img, f"{order}", (vx, vy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         for cell in t.get("cell_list", []):
#             cx = int(_to_float(cell.get("cell_x")))
#             cy = int(_to_float(cell.get("cell_y")))
#             # cw = int(_to_float(cell.get("cell_width"))); ch = int(_to_float(cell.get("cell_height")))
#             cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)
#             # 若要框：cv2.rectangle(img, (cx, cy), (cx+cw, cy+ch), (0, 255, 0), 1)

#     out_path = os.path.join(OUT_DIR, f"Pos[{col}][{row}]_40x_annot.jpg")
#     ok = cv2.imwrite(out_path, img)
#     if not ok:
#         print(f"[WARN] 保存失败: {out_path}")
#     else:
#         print(f"[OK] 保存: {out_path}")

# -*- coding: utf-8 -*-
import os
import json
from collections import defaultdict

import cv2
import numpy as np

# ---------- paths ----------
# project_name = "data2025063009"

# DATA_JSON = f"/home/ubuntu/VScodeProjects/megLoc_heatmap/data/{project_name}.json"
# TASK_JSON = f"/home/ubuntu/VScodeProjects/megLoc_heatmap/output5/{project_name}/x100_task_list.json"
# IMAGES_DIR_40X = f"/home/ubuntu/VScodeProjects/megLoc_heatmap/image/{project_name}/Images"
# OUT_DIR = f"/home/ubuntu/VScodeProjects/megLoc_heatmap/output5/{project_name}/tiles_40x_annot"
# os.makedirs(OUT_DIR, exist_ok=True)



project_name = '64afc329999f473786f1b439c944aeb2'
DATA_JSON = f"/home/ubuntu/VScodeProjects/megLoc_heatmap/data/{project_name}.json"
TASK_JSON = f"/home/ubuntu/VScodeProjects/megLoc_heatmap/output7/{project_name}/x100_task_list.json"
IMAGES_DIR_40X = f"/home/ubuntu/VScodeProjects/megLoc_heatmap/image/{project_name}/Images/"
OUT_DIR =  f"/home/ubuntu/VScodeProjects/megLoc_heatmap/output7/{project_name}"
os.makedirs(OUT_DIR, exist_ok=True)
# ---------- utils ----------
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

def _to_xy_array(points):
    """统一为 (N,2) float 数组"""
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
    if arr.ndim == 1:
        if arr.size % 2 != 0:
            return np.empty((0, 2), dtype=float)
        return arr.reshape(-1, 2)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, :2].astype(float)
    return np.empty((0, 2), dtype=float)

def _sorted_tasks(tasks):
    BIG = 10**9
    return sorted(
        tasks or [],
        key=lambda task: (
            _to_int(task.get('sort_id'), BIG),
            _to_float(task.get('view_pos_y'), 0.0),
            _to_float(task.get('view_pos_x'), 0.0),
        )
    )

# ---------- load ----------
with open(DATA_JSON, "r") as f:
    data = json.load(f)
with open(TASK_JSON, "r") as f:
    x100_tasks = json.load(f)

# 建 node 索引，后续取四宫格/原始点
node_by_rc = {(int(n["row"]), int(n["col"])): n for n in data["nodes"]}

# ---------- 仅保留“被 WBC 任务涉及到的 40× tile” ----------
# 找到所有 WBC 涉及的 (row, col)
wbc_tiles = set()
for task_list in x100_tasks:
    for t in task_list:
        if not isinstance(t, dict):
            continue
        vtype = str(t.get("view_type", "")).strip().lower()
        is_wbc = (vtype == "wbc") or ("wbc" in vtype)
        if not is_wbc:
            continue
        r = _to_int(t.get("row_index"))
        c = _to_int(t.get("col_index"))
        if r is not None and c is not None:
            wbc_tiles.add((r, c))

if not wbc_tiles:
    print("[INFO] 没有检测到任何 WBC 相关的 40× tile，结束。")
    raise SystemExit(0)

# 结构：tasks_by_tile[(row,col)] = {"wbc":[...], "meg":[...], "org_wbc":[(x,y),...]}
tasks_by_tile = defaultdict(lambda: {"wbc": [], "meg": [], "org_wbc": []})

# 只把“属于 wbc_tiles 的任务”归入 tasks_by_tile（WBC 与 MEG 都可以画，但仅限这些 tile）
for task_list in x100_tasks:
    for i, t in enumerate(task_list):
        if not isinstance(t, dict):
            continue
        r = _to_int(t.get("row_index"))
        c = _to_int(t.get("col_index"))
        if r is None or c is None or (r, c) not in wbc_tiles:
            continue

        vtype_raw = str(t.get("view_type", "")).strip().lower()
        if "wbc" in vtype_raw:
            vtype = "wbc"
        elif "meg" in vtype_raw:
            vtype = "meg"
        else:
            continue

        t2 = dict(t)
        sid = _to_int(t2.get("sort_id"))
        t2["sort_id"] = sid if sid is not None else (i + 1)
        tasks_by_tile[(r, c)][vtype].append(t2)
print(f"[INFO] 共检测到 {len(tasks_by_tile)} 个被 WBC 任务涉及到的 40× tile。")


# 为仅这些 wbc_tiles 补充“原始 WBC 点”（org_wbc）
for (row, col) in wbc_tiles:
    node = node_by_rc.get((row, col))
    if node is None:
        continue
    pts = _to_xy_array(node["data"].get("wbc_center_pt", []))
    if pts.size:
        tasks_by_tile[(row, col)]["org_wbc"] = [(int(round(x)), int(round(y))) for x, y in pts]
print(f"[INFO] 共检测到 {len(tasks_by_tile)} 个被 WBC 任务涉及到的 40× tile。")


# ---------- 绘制每张“WBC 涉及到的” 40× 图 ----------
for (row, col), groups in tasks_by_tile.items():
    img_path = f"{IMAGES_DIR_40X}/Pos[{col}][{row}].jpg"
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] 载入失败，跳过: {img_path}")
        continue

    # 四宫格 typeid（从对应 node 里取）
    node = node_by_rc.get((row, col))
    if node is not None:
        area_info = node["data"].get("area_score_info", [])
        for i, item in enumerate(area_info):
            try:
                typeid_ = (item.get('typeid') if isinstance(item, dict)
                           else (item[5] if isinstance(item, (list, tuple)) and len(item) >= 6 else None))
                if typeid_ is None:
                    continue
                if i == 0:
                    tx, ty = 0, 0
                elif i == 1:
                    tx, ty = 1224, 0
                elif i == 2:
                    tx, ty = 0, 1024
                else:
                    tx, ty = 1224, 1024
                cv2.putText(img, f"{int(typeid_)}", (tx, ty + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            except Exception:
                pass

    # 原始 WBC（绿色空心小圆）
    for (px, py) in groups.get("org_wbc", []):
        cv2.circle(img, (int(px), int(py)), 4, (0, 255, 0), 2)
        # cv2.putText(img, f"{int(px), int(py)}", (px-48, py + 12),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # WBC 任务（红框+红点）
    for idx, t in enumerate(_sorted_tasks(groups.get("wbc", [])), start=1):
        vx = int(_to_float(t.get("view_pos_x")))
        vy = int(_to_float(t.get("view_pos_y")))
        vw = int(_to_float(t.get("view_width")))
        vh = int(_to_float(t.get("view_height")))
        cv2.rectangle(img, (vx, vy), (vx + vw, vy + vh), (0, 0, 255), 2)
        order = _to_int(t.get("sort_id"), idx)
        cv2.putText(img, f"{order}", (vx, vy + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for cell in t.get("cell_list", []):
            cx = int(_to_float(cell.get("cell_x")))
            cy = int(_to_float(cell.get("cell_y")))
            cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)

    # MEG 任务（绿框+绿点）——只画在这些 WBC 涉及到的 tile 上（若存在）
    for idx, t in enumerate(_sorted_tasks(groups.get("meg", [])), start=1):
        vx = int(_to_float(t.get("view_pos_x")))
        vy = int(_to_float(t.get("view_pos_y")))
        vw = int(_to_float(t.get("view_width")))
        vh = int(_to_float(t.get("view_height")))
        cv2.rectangle(img, (vx, vy), (vx + vw, vy + vh), (0, 255, 0), 2)
        order = _to_int(t.get("sort_id"), idx)
        cv2.putText(img, f"{order}", (vx, vy + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for cell in t.get("cell_list", []):
            cx = int(_to_float(cell.get("cell_x")))
            cy = int(_to_float(cell.get("cell_y")))
            cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)

    out_path = os.path.join(OUT_DIR, f"Pos[{col}][{row}]_40x_annot.jpg")
    ok = cv2.imwrite(out_path, img)
    if not ok:
        print(f"[WARN] 保存失败: {out_path}")
    else:
        print(f"[OK] 保存: {out_path}")
