import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import os
from collections import OrderedDict, defaultdict   


# ================== 读 JSON ==================
with open("/home/ubuntu/VScodeProjects/megLoc_heatmap/data/data2025063002.json", 'r') as f:
    data = json.load(f)


# ===== 读取 x100 百倍任务（首个 WBC 初始框、最后一个或若干个 MEG 框）=====
TASK_JSON = "/home/ubuntu/VScodeProjects/megLoc_heatmap/output/data2025063002/x100_task_list.json"  
with open(TASK_JSON, "r") as f:
    x100_tasks = json.load(f)   


# 把任务按 (row_index, col_index) 建索引，便于在每个瓦片上绘制
# 结构： tasks_by_tile[(row,col)] = {"wbc":[...], "meg":[...]}
tasks_by_tile = defaultdict(lambda: {"wbc": [], "meg": []})
for task in [x100_tasks[0], x100_tasks[-1]]:  # 支持有核任务初始框和巨核任务
    for i, t in enumerate(task):

        r = int(t.get("row_index"))
        c = int(t.get("col_index"))
        vtype = str(t.get("view_type", "")).lower()
        tasks_by_tile[(r, c)][vtype].append(t)


# ================== 图块列表 ==================
images_info = []
TILE_W, TILE_H = 2448, 2048
image_list = data['nodes']
for one in image_list:
    row = one['row']
    col = one['col']
    images_info.append({
        'path': f"/home/ubuntu/VScodeProjects/megLoc_heatmap/image/data2025063002/Images/Pos[{col}][{row}].jpg",
        'top_left': (one['data']['position'][0], one['data']['position'][1]),
        'node': (row, col),
        "residual": one['data'].get('residual', 0),
        "component_id": one['data']['component_id'],
        'wbc_center_pt': one['data'].get('wbc_center_pt', []),
        'meg_center_pt': one['data'].get('meg_center_pt', []),   # [x,y,w,h]
        'area_score_info': one['data'].get('area_score_info', []),
    })

max_component_id = max(info['component_id'] for info in images_info) if images_info else 0
cmap = plt.get_cmap('tab10')
COLORS = [cmap(i % cmap.N) for i in range(max_component_id + 1)]
COLORS = [(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in COLORS]

# 平移到从(0,0)开始
min_x = min(info['top_left'][0] for info in images_info)
min_y = min(info['top_left'][1] for info in images_info)
for info in images_info:
    x, y = info['top_left']
    info['top_left'] = (x - min_x, y - min_y)

# ================== 字节数受限 LRU ==================
class ByteLRU(OrderedDict):
    def __init__(self, max_bytes):
        super().__init__()
        self.max_bytes = int(max_bytes)
        self.bytes = 0

    def _size(self, v):
        # numpy array 有 nbytes，普通对象用 sys.getsizeof
        return getattr(v, "nbytes", None) or sys.getsizeof(v)

    def __setitem__(self, key, value):
        sz = self._size(value)
        if key in self:
            self.bytes -= self._size(super().__getitem__(key))
            super().__delitem__(key)
        super().__setitem__(key, value)
        self.bytes += sz
        # 淘汰到不超标
        while self.bytes > self.max_bytes and len(self) > 0:
            _, old = self.popitem(last=False)
            self.bytes -= self._size(old)

    def clear(self):
        super().clear()
        self.bytes = 0

# 缓存容量（按机器内存酌情调整）
# 原图缓存 512MB，缩放缓存 1GB
image_cache  = ByteLRU(max_bytes=512 * 1024**2)
scaled_cache = ByteLRU(max_bytes=1 * 1024**3)

# 线程池
executor = ThreadPoolExecutor(max_workers=6)

# ================== 异步加载原图 ==================
def load_image_async(path):
    if path in image_cache:
        return
    img = cv2.imread(path)  # BGR
    if img is not None:
        image_cache[path] = img

# ================== 获取缩放图（只缓存缩小图） ==================
_last_zoom_for_cache = 1.0

def on_zoom_changed(new_zoom, threshold=0.15):
    """缩放变化较大时清空缩放缓存，避免不同层级缩放图长期占用内存。"""
    global _last_zoom_for_cache
    if abs(new_zoom - _last_zoom_for_cache) / max(_last_zoom_for_cache, 1e-6) > threshold:
        scaled_cache.clear()
    _last_zoom_for_cache = new_zoom







def get_scaled_image(path, zoom, info):
    """
    - zoom <= 1.0: 允许缓存（按两位小数量化）
    - zoom > 1.0 : 不缓存（避免放大图撑爆内存）
    """
    if path not in image_cache:
        # 异步加载；本次返回 None，下一帧再渲染
        executor.submit(load_image_async, path)
        return None

    img = image_cache[path]
    # 量化 zoom 作为 cache key（两位小数），减少 key 数
    zq = round(float(zoom), 2)

    use_cache = (zq <= 1.0)
    cache_key = (path, zq)

    if use_cache and cache_key in scaled_cache:
        scaled_img = scaled_cache[cache_key].copy()
    else:
        # 选择合适的插值：缩小->INTER_AREA；放大->INTER_LINEAR
        interp = cv2.INTER_AREA if zq < 1.0 else cv2.INTER_LINEAR
        h, w = int(img.shape[0] * zq), int(img.shape[1] * zq)
        if h <= 0 or w <= 0:
            return None
        scaled_img = cv2.resize(img, (w, h), interpolation=interp)
        if use_cache:
            scaled_cache[cache_key] = scaled_img
            scaled_img = scaled_img.copy()  # 避免在缓存对象上直接画

    # ====== 叠加可视化元素（在 scaled_img 上作画）======
    # WBC 点
    if info.get('wbc_center_pt'):
        for x0, y0 in info['wbc_center_pt']:
            x = int(x0 * zq)
            y = int(y0 * zq)
            cv2.circle(scaled_img, (x, y), max(1, int(5 * zq)), (0, 255, 255), -1)

    # MEG 矩形（这里保留你原来的 *4 逻辑）
    if info.get('meg_center_pt'):
        for x0, y0, w0, h0 in info['meg_center_pt']:
            x = int(x0 * zq * 4)
            y = int(y0 * zq * 4)
            w = int(w0 * zq * 4)
            h = int(h0 * zq * 4)
            # cv2.rectangle(scaled_img, (x, y), (x + w, y + h), (127, 127, 255), 1)
            cv2.circle(scaled_img, (x, y), max(1, int(5 * zq)), (127, 127, 255), -1)

    # 四宫格 typeid 标注
    for i, (ax, ay, aw, ah, rating, typeid) in enumerate(info.get('area_score_info', [])):
        # 你的原逻辑：四块固定写死位置
        if i == 0:
            tx, ty = 0, 0
        elif i == 1:
            tx, ty = int(1224 * zq), 0
        elif i == 2:
            tx, ty = 0, int(1024 * zq)
        else:
            tx, ty = int(1224 * zq), int(1024 * zq)
        cv2.putText(scaled_img, f"{int(typeid)}", (tx, ty + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # === 把 x100 任务画到该 tile 上（按 sort_id 排序并标注顺序号）===
    row, col = info['node']               # 本瓦片的 行/列 索引
    tile_tasks = tasks_by_tile.get((row, col))

    def _as_int(x, default=None):
        try:
            return int(x)
        except (TypeError, ValueError):
            return default

    def _sorted_tasks(tasks):
        # 先按 sort_id 排（没有的放最后），再用 (y, x) 做稳定次序
        BIG = 10**9
        return sorted(
            tasks,
            key=lambda t: (
                _as_int(t.get('sort_id'), BIG),
                float(t.get('view_pos_y', 0)),
                float(t.get('view_pos_x', 0)),
            )
        )

    def _order_label(t, fallback_idx):
        # 有 sort_id 用 sort_id；没有就用枚举序
        sid = _as_int(t.get('sort_id'))
        return sid if sid is not None else fallback_idx

    if tile_tasks:
        # 便于观感：字号/线宽随缩放变化
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, 0.6 * zq)
        thick = max(1, int(2 * zq))

        print(f"Drawing tasks on tile ({row},{col}): "
              f"WBC={len(tile_tasks.get('wbc', []))}, MEG={len(tile_tasks.get('meg', []))}")

        # ---------- WBC（红色）----------
        wbc_list = _sorted_tasks(tile_tasks.get("wbc", []))
        for idx, t in enumerate(wbc_list, start=1):
            vx = int(t.get("view_pos_x", 0) * zq)
            vy = int(t.get("view_pos_y", 0) * zq)
            vw = int(t.get("view_width", 0)  * zq)
            vh = int(t.get("view_height", 0) * zq)

            cv2.rectangle(scaled_img, (vx, vy), (vx + vw, vy + vh), (0, 0, 255), 2)

            # 顺序号：优先用 sort_id；没有就用 idx
            order = _order_label(t, idx)
            cv2.putText(scaled_img, f"{order}", (vx, vy + int(0 * zq)),
                        font, font_scale, (0, 0, 255), thick)

            # 细胞点
            for cell in t.get("cell_list", []):
                cx = int(cell.get("cell_x", 0) * zq)
                cy = int(cell.get("cell_y", 0) * zq)
                cv2.circle(scaled_img, (cx, cy), max(1, int(3 * zq)), (0, 0, 255), -1)

        # ---------- MEG（绿色）----------
        meg_list = _sorted_tasks(tile_tasks.get("meg", []))
        for idx, t in enumerate(meg_list, start=1):
            vx = int(t.get("view_pos_x", 0) * zq)
            vy = int(t.get("view_pos_y", 0) * zq)
            vw = int(t.get("view_width", 0)  * zq)
            vh = int(t.get("view_height", 0) * zq)

            cv2.rectangle(scaled_img, (vx, vy), (vx + vw, vy + vh), (0, 255, 0), 2)

            # 顺序号：优先用 sort_id；没有就用 idx
            order = _order_label(t, idx)
            cv2.putText(scaled_img, f"{order}", (vx, vy + int(0 * zq)),
                        font, font_scale, (0, 255, 0), thick)

            for cell in t.get("cell_list", []):
                cx = int(cell.get("cell_x", 0) * zq)
                cy = int(cell.get("cell_y", 0) * zq)
                cv2.circle(scaled_img, (cx, cy), max(1, int(3 * zq)), (0, 255, 0), -1)

    return scaled_img

# ================== 视口与交互 ==================
VIEW_W, VIEW_H = 1200, 800
zoom = 1.0
offset_x, offset_y = 0, 0
dragging = False
start_x = start_y = 0

def get_visible_tiles(offset_x, offset_y, zoom):
    visible = []
    for info in images_info:
        x, y = info['top_left']
        xz, yz = int(x * zoom), int(y * zoom)
        w, h = int(TILE_W * zoom), int(TILE_H * zoom)
        if (xz + w < offset_x) or (xz > offset_x + VIEW_W) or (yz + h < offset_y) or (yz > offset_y + VIEW_H):
            continue
        visible.append(info)
    return visible

def update_view():
    view = np.zeros((VIEW_H, VIEW_W, 3), dtype=np.uint8)
    tiles = get_visible_tiles(offset_x, offset_y, zoom)

    for info in tiles:
        x, y = info['top_left']
        row, col = info['node']
        xz, yz = int(x * zoom), int(y * zoom)

        img = get_scaled_image(info['path'], zoom, info)
        if img is None:
            continue

        x1 = xz - offset_x
        y1 = yz - offset_y
        x2 = min(VIEW_W, x1 + img.shape[1])
        y2 = min(VIEW_H, y1 + img.shape[0])

        if 0 <= x1 < VIEW_W and 0 <= y1 < VIEW_H:
            view[y1:y2, x1:x2] = img[:y2 - y1, :x2 - x1]
            # 边框 + 节点标注
            cv2.rectangle(view, (x1, y1), (x2-1, y2-1), (125, 125, 125), 1)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            text = f"({row},{col})"
            color = COLORS[info['component_id'] % len(COLORS)]
            cv2.putText(view, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    cv2.imshow("Dynamic Mosaic Viewer", view)

def mouse_callback(event, x, y, flags, param):
    global offset_x, offset_y, dragging, start_x, start_y, zoom

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        dx = x - start_x
        dy = y - start_y
        offset_x = max(0, offset_x - dx)
        offset_y = max(0, offset_y - dy)
        start_x, start_y = x, y
        update_view()

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

    elif event == cv2.EVENT_MOUSEWHEEL:
        old_zoom = zoom
        if flags > 0:
            zoom = min(zoom * 1.1, 5.0)
        else:
            zoom = max(zoom * 0.9, 0.1)
        on_zoom_changed(zoom)

        # 保持光标处为视觉中心
        mx, my = x + offset_x, y + offset_y
        offset_x = int(mx * zoom / old_zoom) - x
        offset_y = int(my * zoom / old_zoom) - y
        update_view()

def initialize_fullscreen_window(window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ================== 运行 ==================
window_name = "Dynamic Mosaic Viewer"
initialize_fullscreen_window(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

update_view()

while True:
    rect = cv2.getWindowImageRect(window_name)
    VIEW_W, VIEW_H = rect[2], rect[3]

    key = cv2.waitKey(20)
    if key == 27:  # ESC
        break
    elif key == ord('+'):
        old_zoom = zoom
        zoom = min(zoom * 1.1, 5.0)
        on_zoom_changed(zoom)
        offset_x = int((offset_x + VIEW_W // 2) * zoom / old_zoom) - VIEW_W // 2
        offset_y = int((offset_y + VIEW_H // 2) * zoom / old_zoom) - VIEW_H // 2
        update_view()
    elif key == ord('-'):
        old_zoom = zoom
        zoom = max(zoom * 0.9, 0.1)
        on_zoom_changed(zoom)
        offset_x = int((offset_x + VIEW_W // 2) * zoom / old_zoom) - VIEW_W // 2
        offset_y = int((offset_y + VIEW_H // 2) * zoom / old_zoom) - VIEW_H // 2
        update_view()

cv2.destroyAllWindows()
