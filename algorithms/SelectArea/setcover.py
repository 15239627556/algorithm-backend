import pyscipopt
import numpy as np



class SetCoverSolverParameter(object):

    def __init__(
        self,
        rect_width = 384,
        rect_height = 283,
        max_num_points_to_split = 400,
        img_split_overlap_w_factor = 1,
        stride_ratio = 0.25,
        min_image_height = 800,
        min_image_width = 800,
        refine_rects = True,
        solve_overlapping = True
    ):
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.max_num_points_to_split = max_num_points_to_split
        self.img_split_overlap_w_factor = img_split_overlap_w_factor
        self.stride_ratio = stride_ratio
        self.min_image_height = min_image_height
        self.min_image_width = min_image_width
        self.refine_rects = refine_rects
        self.solve_overlapping = solve_overlapping

def assert_validity(aff):
    for i in range(aff.shape[0]): assert aff[i].any(), "Selected rect {} does not cover any points".format(i)
    for j in range(aff.shape[1]): assert aff[:, j].any(), "Point {} is not covered by any rect".format(j)

def rect_contains(rect, points):
    # return the indices of the points that are inside the rectangle
    x1, y1, w, h = rect
    x2, y2 = x1 + w, y1 + h
    return np.where((points[:, 0] >= x1) & (points[:, 0] <= x2) &
                    (points[:, 1] >= y1) & (points[:, 1] <= y2))[0]

def affinity_matrix(rects, points):
    xmin = rects[:, 0]
    ymin = rects[:, 1]
    xmax = rects[:, 0] + rects[:, 2]
    ymax = rects[:, 1] + rects[:, 3]
    px = points[:, 0][:, None]
    py = points[:, 1][:, None]
    aff = (px >= xmin) & (px <= xmax) & (py >= ymin) & (py <= ymax)
    return aff.T

def solve_set_cover(aff):
    n_rects, n_points = aff.shape
    model = pyscipopt.Model('Set Cover')
    model.setParam('display/verblevel', 0)
    vars = [model.addVar(vtype="B", name=f"rect_{i}") for i in range(n_rects)]
    for j in range(n_points):
        covering_rects = np.where(aff[:, j])[0]
        assert len(covering_rects) > 0, "No covering rects for point {}".format(j)
        model.addCons(pyscipopt.quicksum(vars[i] for i in covering_rects) >= 1,
                      name=f"cover_point_{j}")
    model.setObjective(pyscipopt.quicksum(vars), "minimize")
    model.optimize()
    selected = [j for j in range(n_rects) if model.getVal(vars[j]) > 0.5]
    return selected

def generate_candidate_rects(image_r, points, params: SetCoverSolverParameter):
    # generate sliding windows
    stride_ratio = _auto_stride_ratio(image_r, params, target=20000)  # ★ 新增
    sx = int(max(1, params.rect_width  * stride_ratio))
    sy = int(max(1, params.rect_height * stride_ratio))
    window_x = np.arange(image_r[0], image_r[0] + image_r[2], sx)
    window_y = np.arange(image_r[1], image_r[1] + image_r[3], sy)

    # window_x = np.arange(image_r[0], image_r[0] + image_r[2],
    #                      int(params.rect_width * params.stride_ratio))
    # window_y = np.arange(image_r[1], image_r[1] + image_r[3],
    #                      int(params.rect_height * params.stride_ratio))
    window_xx, window_yy = np.meshgrid(window_x, window_y, indexing='ij')
    window = np.stack([window_xx, window_yy], axis=-1).reshape(-1, 2)
    window_shape = np.array([params.rect_width, params.rect_height])
    windows = np.hstack((window, window_shape[None, :].repeat(window.shape[0], axis=0)))
    aff = affinity_matrix(windows, points)
    empty_rect_index = np.where(~aff.any(axis=1))[0]
    windows = np.delete(windows, empty_rect_index, axis=0)
    aff = np.delete(aff, empty_rect_index, axis=0)
    return windows, aff

def pairwise_iou(rects):
    x1 = rects[:, 0]
    y1 = rects[:, 1]
    x2 = x1 + rects[:, 2]
    y2 = y1 + rects[:, 3]
    areas = rects[:, 2] * rects[:, 3]
    xx1 = np.maximum(x1[:, None], x1[None, :])
    yy1 = np.maximum(y1[:, None], y1[None, :])
    xx2 = np.minimum(x2[:, None], x2[None, :])
    yy2 = np.minimum(y2[:, None], y2[None, :])
    inter_w = np.maximum(0, xx2 - xx1)
    inter_h = np.maximum(0, yy2 - yy1)
    inter_area = inter_w * inter_h
    union_area = areas[:, None] + areas[None, :] - inter_area
    iou = inter_area / np.clip(union_area, a_min=1e-9, a_max=None)
    return iou

def span_area(points):
    spanx = np.max(points[:, 0]) - np.min(points[:, 0])
    spany = np.max(points[:, 1]) - np.min(points[:, 1])
    return max(spanx, spany)

def refine_rects(rects, points, params):
    aff = affinity_matrix(rects, points)
    iou = pairwise_iou(rects)
    iou = np.triu(iou, k=1)
    for i, j in zip(*np.where(iou > 0.01)):
        mutual = aff[i] & aff[j]
        if ~np.any(mutual): continue
        r1_span_with_mutual = span_area(points[aff[i]])
        r2_span_with_mutual = span_area(points[aff[j]])
        r1_span_without_mutual = span_area(points[aff[i] & ~aff[j]])
        r2_span_without_mutual = span_area(points[aff[j] & ~aff[i]])
        r1_diff = r1_span_with_mutual - r1_span_without_mutual
        r2_diff = r2_span_with_mutual - r2_span_without_mutual
        assert r1_diff >= 0 and r2_diff >= 0, "Span area should not decrease"
        if (r2_diff > r1_diff):
            aff[j, mutual] = False
        else:
            aff[i, mutual] = False
    for i in range(aff.shape[0]):
        points_in_rect = points[aff[i]]
        xmin, xmax = np.min(points_in_rect[:, 0]), np.max(points_in_rect[:, 0])
        ymin, ymax = np.min(points_in_rect[:, 1]), np.max(points_in_rect[:, 1])
        rects[i][0] = int((xmax + xmin) / 2 - params.rect_width / 2)
        rects[i][1] = int((ymax + ymin) / 2 - params.rect_height / 2)
    return rects

def _auto_stride_ratio(image_r, params, target=20000):
    # 计算当前窗口数，超标则把 stride_ratio 线性放大到 <=1.0
    cur = _estimate_windows(image_r, params, params.stride_ratio)
    if cur <= target:
        return params.stride_ratio
    scale = (cur / target) ** 0.5
    return min(1.0, params.stride_ratio * scale)

# 经验阈值：限制 base case 下的候选窗口数量，防止在大空域里密集滑窗拖慢
MAX_WINDOWS = 50000

def _tighten_to_points(image_r: np.ndarray, pts: np.ndarray, params: SetCoverSolverParameter,
                       pad_frac: float = 0.10) -> np.ndarray:
    """
    将 image_r 收紧到当前节点内点的包围盒，并按视野尺寸加少量 padding。
    这样候选窗口规模与点的真实分布成正比，避免在大空域滑窗。
    """
    if pts.size == 0:
        return image_r.astype(np.int32, copy=False)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    pad_w = int(params.rect_width  * pad_frac)
    pad_h = int(params.rect_height * pad_frac)
    x = int(xmin) - pad_w
    y = int(ymin) - pad_h
    w = int(xmax - xmin) + 2 * pad_w
    h = int(ymax - ymin) + 2 * pad_h
    # 至少保持一个视野大小，避免被裁成 0
    w = max(w, params.rect_width)
    h = max(h, params.rect_height)
    return np.array([x, y, w, h], dtype=np.int32)



def _estimate_windows(image_r: np.ndarray, params: SetCoverSolverParameter,
                      stride_ratio: float) -> int:
    """
    估算在 image_r 上以 stride_ratio 滑窗时的候选窗口数量（与 generate_candidate_rects 对齐的粗算）。
    """
    W, H = int(image_r[2]), int(image_r[3])
    sx = max(1, int(params.rect_width  * stride_ratio))
    sy = max(1, int(params.rect_height * stride_ratio))
    nx = max(1, (W + sx - 1) // sx)  # ceil(W/sx)
    ny = max(1, (H + sy - 1) // sy)  # ceil(H/sy)
    return nx * ny


def solve(points: np.ndarray, image_r: np.ndarray, params: SetCoverSolverParameter, rec_depth: int = 0):
    """
    修改要点：
    1) 进入每个递归节点先把 image_r 收紧到点的包围盒（_tighten_to_points）。
    2) base case 不仅看“点数/最小尺寸”，还要看“候选窗口规模”是否 <= MAX_WINDOWS。
    3) 若窗口规模过大，则继续分治，直到规模可控再进入 base case。
    """
    # 仅保留当前 image_r 覆盖到的点
    idx = rect_contains(image_r, points)
    points = points[idx]
    # 若无点，直接返回空
    if points.size == 0:
        return np.empty((0, 4), dtype=np.int32)

    # 将 image_r 收紧到点包围盒（加少量 padding）
    image_r = _tighten_to_points(image_r, points, params)

    # 估算当前节点的候选窗口数
    nwin_est = _estimate_windows(image_r, params, params.stride_ratio)

    # base case 条件
    base_ok = (len(points) <= params.max_num_points_to_split or
               image_r[2] < params.min_image_width or
               image_r[3] < params.min_image_height)

    # 同时要求候选窗口不超过阈值，才真正进入 base case
    if base_ok and nwin_est <= MAX_WINDOWS:
        rects, aff = generate_candidate_rects(image_r, points, params)
        selected = solve_set_cover(aff)
        rects = rects[selected]
        # 为顶层 refine/校正准备一次最新 aff
        aff = affinity_matrix(rects, points)
        if (rec_depth == 0 and params.refine_rects):
            rects = refine_rects(rects, points, params)
        return rects.astype(np.int32, copy=False)

    # 仍然太大：分治
    pad_value = int(max(params.rect_width, params.rect_height) * 0.1)
    factor = params.img_split_overlap_w_factor

    if image_r[2] >= image_r[3]:
        # 横向切分
        overlap_w = int(params.rect_width * factor)
        mid = int(image_r[0] + image_r[2] / 2)
        box_width = int(image_r[2] / 2 + overlap_w / 2)

        bound1 = np.array([
            image_r[0] - pad_value,
            image_r[1] - pad_value,
            box_width + 2 * pad_value,
            image_r[3] + 2 * pad_value
        ], dtype=np.int32)

        bound2 = np.array([
            mid - overlap_w // 2 - pad_value,
            image_r[1] - pad_value,
            box_width + 2 * pad_value,
            image_r[3] + 2 * pad_value
        ], dtype=np.int32)
    else:
        # 纵向切分
        overlap_h = int(params.rect_height * factor)
        mid = int(image_r[1] + image_r[3] / 2)
        box_height = int(image_r[3] / 2 + overlap_h / 2)

        bound1 = np.array([
            image_r[0] - pad_value,
            image_r[1] - pad_value,
            image_r[2] + 2 * pad_value,
            box_height + 2 * pad_value
        ], dtype=np.int32)

        bound2 = np.array([
            image_r[0] - pad_value,
            mid - overlap_h // 2 - pad_value,
            image_r[2] + 2 * pad_value,
            box_height + 2 * pad_value
        ], dtype=np.int32)

    # 递归求解两半
    rects1 = solve(points, bound1, params, rec_depth + 1)
    rects2 = solve(points, bound2, params, rec_depth + 1)

    # 合并
    if rects1.size == 0:
        rects = rects2
    elif rects2.size == 0:
        rects = rects1
    else:
        rects = np.vstack((rects1, rects2))

    # 顶层再做一次 set cover（在合并候选上），以获得更紧的解
    if rec_depth == 0:
        aff = affinity_matrix(rects, points)
        selected = solve_set_cover(aff)
        rects = rects[selected]
        if params.refine_rects:
            rects = refine_rects(rects, points, params)

    return rects.astype(np.int32, copy=False)
