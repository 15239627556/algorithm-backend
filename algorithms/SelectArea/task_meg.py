from __future__ import annotations
import numpy as np
from typing import List, Optional
from .config import BM40Config
from .data_structure import TaskOutput, CellOutput
from .setcover import solve, SetCoverSolverParameter


def generate_meg_view_tasks(
    meg_cell_bounds: np.ndarray,            # 过滤后的有效 MEG 细胞 (N, 4) [xmin, ymin, xmax, ymax]
    config: BM40Config,
    wbc_rects: List[List[int]],          # 外部传入的 WBC 视野列表 [[X, Y, W, H], ...]
    params: Optional[SetCoverSolverParameter] = None,
) -> List[TaskOutput]:
    """
    生成巨核细胞视野任务：
    1）对巨核细胞做 set-cover 得到候选视野；
    2）根据传入的 wbc_rects 计算一个 WBC 中心点；
    3）按视野中心到 WBC 中心的距离排序；
    4）按 config.target_cell_num_MEG 截断视野数量；
    5）返回 TaskOutput 列表（view_type 可用 'MEG' 标记）。
    """
    if meg_cell_bounds.size == 0:
        print("警告：无有效巨核细胞，无法生成 MEG 任务。")
        return []

    # 1. 计算巨核细胞中心点
    centers = 0.5 * (meg_cell_bounds[:, 0:2] + meg_cell_bounds[:, 2:4])


    # 2. 计算 WBC 中心点（用所有 WBC 视野中心的平均值）
    if not wbc_rects:
        wbc_center = None
        print("警告：无 WBC 细胞框，无法计算 WBC 中心点。")
    else:
        # wbc_rects: [[x, y, w, h], ...]
        wbc_arr = np.asarray(wbc_rects, dtype=np.float32)  # shape (M, 4)
        # 中心 = 左上角 + 宽高的一半
        wbc_centers = np.empty((wbc_arr.shape[0], 2), dtype=np.float32)
        wbc_centers[:, 0] = wbc_arr[:, 0] + wbc_arr[:, 2] * 0.5  # cx
        wbc_centers[:, 1] = wbc_arr[:, 1] + wbc_arr[:, 3] * 0.5  # cy
        wbc_center = wbc_centers.mean(axis=0)  # [cx, cy]

    # 3. 准备 set-cover 输入
    pad = config.setcover_pad
    x_min_all, y_min_all = meg_cell_bounds[:, 0:2].min(axis=0) - pad
    x_max_all, y_max_all = meg_cell_bounds[:, 2:4].max(axis=0) + pad
    bounding_rect = np.array(
        [
            x_min_all,
            y_min_all,
            x_max_all - x_min_all + 1,
            y_max_all - y_min_all + 1,
        ],
        dtype=np.int32,
    )

    params = params or SetCoverSolverParameter(
        rect_width=config.x100_rect_width,
        rect_height=config.x100_rect_height,
    )

    # 4. 求解巨核视野分布（候选视野）
    rects_meg = solve(centers, bounding_rect, params)  # List[(rx, ry, rw, rh)]

    if len(rects_meg) == 0:
        print("警告：Set-cover 未生成任何 MEG 视野。")
        return []

    # 5. 若有 WBC 中心，则按距离排序（使用 NumPy 向量化）
    if wbc_center is not None:
        cx, cy = float(wbc_center[0]), float(wbc_center[1])

        # rects_meg: 形状 (K, 4) -> 中心点 (K, 2)
        rects_arr = np.asarray(rects_meg, dtype=np.float32)  # [rx, ry, rw, rh]
        rect_centers_x = rects_arr[:, 0] + rects_arr[:, 2] * 0.5
        rect_centers_y = rects_arr[:, 1] + rects_arr[:, 3] * 0.5

        # 与 WBC 中心的平方距离
        d2 = (rect_centers_x - cx) ** 2 + (rect_centers_y - cy) ** 2
        order = np.argsort(d2)

        # 根据排序索引重排 rects_meg
        rects_meg = rects_arr[order].tolist()


    # 6. 从最近的视野开始累加巨核细胞数，直到 target_cell_num_MEG
    used = np.zeros(len(meg_cell_bounds), dtype=bool)
    # selected_rects: [(rx, ry, rw, rh, matched_idx_new), ...]
    # matched_idx_new 表示该视野新增覆盖到的巨核细胞索引（避免后续重复扫全量 centers）
    selected_rects: List[tuple] = []
    total_cells = 0
    target_num = int(getattr(config, "target_cell_num_MEG", 0) or 0)
    meg_cell_num = int(meg_cell_bounds.shape[0])

    # 保护逻辑：用户需求超过实际巨核细胞数量时，直接把目标上限钳到实际数量
    if target_num > meg_cell_num:
        print(
            f"警告：target_cell_num_MEG({target_num}) 大于实际巨核细胞数量({meg_cell_num})，"
            f"将按实际巨核细胞数量生成 MEG 视野（等价于返回全部巨核细胞）。"
        )
        # 为确保覆盖所有巨核细胞：直接返回所有候选视野，并缓存每个视野的命中索引
        for rx, ry, rw, rh in rects_meg:
            in_x = (centers[:, 0] >= rx) & (centers[:, 0] < rx + rw)
            in_y = (centers[:, 1] >= ry) & (centers[:, 1] < ry + rh)
            mask = in_x & in_y
            matched_idx = np.where(mask)[0]
            selected_rects.append((rx, ry, rw, rh, matched_idx))
    # 保护逻辑：用户需求为0或负数时，返回空的 meg_tasks。
    elif target_num <= 0:
        print("警告：target_cell_num_MEG <= 0，将返回空的 meg_tasks。")
        return []

    # 正常逻辑：用户需求为正数时，按贪心算法选择视野。
    else:
        for rx, ry, rw, rh in rects_meg:
            if total_cells >= target_num:
                break

            in_x = (centers[:, 0] >= rx) & (centers[:, 0] < rx + rw)
            in_y = (centers[:, 1] >= ry) & (centers[:, 1] < ry + rh)
            mask = in_x & in_y & (~used)

            if not np.any(mask):
                continue

            num_new = int(mask.sum())
            matched_idx_new = np.where(mask)[0]
            selected_rects.append((rx, ry, rw, rh, matched_idx_new))
            used[matched_idx_new] = True
            total_cells += num_new

    if len(selected_rects) == 0:
        print("警告：在候选视野中未能覆盖任何新的巨核细胞。")
        return []

    # 7. 构造 TaskOutput 列表（不再区分 Initial/Extra，统一 region_name='MEG'）
    meg_tasks: List[TaskOutput] = []
    global_counter = 1

    for rx, ry, rw, rh, matched_idx_new in selected_rects:
        current_cell_outputs: List[CellOutput] = []
        if matched_idx_new.size > 0:
            for idx in matched_idx_new:
                current_cell_outputs.append(
                    CellOutput(
                        cell_xmin=int(round(meg_cell_bounds[idx, 0])),
                        cell_ymin=int(round(meg_cell_bounds[idx, 1])),
                        cell_xmax=int(round(meg_cell_bounds[idx, 2])),
                        cell_ymax=int(round(meg_cell_bounds[idx, 3])),
                    )
                )

        task_obj = TaskOutput(
            task_index=global_counter,
            view_type=config.View_type,             
            smear_type=config.Smear_type,
            view_xmin=int(round(rx)),
            view_ymin=int(round(ry)),
            view_xmax=int(round(rx + rw)),
            view_ymax=int(round(ry + rh)),
            region_name=config.View_type,          
            cell_list=current_cell_outputs,
        )
        meg_tasks.append(task_obj)
        global_counter += 1

    return meg_tasks