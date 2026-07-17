# main_meg.py
from __future__ import annotations

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from project.smear_project import SmearProject

from dataclasses import dataclass
from typing import Any, List

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

from .data_structure import TaskOutput, CellOutput
from .config import BM40Config
from .pipeline_meg import MegSamplingPipeline


# ===================== 可视化配置 =====================
@dataclass(frozen=True)
class VizConfigMeg:
    # BM 项目 JSON（SmearProject.load_json），含 40x layers.tiles 与 tile.cells
    json_path: str = "/home/ubuntu/VScodeProjects/项目json数据/巨核拍摄顺序不正确/202607170001/4f1171e158ed4d5aa9470047d933a30e.json"
    out_dir: str = "/home/ubuntu/VScodeProjects/algorithm-backend/algorithms/SelectArea/output"
    # MEG 结果输出文件名
    meg_result_json: str = "results_meg.json"


def collect_wbc_view_rects_from_results(
    results_json_path: Path,
) -> List[List[int]]:
    """
    从 WBC 输出 results.json 中读取视野边界，并转换为 [x, y, w, h]。
    """
    with open(results_json_path, "r", encoding="utf-8") as f:
        task_results = json.load(f)

    wbc_rects: List[List[int]] = []
    for task in task_results:
        view_xmin = int(task["view_xmin"])
        view_ymin = int(task["view_ymin"])
        view_xmax = int(task["view_xmax"])
        view_ymax = int(task["view_ymax"])
        wbc_rects.append([
            view_xmin,
            view_ymin,
            view_xmax - view_xmin,
            view_ymax - view_ymin,
        ])
    return wbc_rects


# ===================== 可视化：MEG 任务 =====================
def visualize_meg_results(
    tasks: List[TaskOutput],
    grid_info: Any,
    save_path_base: Path,
    wbc_rects: List[List[int]]
) -> None:
    """
    可视化 MEG 结果：fig4_meg_tasks.png（视野框 + 顺序编号/路径）
    """
    if not tasks:
        print("[WARNING][MEG] 无 MEG 采样任务，跳过可视化生成。")
        return

    save_path_base.mkdir(parents=True, exist_ok=True)

    meg_tasks = [t for t in tasks if getattr(t, "view_type", "") == "MEG"]
    if not meg_tasks:
        meg_tasks = list(tasks)

    def _draw_order_figure(
        ax,
        *,
        draw_wbc: bool,
        label_fontsize: float,
        box_lw: float,
    ) -> None:
        ax.set_facecolor("black")
        task_xs: List[float] = []
        task_ys: List[float] = []
        centers_x: List[float] = []
        centers_y: List[float] = []

        for task in meg_tasks:
            color = "yellow"
            task_xs.extend([task.view_xmin, task.view_xmax])
            task_ys.extend([task.view_ymin, task.view_ymax])
            cx = 0.5 * (task.view_xmin + task.view_xmax)
            cy = 0.5 * (task.view_ymin + task.view_ymax)
            centers_x.append(cx)
            centers_y.append(cy)

            width = task.view_xmax - task.view_xmin
            height = task.view_ymax - task.view_ymin
            ax.add_patch(
                patches.Rectangle(
                    (task.view_xmin, task.view_ymin),
                    width,
                    height,
                    linewidth=box_lw,
                    edgecolor=color,
                    facecolor="none",
                    alpha=0.95,
                    zorder=5,
                )
            )

            if task.cell_list:
                c_xs = [(c.cell_xmin + c.cell_xmax) / 2 for c in task.cell_list]
                c_ys = [(c.cell_ymin + c.cell_ymax) / 2 for c in task.cell_list]
                ax.scatter(c_xs, c_ys, s=10, c=color, marker=".", edgecolors="none", zorder=6)

            # 顺序编号：白字黑边，提高在深色底上的可读性
            ax.text(
                cx,
                cy,
                str(task.task_index),
                color="white",
                fontsize=label_fontsize,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=10,
                path_effects=[path_effects.withStroke(linewidth=2.5, foreground="black")],
            )

        # 拍摄路径折线
        if len(centers_x) >= 2:
            ax.plot(
                centers_x,
                centers_y,
                color="cyan",
                linewidth=1.2,
                alpha=0.75,
                zorder=4,
                label="shoot order",
            )
            ax.scatter(
                [centers_x[0]],
                [centers_y[0]],
                s=120,
                c="lime",
                marker="o",
                zorder=11,
                label="start #1",
            )
            ax.scatter(
                [centers_x[-1]],
                [centers_y[-1]],
                s=120,
                c="magenta",
                marker="s",
                zorder=11,
                label=f"end #{meg_tasks[-1].task_index}",
            )

        if task_xs and task_ys:
            # 相对视口尺寸自适应边距，避免全图过大
            x_span = max(task_xs) - min(task_xs)
            y_span = max(task_ys) - min(task_ys)
            margin = max(800.0, 0.03 * max(x_span, y_span))
            ax.set_xlim(min(task_xs) - margin, max(task_xs) + margin)
            ax.set_ylim(min(task_ys) - margin, max(task_ys) + margin)

        if draw_wbc and len(wbc_rects) > 0:
            wbc_arr = np.asarray(wbc_rects, dtype=np.float32)
            wbc_cx = wbc_arr[:, 0] + wbc_arr[:, 2] * 0.5
            wbc_cy = wbc_arr[:, 1] + wbc_arr[:, 3] * 0.5
            ax.scatter(
                wbc_cx,
                wbc_cy,
                s=40,
                c="red",
                marker="+",
                linewidths=1.2,
                zorder=3,
                alpha=0.5,
                label="WBC centers",
            )

        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.tick_params(colors="white")
        ax.set_xlabel("Global X (pixel)", color="white")
        ax.set_ylabel("Global Y (pixel)", color="white")
        ax.legend(
            loc="upper right",
            facecolor="black",
            framealpha=0.55,
            labelcolor="white",
            fontsize=9,
        )

    # 字号随任务数量调整：任务少就更大
    n = max(len(meg_tasks), 1)
    label_fs = float(np.clip(14 - 0.04 * n, 7, 14))

    # 图 4：含 WBC 参考
    fig4, ax4 = plt.subplots(figsize=(16, 12))
    fig4.patch.set_facecolor("black")
    _draw_order_figure(ax4, draw_wbc=True, label_fontsize=label_fs, box_lw=1.4)
    ax4.set_title(
        f"Figure 4 (MEG): Views + Shoot Order (n={len(meg_tasks)})",
        color="white",
        fontsize=15,
    )
    fig4.savefig(
        save_path_base / "fig4_meg_tasks.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="black",
    )
    plt.close(fig4)

    print(f"[SUCCESS][MEG] MEG 可视化图片已保存至: {save_path_base / 'fig4_meg_tasks.png'}")


# ===================== 主逻辑 =====================
def main() -> None:
    viz_cfg = VizConfigMeg()
    json_path = Path(viz_cfg.json_path)
    out_dir = Path(viz_cfg.out_dir)

    # 1. 加载 SmearProject（与 main_wbc 相同入口）
    project = SmearProject.load_json(str(json_path))
    print(f"[INFO][MEG] 成功加载项目: {project.smear_type}")

    # 2. 构造 BM40Config（需先取 WBC_cell_type，用于有核细胞过滤）
    bm_cfg = BM40Config(target_cell_num_MEG=50*3, 
                        dpi=134912, 
                        x100_rect_width=605,
                        x100_rect_height=445,
                        View_type="MEG", 
                        Smear_type=project.smear_type)

    # 3. 从 WBC 选区结果读取视野边界，用于 MEG 排序参考
    wbc_results_path = out_dir / "results.json"
    wbc_rects = collect_wbc_view_rects_from_results(wbc_results_path)
    print(f"[INFO][MEG] 从 WBC results.json 中解析到 {len(wbc_rects)} 个拍摄视野用于 MEG 排序。")
    if len(wbc_rects) == 0:
        print(
            "[ERROR][MEG] 未解析到任何 WBC 拍摄视野，"
            "无法计算 MEG 排序参考。"
        )
        return

    import time
    # start_time = time.time()
    # pipeline = MegSamplingPipeline(bm_cfg)
    # end_time = time.time()
    # print(f"[INFO][MEG] 构造 MegSamplingPipeline 时间: {end_time - start_time} 秒")

    start_time = time.time()
    # 4. 运行 MEG 采样
    pipeline = MegSamplingPipeline(bm_cfg)
    meg_tasks: List[TaskOutput] = pipeline.run_meg(
        project=project,
        wbc_rects=wbc_rects,
    )
    end_time = time.time()
    print(f"[INFO][MEG] 运行 MEG 采样时间: {end_time - start_time} 秒")
    print(f"[INFO][MEG] 算法执行完成，生成了 {len(meg_tasks)} 个 MEG 拍摄视野")

    # 5. 保存 MEG 结果到 JSON
    json_ready_results = [task.to_dict() for task in meg_tasks]
    meg_result_path = out_dir / viz_cfg.meg_result_json
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(meg_result_path, "w", encoding="utf-8") as f:
        json.dump(json_ready_results, f, indent=2, ensure_ascii=False)

    print(f"[INFO][MEG] MEG 任务结果已保存至: {meg_result_path}")

    # 6. 执行 MEG 可视化（只用到 grid，不需要 best_res 和 user_search_mask）
    if pipeline.grid is not None and meg_tasks:
        visualize_meg_results(
            tasks=meg_tasks,
            grid_info=pipeline.grid,
            save_path_base=out_dir,
            wbc_rects=wbc_rects,   # 把有核视野列表传进去
        )


if __name__ == "__main__":
    main()