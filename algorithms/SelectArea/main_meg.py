# main_meg.py
from __future__ import annotations

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from project.smear_project import SmearProject

from dataclasses import dataclass
from typing import Any, Dict, List

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .data_structure import TaskOutput
from .config import BM40Config
from .pipeline_meg import MegSamplingPipeline


# ===================== 可视化配置 =====================
@dataclass(frozen=True)
class VizConfigMeg:
    # 与 WBC 共用同一 JSON 与输出目录，方便协同调试
    json_path: str = "/home/ubuntu/VScodeProjects/项目json数据/data2025063005.json"
    out_dir: str = "/home/ubuntu/VScodeProjects/algorithm-backend/algorithms/SelectArea/output"
    # WBC 结果默认文件名
    wbc_result_json: str = "results.json"
    # MEG 结果输出文件名
    meg_result_json: str = "results_meg.json"


# ===================== 可视化：MEG 任务 =====================
def visualize_meg_results(
    tasks: List[TaskOutput],
    grid_info: Any,
    save_path_base: Path,
    wbc_rects: List[List[float]]
) -> None:
    """
    可视化 MEG 结果：
    - 图 3：灰度 Score Heatmap（沿用 WBC）
    - 图 4：MEG 视野框 + 巨核细胞点（物理坐标，截取视口）
    """
    if not tasks:
        print("[WARNING][MEG] 无 MEG 采样任务，跳过可视化生成。")
        return

    # 1. 确保输出目录存在
    save_path_base.mkdir(parents=True, exist_ok=True)

    # 2. 准备热力图数据
    heatmap_data = grid_info.finalize(fill_value=np.nan)

    # 计算归一化范围
    valid_scores = heatmap_data[~np.isnan(heatmap_data)]
    vmin, vmax = (
        (np.nanmin(valid_scores), np.nanmax(valid_scores))
        if valid_scores.size > 0
        else (0, 1)
    )

    # ---------------------------------------------------------
    # 图 3：纯分值图 (Heatmap Only) - 复用 WBC 的输出文件名
    # ---------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    im3 = ax3.imshow(
        heatmap_data,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    plt.colorbar(im3, ax=ax3, label="Score (Grayscale)")
    ax3.set_title("Figure 3 (MEG): Score Grid Heatmap (Grayscale)")
    fig3.savefig(save_path_base / "heatmap_meg.png", dpi=200, bbox_inches="tight")
    plt.close(fig3)

    # ---------------------------------------------------------
    # 图 4：MEG 视野框 + 巨核细胞点 - 物理坐标，强制截取
    # ---------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(14, 12))
    ax4.set_facecolor("black")
    fig4.patch.set_facecolor("black")

    task_xs, task_ys = [], []

    for task in tasks:
        # 只绘制 view_type == "MEG" 的任务（容错，避免混入 WBC）
        if getattr(task, "view_type", "") != "MEG":
            continue

        color = "yellow"  # MEG 统一用黄色，便于区分

        task_xs.extend([task.view_xmin, task.view_xmax])
        task_ys.extend([task.view_ymin, task.view_ymax])

        # 1. 绘制视野框 (全局物理坐标)
        width = task.view_xmax - task.view_xmin
        height = task.view_ymax - task.view_ymin
        rect = patches.Rectangle(
            (task.view_xmin, task.view_ymin),
            width,
            height,
            linewidth=1.5,
            edgecolor=color,
            facecolor="none",
            alpha=0.9,
            zorder=5,
        )
        ax4.add_patch(rect)

        # 2. 绘制视野内的巨核细胞点
        if task.cell_list:
            c_xs = [(c.cell_xmin + c.cell_xmax) / 2 for c in task.cell_list]
            c_ys = [(c.cell_ymin + c.cell_ymax) / 2 for c in task.cell_list]
            ax4.scatter(
                c_xs,
                c_ys,
                s=6,
                c=color,
                marker=".",
                edgecolors="none",
                zorder=6,
            )

    # 3. 截取有视野的范围，并留出 margin
    if task_xs and task_ys:
        margin = 1500  # 边距像素
        ax4.set_xlim(min(task_xs) - margin, max(task_xs) + margin)
        ax4.set_ylim(min(task_ys) - margin, max(task_ys) + margin)

    # === 新增：绘制 WBC 视野中心点（更明显） ===
    if wbc_rects:
        wbc_arr = np.asarray(wbc_rects, dtype=np.float32)  # [x, y, w, h]
        wbc_cx = wbc_arr[:, 0] + wbc_arr[:, 2] * 0.5
        wbc_cy = wbc_arr[:, 1] + wbc_arr[:, 3] * 0.5
        # 用红色大十字标出有核视野中心
        ax4.scatter(
            wbc_cx,
            wbc_cy,
            s=60,
            c="red",
            marker="+",
            linewidths=1.5,
            zorder=8,
            label="WBC centers",
        )
        # 可选：显示图例
        ax4.legend(loc="upper right", facecolor="black", framealpha=0.5, labelcolor="white")

    ax4.set_aspect("equal")
    ax4.invert_yaxis()
    ax4.set_title(
        "Figure 4 (MEG): MEG Views & Cells (Clipped View)",
        color="white",
        fontsize=15,
    )
    ax4.set_xlabel("Global X (pixel)", color="white")
    ax4.set_ylabel("Global Y (pixel)", color="white")
    ax4.tick_params(colors="white")

    fig4.savefig(
        save_path_base / "fig4_meg_tasks.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="black",
    )
    plt.close(fig4)

    print(f"[SUCCESS][MEG] MEG 可视化图片已保存至: {save_path_base}")


# ===================== 主逻辑 =====================
def main() -> None:
    viz_cfg = VizConfigMeg()
    json_path = Path(viz_cfg.json_path)
    out_dir = Path(viz_cfg.out_dir)

    # 1. 加载 SmearProject
    project = SmearProject.load_json(str(json_path))
    print(f"[INFO][MEG] 成功加载项目: {project.smear_type}")

    # 2. 从 WBC 结果 JSON 中构造 wbc_rects
    wbc_result_path = out_dir / viz_cfg.wbc_result_json
    if not wbc_result_path.exists():
        print(f"[ERROR][MEG] 未找到 WBC 结果文件: {wbc_result_path}")
        return

    with open(wbc_result_path, "r", encoding="utf-8") as f:
        wbc_results = json.load(f)

    # 只使用 view_type == "WBC" 的视野，并转换为 [x, y, w, h]
    wbc_rects: List[List[float]] = []
    for item in wbc_results:
        if item.get("view_type") != "WBC":
            continue
        x = float(item["view_xmin"])
        y = float(item["view_ymin"])
        w = float(item["view_xmax"] - item["view_xmin"])
        h = float(item["view_ymax"] - item["view_ymin"])
        wbc_rects.append([x, y, w, h])

    if not wbc_rects:
        print("[ERROR][MEG] 从 WBC 结果中未解析到任何 WBC 视野，无法计算 MEG 排序参考。")
        return

    print(f"[INFO][MEG] 解析到 {len(wbc_rects)} 个 WBC 视野用于 MEG 排序。")

    # 3. 构造 BM40Config
    bm_cfg = BM40Config(target_cell_num_MEG=500)
    pipeline = MegSamplingPipeline(bm_cfg)

    # 4. 运行 MEG 采样
    meg_tasks: List[TaskOutput] = pipeline.run_meg(
        project=project,
        wbc_rects=wbc_rects,
    )
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