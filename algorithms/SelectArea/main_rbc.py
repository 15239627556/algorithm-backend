from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from project.smear_project import SmearProject

from .config import BM40Config
from .data_structure import SelectionResult, TaskOutput
from .pipeline_rbc import RBCSamplingPipeline


@dataclass(frozen=True)
class VizConfigRBC:
    # 这里改成你自己的测试 json
    json_path: str = "/home/ubuntu/Downloads/daa17e713efa4f1ea92dd49a489e7587.json"
    out_dir: str = "/home/ubuntu/VScodeProjects/algorithm-backend/algorithms/SelectArea/output"
    result_json_name: str = "results_rbc.json"

    def get_color(self, region_name: str) -> str:
        if "Primary_Region" in region_name:
            return "red"
        colors = ["lime", "cyan", "magenta", "orange", "yellow", "hotpink"]
        try:
            idx = int(region_name.split("_")[-1])
            return colors[(idx - 1) % len(colors)]
        except (ValueError, IndexError):
            return "blue"


def visualize_results(
    best_res: SelectionResult,
    tasks: List[TaskOutput],
    grid_info: Any,
    save_path_base: Path,
    user_search_mask: np.ndarray | None = None,
) -> None:
    if not tasks:
        print("[WARNING][RBC] 无采样任务，跳过可视化生成。")
        return

    save_path_base.mkdir(parents=True, exist_ok=True)
    heatmap_data = grid_info.finalize(fill_value=np.nan)
    valid_scores = heatmap_data[~np.isnan(heatmap_data)]
    vmin, vmax = (np.nanmin(valid_scores), np.nanmax(valid_scores)) if valid_scores.size > 0 else (0, 1)

    fig3, ax3 = plt.subplots(figsize=(12, 10))
    im3 = ax3.imshow(heatmap_data, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.colorbar(im3, ax=ax3, label="Score (Grayscale)")
    ax3.set_title("Figure 3 (RBC): Score Grid Heatmap (Grayscale)")
    fig3.savefig(save_path_base / "heatmap_rbc.png", dpi=200, bbox_inches="tight")
    plt.close(fig3)

    fig2, ax2 = plt.subplots(figsize=(12, 10))
    ax2.imshow(heatmap_data, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

    if user_search_mask is not None and np.any(user_search_mask > 0):
        y_indices, x_indices = np.where(user_search_mask > 0)
        cmin, cmax = x_indices.min(), x_indices.max()
        rmin, rmax = y_indices.min(), y_indices.max()
        user_rect = patches.Rectangle(
            (cmin - 0.5, rmin - 0.5),
            cmax - cmin + 1,
            rmax - rmin + 1,
            linewidth=2,
            edgecolor="cyan",
            facecolor="none",
            linestyle="--",
            label="User Search Area",
            zorder=9,
        )
        ax2.add_patch(user_rect)

    polygon = patches.Polygon(
        best_res.vertices_grid,
        linewidth=2,
        edgecolor="yellow",
        facecolor="none",
        linestyle="-",
        label="Final Selection",
        zorder=10,
    )
    ax2.add_patch(polygon)
    ax2.set_title("Figure 2 (RBC): Final Selection & User Constraint on Heatmap")
    ax2.legend(loc="upper right")
    fig2.savefig(save_path_base / "fig2_selection_rbc.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    fig4, ax4 = plt.subplots(figsize=(14, 12))
    ax4.set_facecolor("black")
    fig4.patch.set_facecolor("black")

    viz_cfg = VizConfigRBC()
    task_xs, task_ys = [], []
    for task in tasks:
        color = viz_cfg.get_color(task.region_name)
        task_xs.extend([task.view_xmin, task.view_xmax])
        task_ys.extend([task.view_ymin, task.view_ymax])

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

        if task.cell_list:
            c_xs = [(c.cell_xmin + c.cell_xmax) / 2 for c in task.cell_list]
            c_ys = [(c.cell_ymin + c.cell_ymax) / 2 for c in task.cell_list]
            ax4.scatter(c_xs, c_ys, s=4, c=color, marker=".", edgecolors="none", zorder=6)

    if task_xs and task_ys:
        margin = 1500
        ax4.set_xlim(min(task_xs) - margin, max(task_xs) + margin)
        ax4.set_ylim(min(task_ys) - margin, max(task_ys) + margin)

    ax4.set_aspect("equal")
    ax4.invert_yaxis()
    ax4.set_title("Figure 4 (RBC): Views & Cells (Clipped View)", color="white", fontsize=15)
    ax4.set_xlabel("Global X (pixel)", color="white")
    ax4.set_ylabel("Global Y (pixel)", color="white")
    ax4.tick_params(colors="white")
    fig4.savefig(save_path_base / "fig4_tasks_rbc.png", dpi=200, bbox_inches="tight", facecolor="black")
    plt.close(fig4)

    print(f"[SUCCESS][RBC] 可视化图片已保存至: {save_path_base}")


def main() -> None:
    viz_cfg = VizConfigRBC()
    json_path = Path(viz_cfg.json_path)
    out_dir = Path(viz_cfg.out_dir)

    project = SmearProject.load_json(str(json_path))
    print(f"[INFO][RBC] 成功加载项目: {project.smear_type}")

    # 血片选区仍按有核细胞数量进行选区
    bm_cfg = BM40Config(target_cell_num_WBC=300, 
                        dpi=138430, 
                        x100_rect_width=1200,
                        x100_rect_height=1000,
                        View_type="RBC", 
                        Smear_type=project.smear_type)

    import time
    start_time = time.time()
    pipeline = RBCSamplingPipeline(bm_cfg)
    final_task_list = pipeline.run(project)
    end_time = time.time()
    print(f"[INFO][RBC] 算法执行时间: {end_time - start_time} 秒")
    print(f"[INFO][RBC] 算法执行完成，生成了 {len(final_task_list)} 个拍摄视野")

    json_ready_results = [task.to_dict() for task in final_task_list]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / viz_cfg.result_json_name, "w", encoding="utf-8") as f:
        json.dump(json_ready_results, f, indent=2, ensure_ascii=False)
    print(f"[INFO][RBC] 转换完成，共 {len(json_ready_results)} 条任务数据")

    if pipeline.best_res and pipeline.grid:
        visualize_results(
            best_res=pipeline.best_res,
            tasks=final_task_list,
            grid_info=pipeline.grid,
            user_search_mask=pipeline.user_search_mask,
            save_path_base=out_dir,
        )


if __name__ == "__main__":
    main()
