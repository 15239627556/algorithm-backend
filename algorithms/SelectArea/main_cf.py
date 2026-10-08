# main_cf.py
"""脑脊液 FOCUS_POINT 选区本地入口：加载方式对齐 main_wbc.py。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from project.roi_store import RoiDataset
from project.smear_project import SmearProject

from .config import BM40Config
from .data_structure import TaskOutput
from .pipeline_cf import CFSamplingPipeline
from .project_info import load_dpi_and_orientation


@dataclass(frozen=True)
class VizConfigCF:
    json_path: str = (
        "/home/ubuntu/VScodeProjects/项目json数据/脑脊液十倍平扫数据/1/slide1/"
        "d4095c7e352d489e8e1daa9006d22d3e.json"
    )
    roi_path: str = (
        "/home/ubuntu/VScodeProjects/项目json数据/脑脊液十倍平扫数据/1/slide1/"
        "d4095c7e352d489e8e1daa9006d22d3e.roi.npz"
    )
    input_source: str = "roi"
    out_dir: str = (
        "/home/ubuntu/VScodeProjects/项目json数据/脑脊液十倍平扫数据/1/slide1/output"
    )
    focus_point: int = 9
    view_width: int = 504
    view_height: int = 422


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CF FOCUS_POINT SelectArea")
    p.add_argument("--json", dest="json_path", default=None, help="slide .json 路径")
    p.add_argument("--roi", dest="roi_path", default=None, help="slide .roi.npz 路径")
    p.add_argument("--out-dir", dest="out_dir", default=None, help="结果输出目录")
    p.add_argument("--focus-point", type=int, default=None)
    return p.parse_args()


def _build_viz_cfg(args: argparse.Namespace) -> VizConfigCF:
    cfg = VizConfigCF()
    updates = {}
    if args.json_path:
        updates["json_path"] = args.json_path
    if args.roi_path:
        updates["roi_path"] = args.roi_path
    if args.out_dir:
        updates["out_dir"] = args.out_dir
    if args.focus_point is not None:
        updates["focus_point"] = args.focus_point
    return replace(cfg, **updates) if updates else cfg


def visualize_cf_results(
    *,
    cell_matrix: np.ndarray,
    grid_info: Any,
    task_rect: List[int],
    tasks: List[TaskOutput],
    save_path_base: Path,
) -> None:
    """绘制细胞密度热力图 + 密集区 task_rect + FOCUS_POINT 视野。"""
    save_path_base.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(cell_matrix, cmap="hot", interpolation="nearest")
    ax.set_title("CF: densest region + FOCUS_POINT FOVs on cell density")

    cs = float(grid_info.cell_size)
    ox, oy = float(grid_info.origin_x), float(grid_info.origin_y)

    def to_grid(x: float, y: float) -> tuple[float, float]:
        return ((x - ox) / cs, (y - oy) / cs)

    if task_rect and len(task_rect) == 4:
        x0, y0, x1, y1 = task_rect
        gx0, gy0 = to_grid(x0, y0)
        gx1, gy1 = to_grid(x1, y1)
        rect = patches.Rectangle(
            (gx0 - 0.5, gy0 - 0.5),
            gx1 - gx0,
            gy1 - gy0,
            linewidth=2.5,
            edgecolor="cyan",
            facecolor="none",
            linestyle="--",
            label="task_rect",
            zorder=8,
        )
        ax.add_patch(rect)

    for t in tasks:
        gx0, gy0 = to_grid(t.view_xmin, t.view_ymin)
        gx1, gy1 = to_grid(t.view_xmax, t.view_ymax)
        fov = patches.Rectangle(
            (gx0 - 0.5, gy0 - 0.5),
            gx1 - gx0,
            gy1 - gy0,
            linewidth=1.5,
            edgecolor="lime",
            facecolor="none",
            zorder=9,
        )
        ax.add_patch(fov)
        ax.text(
            (gx0 + gx1) * 0.5,
            (gy0 + gy1) * 0.5,
            str(t.task_index),
            color="white",
            fontsize=9,
            ha="center",
            va="center",
            zorder=10,
        )

    ax.legend(loc="upper right")
    fig.savefig(save_path_base / "fig_cf_focus_points.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO][CF] 可视化已保存: {save_path_base / 'fig_cf_focus_points.png'}")


def main() -> None:
    viz_cfg = _build_viz_cfg(_parse_args())
    out_dir = Path(viz_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    project = None
    roi = None
    json_path = Path(viz_cfg.json_path)
    info = load_dpi_and_orientation(json_path)
    print(
        f"[INFO] 从 info 读取: dpi={info.dpi}, heatmap_orientation={info.heatmap_orientation}, "
        f"tile=({info.tile_w},{info.tile_h}), smear_type={info.smear_type}, info={info.info_path}"
    )

    input_source = os.getenv("SELECT_AREA_INPUT_SOURCE", viz_cfg.input_source).strip().lower()
    if input_source == "roi":
        if not viz_cfg.roi_path:
            raise ValueError("input_source='roi' 时必须配置 roi_path")
        roi = RoiDataset.load(viz_cfg.roi_path)
        smear_type = info.smear_type or roi.smear_type
        if not roi.tiles:
            raise ValueError("ROI 数据集不含 Tile")
        print(f"[INFO] 成功加载 ROI 数据集: {viz_cfg.roi_path}")
    elif input_source == "json":
        project = SmearProject.load_json(str(json_path))
        smear_type = info.smear_type or project.smear_type
        layer = project.get_layer(info.dpi)
        if layer is None or not layer.tiles:
            raise ValueError(f"项目中缺少 dpi={info.dpi} 的有效 Tile")
        print(f"[INFO] 成功加载项目: {smear_type}")
    else:
        raise ValueError(f"不支持的输入来源: {input_source!r}（仅支持 'json' 或 'roi'）")

    cfg = BM40Config(
        focus_point=viz_cfg.focus_point,
        dpi=info.dpi,
        x100_rect_width=viz_cfg.view_width,
        x100_rect_height=viz_cfg.view_height,
        View_type="FOCUS_POINT",
        heatmap_orientation=info.heatmap_orientation,
        Smear_type=smear_type or "CF",
        tile_w=info.tile_w,
        tile_h=info.tile_h,
    )
    print(
        f"[INFO] Tile={cfg.tile_w}x{cfg.tile_h}, FOV={cfg.x100_rect_width}x{cfg.x100_rect_height}, "
        f"focus_point={cfg.focus_point}"
    )

    t0 = time.time()
    pipeline = CFSamplingPipeline(cfg)
    task_rect, tasks = pipeline.run(
        project=project, roi=roi, focus_point=viz_cfg.focus_point
    )
    print(f"[INFO] 算法耗时 {time.time() - t0:.3f}s，视野数={len(tasks)}")

    payload = {
        "ret_code": 200,
        "ret_desc": "API success",
        "task_rect": task_rect,
        "task_list_num": len(tasks),
        "task_list": [t.to_dict() for t in tasks],
    }
    out_json = out_dir / "results_cf.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 结果已写入: {out_json}")

    if pipeline.cell_matrix is not None and pipeline.grid is not None:
        visualize_cf_results(
            cell_matrix=pipeline.cell_matrix,
            grid_info=pipeline.grid,
            task_rect=task_rect,
            tasks=tasks,
            save_path_base=out_dir,
        )


if __name__ == "__main__":
    main()
