# plot_meg_cells_vs_fov.py
"""把项目 JSON 中的巨核细胞点与 results_meg.json 中的 MEG 视野框画在同一张图上，检查覆盖情况。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from _paths import OUTPUT_DIR, setup_sys_path

setup_sys_path()

from project.smear_project import SmearProject
from project.tiles import Tile

BBox = Tuple[int, int, int, int]  # xmin, ymin, xmax, ymax


@dataclass(frozen=True)
class VizConfig:
    json_path: str = (
        "/home/ubuntu/VScodeProjects/项目json数据/巨核拍摄顺序不正确/"
        "20260720003/3a341f76f9984792b656d84d952107a3.json"
    )
    meg_results_path: str = str(OUTPUT_DIR / "results_meg.json")
    cell_type: int = 100001  # 巨核细胞
    out_dir: str = str(OUTPUT_DIR)
    out_name: str = "fig_meg_cells_vs_fov_20260720003.png"


def load_meg_tasks(meg_results_path: Path) -> List[Dict[str, Any]]:
    with open(meg_results_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    meg_tasks = [t for t in tasks if t.get("view_type") == "MEG"]
    return sorted(meg_tasks, key=lambda t: int(t.get("task_index", 0)))


def collect_meg_boxes(tiles: Sequence[Tile], cell_type: int) -> List[BBox]:
    """tile.cells 为局部坐标，需加上 tile.x/tile.y 转为全局坐标。"""
    boxes: List[BBox] = []
    for tile in tiles:
        ax = int(tile.x) if tile.x is not None else 0
        ay = int(tile.y) if tile.y is not None else 0
        for cell in tile.cells or []:
            if cell.cell_type != cell_type:
                continue
            boxes.append(
                (
                    int(cell.cell_xmin + ax),
                    int(cell.cell_ymin + ay),
                    int(cell.cell_xmax + ax),
                    int(cell.cell_ymax + ay),
                )
            )
    return boxes


def box_center(b: BBox) -> Tuple[float, float]:
    return 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])


def point_in_view(cx: float, cy: float, task: Dict[str, Any]) -> bool:
    return (
        float(task["view_xmin"]) <= cx <= float(task["view_xmax"])
        and float(task["view_ymin"]) <= cy <= float(task["view_ymax"])
    )


def coverage(
    cell_boxes: Sequence[BBox], meg_tasks: Sequence[Dict[str, Any]]
) -> Tuple[List[BBox], List[BBox]]:
    covered: List[BBox] = []
    uncovered: List[BBox] = []
    for box in cell_boxes:
        cx, cy = box_center(box)
        if any(point_in_view(cx, cy, t) for t in meg_tasks):
            covered.append(box)
        else:
            uncovered.append(box)
    return covered, uncovered


def plot_cells_vs_fov(
    *,
    cell_boxes: Sequence[BBox],
    meg_tasks: Sequence[Dict[str, Any]],
    covered: Sequence[BBox],
    uncovered: Sequence[BBox],
    save_path: Path,
    cell_type: int,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    task_xs: List[float] = []
    task_ys: List[float] = []

    for task in meg_tasks:
        xmin = float(task["view_xmin"])
        ymin = float(task["view_ymin"])
        xmax = float(task["view_xmax"])
        ymax = float(task["view_ymax"])
        task_xs.extend([xmin, xmax])
        task_ys.extend([ymin, ymax])
        ax.add_patch(
            patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=1.2,
                edgecolor="yellow",
                facecolor="none",
                alpha=0.9,
                zorder=4,
            )
        )
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        ax.text(
            cx,
            cy,
            str(task.get("task_index")),
            color="white",
            fontsize=7,
            ha="center",
            va="center",
            zorder=5,
            alpha=0.85,
        )

    if covered:
        ccx = [box_center(b)[0] for b in covered]
        ccy = [box_center(b)[1] for b in covered]
        ax.scatter(
            ccx,
            ccy,
            s=18,
            c="lime",
            marker="o",
            alpha=0.9,
            linewidths=0,
            zorder=6,
            label=f"covered MEG cells n={len(covered)}",
        )

    if uncovered:
        ucx = [box_center(b)[0] for b in uncovered]
        ucy = [box_center(b)[1] for b in uncovered]
        ax.scatter(
            ucx,
            ucy,
            s=36,
            c="red",
            marker="x",
            alpha=0.95,
            linewidths=1.4,
            zorder=7,
            label=f"UNCOVERED MEG cells n={len(uncovered)}",
        )

    xs = list(task_xs)
    ys = list(task_ys)
    for b in cell_boxes:
        xs.extend([b[0], b[2]])
        ys.extend([b[1], b[3]])
    if xs and ys:
        margin = 1500
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.tick_params(colors="white")
    ax.set_xlabel("Global X (pixel)", color="white")
    ax.set_ylabel("Global Y (pixel)", color="white")
    ax.set_title(
        f"MEG cells vs MEG FOV\n"
        f"cells={len(cell_boxes)}, FOV={len(meg_tasks)}, "
        f"covered={len(covered)}, uncovered={len(uncovered)}, cell_type={cell_type}",
        color="white",
        fontsize=12,
    )

    ax.plot([], [], color="yellow", linewidth=1.5, label=f"MEG FOV n={len(meg_tasks)}")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="upper right",
        facecolor="black",
        labelcolor="white",
        framealpha=0.55,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def main() -> None:
    cfg = VizConfig()
    json_path = Path(cfg.json_path)
    meg_results_path = Path(cfg.meg_results_path)
    out_path = Path(cfg.out_dir) / cfg.out_name

    print(f"[INFO] 加载 MEG 结果: {meg_results_path}")
    meg_tasks = load_meg_tasks(meg_results_path)
    print(f"[INFO] MEG 视野={len(meg_tasks)}")

    print(f"[INFO] 加载项目 JSON: {json_path}")
    project = SmearProject.load_json(str(json_path))
    if not project.layers:
        raise RuntimeError("项目中没有任何 layer")
    layer = project.layers[0]
    dpi = int(layer.dpi)
    print(f"[INFO] 使用 dpi={dpi}, cell_type={cfg.cell_type}")

    cell_boxes = collect_meg_boxes(list(layer.tiles.values()), cfg.cell_type)
    print(f"[INFO] JSON 巨核细胞数={len(cell_boxes)}")

    covered, uncovered = coverage(cell_boxes, meg_tasks)
    print(
        f"[RESULT] covered={len(covered)}, uncovered={len(uncovered)}, "
        f"coverage={100.0 * len(covered) / max(len(cell_boxes), 1):.2f}%"
    )
    if uncovered:
        print("[RESULT] 未被 MEG 视野覆盖的巨核细胞中心 (前 30 个):")
        for i, b in enumerate(uncovered[:30], 1):
            cx, cy = box_center(b)
            print(f"  {i:02d}. center=({cx:.1f}, {cy:.1f}) bbox={b}")

    plot_cells_vs_fov(
        cell_boxes=cell_boxes,
        meg_tasks=meg_tasks,
        covered=covered,
        uncovered=uncovered,
        save_path=out_path,
        cell_type=cfg.cell_type,
    )
    print(f"[SUCCESS] 已保存: {out_path}")


if __name__ == "__main__":
    main()
