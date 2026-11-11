# plot_meg_cells_vs_fov.py
"""把 log 中的 MEG 视野与 JSON 巨核细胞画在一起。

默认两张图：
1) 全图：仅 90 个 MEG 视野 + 落在视野内的细胞；
2) 局部：视野 19/20 附近，把视野外的 JSON 巨核也画出来，便于核对漏拍。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

from _paths import OUTPUT_DIR, setup_sys_path
from plot_meg_from_log import parse_task_list_from_log_line

setup_sys_path()

from algorithms.SelectArea.config import BM40Config
from project.smear_project import SmearProject
from project.tiles import Tile

BBox = Tuple[int, int, int, int]  # xmin, ymin, xmax, ymax

_TASK_ID = "031bfdfd84bf4d4ea7a97e54cdd4ed7c"


@dataclass(frozen=True)
class VizConfig:
    json_path: str = (
        "/home/ubuntu/VScodeProjects/项目json数据/巨核拍摄顺序不正确/"
        f"20260902002/{_TASK_ID}.json"
    )
    log_path: str = str(OUTPUT_DIR / "log-20260902002.txt")
    task_id: str = _TASK_ID
    cell_type: int = BM40Config().MEG_cell_type  # 100001
    out_dir: str = str(OUTPUT_DIR)
    out_name: str = "fig_meg_cells_vs_fov_20260902002.png"
    zoom_name: str = "fig_meg_fov19_20_nearby_20260902002.png"
    zoom_fov_indices: Tuple[int, ...] = field(default=(19, 20))
    zoom_margin: float = 2500.0


def load_meg_tasks_from_log(log_path: Path, task_id: str | None = None) -> List[Dict[str, Any]]:
    """从 roi_selection finished 行解析 MEG 视野；同一 task 多次调用时取第一次。"""
    matched: List[Dict[str, Any]] | None = None
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "roi_selection finished" not in line:
                continue
            if task_id and task_id not in line:
                continue
            tasks = parse_task_list_from_log_line(line)
            meg_tasks = [t for t in tasks if t.get("view_type") == "MEG"]
            if meg_tasks:
                matched = sorted(meg_tasks, key=lambda t: int(t.get("task_index", 0)))
                break
    if not matched:
        hint = f" task_id={task_id}" if task_id else ""
        raise ValueError(f"未在 {log_path} 中找到{hint} 的 MEG 视野")
    return matched


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


def _task_center(task: Dict[str, Any]) -> Tuple[float, float]:
    cx = 0.5 * (float(task["view_xmin"]) + float(task["view_xmax"]))
    cy = 0.5 * (float(task["view_ymin"]) + float(task["view_ymax"]))
    return cx, cy


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


def _draw_fovs(ax, meg_tasks: Sequence[Dict[str, Any]], *, label_fs: float) -> None:
    centers_x: List[float] = []
    centers_y: List[float] = []
    for task in meg_tasks:
        xmin = float(task["view_xmin"])
        ymin = float(task["view_ymin"])
        xmax = float(task["view_xmax"])
        ymax = float(task["view_ymax"])
        cx, cy = _task_center(task)
        centers_x.append(cx)
        centers_y.append(cy)
        ax.add_patch(
            patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=1.4,
                edgecolor="yellow",
                facecolor="none",
                alpha=0.95,
                zorder=4,
            )
        )
        ax.text(
            cx,
            cy,
            str(task.get("task_index")),
            color="white",
            fontsize=label_fs,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=8,
            path_effects=[path_effects.withStroke(linewidth=2.5, foreground="black")],
        )
    if len(centers_x) >= 2:
        ax.plot(
            centers_x,
            centers_y,
            color="cyan",
            linewidth=1.1,
            alpha=0.7,
            zorder=3,
            label="FOV order",
        )


def plot_cells_vs_fov(
    *,
    meg_tasks: Sequence[Dict[str, Any]],
    covered: Sequence[BBox],
    save_path: Path,
    cell_type: int,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    n = max(len(meg_tasks), 1)
    _draw_fovs(ax, meg_tasks, label_fs=float(np.clip(14 - 0.04 * n, 7, 14)))

    if covered:
        ccx = [box_center(b)[0] for b in covered]
        ccy = [box_center(b)[1] for b in covered]
        ax.scatter(
            ccx,
            ccy,
            s=22,
            c="yellow",
            marker="o",
            alpha=0.95,
            linewidths=0,
            zorder=6,
            label=f"MEG cells in FOV n={len(covered)}",
        )

    task_xs = [float(t[k]) for t in meg_tasks for k in ("view_xmin", "view_xmax")]
    task_ys = [float(t[k]) for t in meg_tasks for k in ("view_ymin", "view_ymax")]
    if task_xs and task_ys:
        x_span = max(task_xs) - min(task_xs)
        y_span = max(task_ys) - min(task_ys)
        margin = max(800.0, 0.03 * max(x_span, y_span))
        ax.set_xlim(min(task_xs) - margin, max(task_xs) + margin)
        ax.set_ylim(min(task_ys) - margin, max(task_ys) + margin)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.tick_params(colors="white")
    ax.set_xlabel("Global X (pixel)", color="white")
    ax.set_ylabel("Global Y (pixel)", color="white")
    ax.set_title(
        f"MEG FOV + in-view cells\n"
        f"FOV={len(meg_tasks)}, cells in FOV={len(covered)}, cell_type={cell_type}",
        color="white",
        fontsize=12,
    )
    ax.plot([], [], color="yellow", linewidth=1.5, label=f"MEG FOV n={len(meg_tasks)}")
    ax.legend(
        loc="upper right",
        facecolor="black",
        labelcolor="white",
        framealpha=0.55,
        fontsize=9,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def plot_fov_neighborhood(
    *,
    meg_tasks: Sequence[Dict[str, Any]],
    covered: Sequence[BBox],
    uncovered: Sequence[BBox],
    focus_indices: Sequence[int],
    margin: float,
    save_path: Path,
    cell_type: int,
) -> None:
    """放大 focus 视野附近：黄框=已拍 FOV，黄点=框内细胞，红框/红叉=JSON 有但未拍。"""
    focus = [t for t in meg_tasks if int(t.get("task_index", 0)) in set(focus_indices)]
    if not focus:
        raise ValueError(f"log 中没有 MEG 视野 {list(focus_indices)}")

    x0 = min(float(t["view_xmin"]) for t in focus) - margin
    y0 = min(float(t["view_ymin"]) for t in focus) - margin
    x1 = max(float(t["view_xmax"]) for t in focus) + margin
    y1 = max(float(t["view_ymax"]) for t in focus) + margin

    def _in_win(cx: float, cy: float) -> bool:
        return x0 <= cx <= x1 and y0 <= cy <= y1

    near_fovs = [
        t
        for t in meg_tasks
        if not (
            float(t["view_xmax"]) < x0
            or float(t["view_xmin"]) > x1
            or float(t["view_ymax"]) < y0
            or float(t["view_ymin"]) > y1
        )
    ]
    near_covered = [b for b in covered if _in_win(*box_center(b))]
    near_uncovered = [b for b in uncovered if _in_win(*box_center(b))]

    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    _draw_fovs(ax, near_fovs, label_fs=14)

    if near_covered:
        ccx = [box_center(b)[0] for b in near_covered]
        ccy = [box_center(b)[1] for b in near_covered]
        ax.scatter(
            ccx,
            ccy,
            s=36,
            c="yellow",
            marker="o",
            alpha=0.95,
            linewidths=0,
            zorder=6,
            label=f"in FOV n={len(near_covered)}",
        )

    for b in near_uncovered:
        ax.add_patch(
            patches.Rectangle(
                (b[0], b[1]),
                b[2] - b[0],
                b[3] - b[1],
                linewidth=1.8,
                edgecolor="red",
                facecolor="none",
                alpha=0.95,
                zorder=7,
            )
        )
        cx, cy = box_center(b)
        ax.scatter([cx], [cy], s=80, c="red", marker="x", linewidths=2.0, zorder=8)
        ax.text(
            cx,
            cy - 80,
            f"MISS {int(cx)},{int(cy)}\n{int(b[2]-b[0])}x{int(b[3]-b[1])}",
            color="red",
            fontsize=9,
            ha="center",
            va="top",
            zorder=9,
            path_effects=[path_effects.withStroke(linewidth=2.0, foreground="black")],
        )
    if near_uncovered:
        ax.plot([], [], color="red", linewidth=1.8, label=f"JSON MEG, no FOV n={len(near_uncovered)}")

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.tick_params(colors="white")
    ax.set_xlabel("Global X (pixel)", color="white")
    ax.set_ylabel("Global Y (pixel)", color="white")
    ax.set_title(
        f"MEG FOV {','.join(str(i) for i in focus_indices)} neighborhood\n"
        f"FOV in window={len(near_fovs)}, in-FOV cells={len(near_covered)}, "
        f"JSON MEG no FOV={len(near_uncovered)}, cell_type={cell_type}",
        color="white",
        fontsize=12,
    )
    ax.plot([], [], color="yellow", linewidth=1.5, label=f"MEG FOV n={len(near_fovs)}")
    ax.legend(
        loc="upper right",
        facecolor="black",
        labelcolor="white",
        framealpha=0.55,
        fontsize=9,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def _nearest_fov_gap(box: BBox, meg_tasks: Sequence[Dict[str, Any]]) -> Tuple[int, float]:
    cx, cy = box_center(box)
    best_i, best_d = -1, 1e18
    for t in meg_tasks:
        dx = 0.0
        if cx < float(t["view_xmin"]):
            dx = float(t["view_xmin"]) - cx
        elif cx > float(t["view_xmax"]):
            dx = cx - float(t["view_xmax"])
        dy = 0.0
        if cy < float(t["view_ymin"]):
            dy = float(t["view_ymin"]) - cy
        elif cy > float(t["view_ymax"]):
            dy = cy - float(t["view_ymax"])
        d = float(np.hypot(dx, dy))
        if d < best_d:
            best_d = d
            best_i = int(t.get("task_index", 0))
    return best_i, best_d


def main() -> None:
    cfg = VizConfig()
    json_path = Path(cfg.json_path)
    log_path = Path(cfg.log_path)
    out_path = Path(cfg.out_dir) / cfg.out_name
    zoom_path = Path(cfg.out_dir) / cfg.zoom_name

    print(f"[INFO] 加载 log MEG 视野: {log_path}")
    meg_tasks = load_meg_tasks_from_log(log_path, task_id=cfg.task_id)
    print(f"[INFO] MEG 视野={len(meg_tasks)}")

    print(f"[INFO] 加载项目 JSON: {json_path}")
    project = SmearProject.load_json(str(json_path))
    if not project.layers:
        raise RuntimeError("项目中没有任何 layer")
    layer = project.layers[0]
    print(f"[INFO] dpi={int(layer.dpi)}, MEG_cell_type={cfg.cell_type}")

    cell_boxes = collect_meg_boxes(list(layer.tiles.values()), cfg.cell_type)
    covered, uncovered = coverage(cell_boxes, meg_tasks)
    print(f"[INFO] JSON 巨核细胞={len(cell_boxes)}，视野内={len(covered)}，视野外={len(uncovered)}")

    focus = [t for t in meg_tasks if int(t.get("task_index", 0)) in set(cfg.zoom_fov_indices)]
    x0 = min(float(t["view_xmin"]) for t in focus) - cfg.zoom_margin
    y0 = min(float(t["view_ymin"]) for t in focus) - cfg.zoom_margin
    x1 = max(float(t["view_xmax"]) for t in focus) + cfg.zoom_margin
    y1 = max(float(t["view_ymax"]) for t in focus) + cfg.zoom_margin
    print(f"[RESULT] 视野 {list(cfg.zoom_fov_indices)} 附近 JSON 巨核:")
    for b in cell_boxes:
        cx, cy = box_center(b)
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        inside = [int(t.get("task_index", 0)) for t in meg_tasks if point_in_view(cx, cy, t)]
        if inside:
            print(f"  IN FOV {inside} center=({cx:.0f},{cy:.0f}) bbox={b} size={b[2]-b[0]}x{b[3]-b[1]}")
        else:
            ni, nd = _nearest_fov_gap(b, meg_tasks)
            print(
                f"  NO FOV  center=({cx:.0f},{cy:.0f}) bbox={b} size={b[2]-b[0]}x{b[3]-b[1]} "
                f"nearest=#{ni} gap={nd:.0f}px"
            )

    plot_cells_vs_fov(
        meg_tasks=meg_tasks,
        covered=covered,
        save_path=out_path,
        cell_type=cfg.cell_type,
    )
    print(f"[SUCCESS] 全图(仅视野内): {out_path}")

    plot_fov_neighborhood(
        meg_tasks=meg_tasks,
        covered=covered,
        uncovered=uncovered,
        focus_indices=cfg.zoom_fov_indices,
        margin=cfg.zoom_margin,
        save_path=zoom_path,
        cell_type=cfg.cell_type,
    )
    print(f"[SUCCESS] 局部(19/20 附近含漏拍): {zoom_path}")


if __name__ == "__main__":
    main()
