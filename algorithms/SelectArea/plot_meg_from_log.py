from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np


def _read_line(path: Path, line_no: int) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if i == line_no:
                return line
    raise ValueError(f"Line {line_no} not found in {path}")


def parse_task_list_from_log_line(log_line: str) -> List[Dict[str, Any]]:
    m = re.search(r"task_list=(\[\s*{.*}\s*\])\s*$", log_line.strip())
    if not m:
        m = re.search(r"task_list=(\[\s*{.*}\s*\])", log_line)
    if not m:
        raise ValueError("Could not parse task_list=... from the provided log line")
    return ast.literal_eval(m.group(1))


def _task_center(task: Dict[str, Any]) -> Tuple[float, float]:
    cx = 0.5 * (float(task["view_xmin"]) + float(task["view_xmax"]))
    cy = 0.5 * (float(task["view_ymin"]) + float(task["view_ymax"]))
    return cx, cy


def plot_meg_tasks_like_fig4(
    *,
    log_tasks: List[Dict[str, Any]],
    save_path: Path,
) -> None:
    meg_tasks = [t for t in log_tasks if t.get("view_type") == "MEG"]
    if not meg_tasks:
        raise ValueError("No MEG tasks found in task_list")

    meg_tasks = sorted(meg_tasks, key=lambda t: int(t.get("task_index", 0)))

    wbc_tasks = [t for t in log_tasks if t.get("view_type") == "WBC"]

    n = max(len(meg_tasks), 1)
    label_fs = float(np.clip(14 - 0.04 * n, 7, 14))

    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    task_xs: List[float] = []
    task_ys: List[float] = []
    centers_x: List[float] = []
    centers_y: List[float] = []

    box_lw = 1.4
    color = "yellow"

    for task in meg_tasks:
        view_xmin = float(task["view_xmin"])
        view_ymin = float(task["view_ymin"])
        view_xmax = float(task["view_xmax"])
        view_ymax = float(task["view_ymax"])

        task_xs.extend([view_xmin, view_xmax])
        task_ys.extend([view_ymin, view_ymax])

        cx, cy = _task_center(task)
        centers_x.append(cx)
        centers_y.append(cy)

        width = view_xmax - view_xmin
        height = view_ymax - view_ymin
        ax.add_patch(
            patches.Rectangle(
                (view_xmin, view_ymin),
                width,
                height,
                linewidth=box_lw,
                edgecolor=color,
                facecolor="none",
                alpha=0.95,
                zorder=5,
            )
        )

        cell_list = task.get("cell_list") or []
        if cell_list:
            c_xs = [(c[0] + c[2]) / 2 for c in cell_list]
            c_ys = [(c[1] + c[3]) / 2 for c in cell_list]
            ax.scatter(c_xs, c_ys, s=10, c=color, marker=".", edgecolors="none", zorder=6)

        ax.text(
            cx,
            cy,
            str(task.get("task_index")),
            color="white",
            fontsize=label_fs,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=10,
            path_effects=[path_effects.withStroke(linewidth=2.5, foreground="black")],
        )

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
            label=f"end #{meg_tasks[-1].get('task_index')}",
        )

    if task_xs and task_ys:
        x_span = max(task_xs) - min(task_xs)
        y_span = max(task_ys) - min(task_ys)
        margin = max(800.0, 0.03 * max(x_span, y_span))
        ax.set_xlim(min(task_xs) - margin, max(task_xs) + margin)
        ax.set_ylim(min(task_ys) - margin, max(task_ys) + margin)

    if wbc_tasks:
        wbc_c = np.asarray([_task_center(t) for t in wbc_tasks], dtype=np.float32)
        ax.scatter(
            wbc_c[:, 0],
            wbc_c[:, 1],
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

    ax.set_title(
        f"Figure 4 (MEG): Views + Shoot Order (n={len(meg_tasks)})",
        color="white",
        fontsize=15,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def main() -> None:
    log_path = Path(
        "/home/ubuntu/VScodeProjects/项目json数据/巨核拍摄顺序不正确/202607170001/app.log"
    )
    line_no = 29980
    out_path = Path(__file__).resolve().parent / "output" / "fig4_meg_tasks_from_log.png"

    log_line = _read_line(log_path, line_no)
    tasks = parse_task_list_from_log_line(log_line)
    plot_meg_tasks_like_fig4(log_tasks=tasks, save_path=out_path)
    print(f"[SUCCESS] saved: {out_path}")


if __name__ == "__main__":
    main()

