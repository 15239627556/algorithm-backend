# main_generate_views.py
"""本地复现 /generate_views：rects [[x,y,w,h],...] → 最少视野框覆盖，并出图、计时。"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from algorithms.SelectArea.setcover import SetCoverSolverParameter, solve


# ===================== 运行配置 =====================
@dataclass(frozen=True)
class GenerateViewsConfig:
    data_dir: str = "/home/ubuntu/VScodeProjects/项目json数据/20261109"
    input_names: Tuple[str, ...] = field(default=("巨核.json", "有核.json"))
    # matplotlib 默认字体不含中文，图标题用英文标签
    title_labels: Dict[str, str] = field(default_factory=lambda: {"巨核": "MEG", "有核": "WBC"})
    out_dir: str = "/home/ubuntu/VScodeProjects/项目json数据/20261109/output"
    pad: int = 100
    # json 中缺少 view_width / view_height 时使用接口默认值
    default_view_width: int = 384
    default_view_height: int = 283


def generate_views(
    rects: List[List[float]],
    view_width: int,
    view_height: int,
    pad: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """与 TaskService.generate_views 相同：返回 (细胞框 xyxy, 视野框 xywh)。"""
    arr = np.asarray(rects, dtype=np.float64).reshape(-1, 4)
    cell_bounds = np.column_stack([
        arr[:, 0],
        arr[:, 1],
        arr[:, 0] + arr[:, 2],
        arr[:, 1] + arr[:, 3],
    ])
    centers = 0.5 * (cell_bounds[:, 0:2] + cell_bounds[:, 2:4])

    x_min_all, y_min_all = cell_bounds[:, 0:2].min(axis=0) - pad
    x_max_all, y_max_all = cell_bounds[:, 2:4].max(axis=0) + pad
    bounding_rect = np.array([
        x_min_all, y_min_all,
        x_max_all - x_min_all + 1,
        y_max_all - y_min_all + 1,
    ], dtype=np.int32)

    params = SetCoverSolverParameter(rect_width=view_width, rect_height=view_height)
    views = solve(centers, bounding_rect, params)
    return cell_bounds, np.asarray(views, dtype=np.int64).reshape(-1, 4)


def coverage_stats(cell_bounds: np.ndarray, views: np.ndarray) -> Dict[str, object]:
    """按细胞中心统计：每个视野内的细胞数，以及未被任何视野覆盖的细胞。"""
    centers = 0.5 * (cell_bounds[:, 0:2] + cell_bounds[:, 2:4])
    hit = np.zeros(len(centers), dtype=np.int32)
    per_view = []
    for x, y, w, h in views:
        inside = (
            (centers[:, 0] >= x) & (centers[:, 0] <= x + w)
            & (centers[:, 1] >= y) & (centers[:, 1] <= y + h)
        )
        hit += inside
        per_view.append(int(inside.sum()))
    return {
        "per_view": per_view,
        "uncovered": int((hit == 0).sum()),
        "multi_covered": int((hit >= 2).sum()),
    }


def plot_views(
    *,
    cell_bounds: np.ndarray,
    views: np.ndarray,
    view_width: int,
    view_height: int,
    title: str,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    for xmin, ymin, xmax, ymax in cell_bounds:
        ax.add_patch(patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=0.8, edgecolor="cyan", facecolor="none", zorder=4,
        ))
    centers = 0.5 * (cell_bounds[:, 0:2] + cell_bounds[:, 2:4])
    ax.scatter(centers[:, 0], centers[:, 1], s=6, c="cyan", marker=".", zorder=5)

    for idx, (x, y, w, h) in enumerate(views, start=1):
        ax.add_patch(patches.Rectangle(
            (x, y), w, h,
            linewidth=1.5, edgecolor="lime", facecolor="none", alpha=0.9, zorder=6,
        ))
        ax.text(
            x + w / 2, y + h / 2, str(idx),
            color="lime", ha="center", va="center", fontsize=7, fontweight="bold", zorder=7,
        )

    xs = np.concatenate([cell_bounds[:, [0, 2]].ravel(), views[:, 0], views[:, 0] + views[:, 2]])
    ys = np.concatenate([cell_bounds[:, [1, 3]].ravel(), views[:, 1], views[:, 1] + views[:, 3]])
    margin = max(view_width, view_height)
    ax.set_xlim(xs.min() - margin, xs.max() + margin)
    ax.set_ylim(ys.min() - margin, ys.max() + margin)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.tick_params(colors="white")
    ax.set_xlabel("Global X (pixel)", color="white")
    ax.set_ylabel("Global Y (pixel)", color="white")
    ax.set_title(title, color="white", fontsize=13)
    ax.plot([], [], color="cyan", linewidth=1, label=f"cells n={len(cell_bounds)}")
    ax.plot([], [], color="lime", linewidth=1.5, label=f"views n={len(views)}")
    ax.legend(loc="upper right", facecolor="black", labelcolor="white", framealpha=0.6)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def run_one(input_path: Path, cfg: GenerateViewsConfig) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rects = payload.get("rects") or []
    view_width = int(payload.get("view_width", cfg.default_view_width))
    view_height = int(payload.get("view_height", cfg.default_view_height))
    name = input_path.stem
    print(f"\n[INFO][VIEWS] {name}: 输入细胞框={len(rects)}, 视野={view_width}x{view_height}, pad={cfg.pad}")
    if not rects:
        print(f"[WARN][VIEWS] {name}: rects 为空，跳过。")
        return

    start = time.perf_counter()
    cell_bounds, views = generate_views(rects, view_width, view_height, cfg.pad)
    elapsed_s = time.perf_counter() - start

    stats = coverage_stats(cell_bounds, views)
    per_view = stats["per_view"]
    print(f"[INFO][VIEWS] {name}: 生成视野={len(views)}, 求解耗时={elapsed_s:.3f} 秒")
    print(
        f"[INFO][VIEWS] {name}: 每视野细胞数 min={min(per_view, default=0)} "
        f"max={max(per_view, default=0)}，未覆盖细胞={stats['uncovered']}，"
        f"被多个视野覆盖的细胞={stats['multi_covered']}"
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "rects": views.tolist(),
        "rect_count": int(len(views)),
        "rect_count_input": len(rects),
        "view_width": view_width,
        "view_height": view_height,
        "elapsed_s": round(elapsed_s, 6),
    }
    result_path = out_dir / f"{name}_views.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    fig_path = out_dir / f"{name}_views.png"
    plot_views(
        cell_bounds=cell_bounds,
        views=views,
        view_width=view_width,
        view_height=view_height,
        title=(
            f"{cfg.title_labels.get(name, name)}: cells={len(cell_bounds)}, views={len(views)}, "
            f"FOV={view_width}x{view_height}, time={elapsed_s:.3f}s"
        ),
        save_path=fig_path,
    )
    print(f"[SUCCESS][VIEWS] {name}: 结果={result_path}，图={fig_path}")


def main() -> None:
    cfg = GenerateViewsConfig()
    data_dir = Path(cfg.data_dir)
    for input_name in cfg.input_names:
        input_path = data_dir / input_name
        if not input_path.exists():
            print(f"[ERROR][VIEWS] 输入文件不存在: {input_path}")
            continue
        run_one(input_path, cfg)


if __name__ == "__main__":
    main()
