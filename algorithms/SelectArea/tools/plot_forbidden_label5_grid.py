# plot_forbidden_label5_grid.py
"""在 224×224 热力图网格上可视化 label=5（骨髓小粒规避）区域。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from _paths import OUTPUT_DIR, setup_sys_path

setup_sys_path()

from project.smear_project import SmearProject
from project.tiles import Tile

from algorithms.SelectArea.config import BM40Config
from algorithms.SelectArea.heatmaps import HeatmapGrid, compute_global_bounds_from_tiles
from algorithms.SelectArea.task_region_extraction import build_forbidden_mask

BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class VizConfig:
    json_path: str = (
        "/home/ubuntu/VScodeProjects/项目json数据/巨核拍摄顺序不正确/"
        "20260720003/3a341f76f9984792b656d84d952107a3.json"
    )
    cell_type: int = 100001  # 巨核细胞
    out_dir: str = str(OUTPUT_DIR)
    out_name: str = "fig_forbidden_label5_grid_20260720003.png"


def build_lightweight_grid(tiles: Sequence[Tile], cell_size: float) -> HeatmapGrid:
    min_x, min_y, max_x, max_y = compute_global_bounds_from_tiles(tiles)
    cols = int(math.ceil((max_x - min_x) / cell_size))
    rows = int(math.ceil((max_y - min_y) / cell_size))
    values = np.zeros((rows, cols), dtype=np.float32)
    weights = np.zeros((rows, cols), dtype=np.float32)
    return HeatmapGrid(
        origin_x=min_x,
        origin_y=min_y,
        cell_size=cell_size,
        values=values,
        weights=weights,
    )


def collect_label5_rects(
    tiles: Sequence[Tile], bmp_label: int, scale: float
) -> List[Tuple[float, float, float, float]]:
    rects: List[Tuple[float, float, float, float]] = []
    for tile in tiles:
        if tile.x is None or tile.y is None:
            continue
        for row in tile.meta.get("scores", []):
            if len(row) < 6 or int(row[5]) != bmp_label:
                continue
            lx = float(row[0]) * scale
            ly = float(row[1]) * scale
            lw = float(row[2]) * scale
            lh = float(row[3]) * scale
            rects.append((tile.x + lx, tile.y + ly, lw, lh))
    return rects


def classify_meg_cells(
    tiles: Sequence[Tile],
    cell_type: int,
    grid: HeatmapGrid,
    forbidden_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    kept: List[Tuple[float, float]] = []
    forbidden: List[Tuple[float, float]] = []
    rows, cols = forbidden_mask.shape

    for tile in tiles:
        if tile.x is None or tile.y is None:
            continue
        for cell in tile.cells or []:
            if cell.cell_type != cell_type:
                continue
            cx = 0.5 * (cell.cell_xmin + cell.cell_xmax) + tile.x
            cy = 0.5 * (cell.cell_ymin + cell.cell_ymax) + tile.y
            gx = int((cx - grid.origin_x) // grid.cell_size)
            gy = int((cy - grid.origin_y) // grid.cell_size)
            if 0 <= gx < cols and 0 <= gy < rows and forbidden_mask[gy, gx]:
                forbidden.append((cx, cy))
            else:
                kept.append((cx, cy))

    kept_arr = np.asarray(kept, dtype=np.float32) if kept else np.empty((0, 2), dtype=np.float32)
    forb_arr = np.asarray(forbidden, dtype=np.float32) if forbidden else np.empty((0, 2), dtype=np.float32)
    return kept_arr, forb_arr


def plot_forbidden_grid(
    *,
    grid: HeatmapGrid,
    forbidden_mask: np.ndarray,
    label5_rects: Sequence[Tuple[float, float, float, float]],
    kept_meg: np.ndarray,
    forbidden_meg: np.ndarray,
    save_path: Path,
    bmp_label: int,
    min_component_size: int,
) -> None:
    rows, cols = forbidden_mask.shape
    cell_size = grid.cell_size

    # 高对比配色（深色底）
    COLOR_BG = "#1a1a2e"
    COLOR_FORBIDDEN = "#ff9800"       # 橙色：forbidden 网格
    COLOR_LABEL5_EDGE = "#00e5ff"     # 青色：label=5 源矩形
    COLOR_MEG_KEPT = "#76ff03"        # 亮绿：保留巨核
    COLOR_MEG_FILTERED = "#ff1744"    # 亮红：被过滤巨核

    display = np.zeros((rows, cols), dtype=np.float32)
    display[forbidden_mask > 0] = 1.0

    extent = [
        grid.origin_x,
        grid.origin_x + cols * cell_size,
        grid.origin_y + rows * cell_size,
        grid.origin_y,
    ]

    from matplotlib.colors import ListedColormap
    forbidden_cmap = ListedColormap(["#00000000", COLOR_FORBIDDEN])

    fig, ax = plt.subplots(figsize=(18, 12))
    fig.patch.set_facecolor("black")
    ax.set_facecolor(COLOR_BG)

    ax.imshow(
        display,
        extent=extent,
        origin="upper",
        cmap=forbidden_cmap,
        alpha=0.85,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        zorder=1,
    )

    # 原始 label=5 矩形边界（896×896），便于对照网格映射
    for x, y, w, h in label5_rects:
        ax.add_patch(
            patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=0.8,
                edgecolor=COLOR_LABEL5_EDGE,
                facecolor="none",
                alpha=0.9,
                zorder=2,
            )
        )

    if kept_meg.size:
        ax.scatter(
            kept_meg[:, 0],
            kept_meg[:, 1],
            s=12,
            c=COLOR_MEG_KEPT,
            marker="o",
            alpha=0.85,
            linewidths=0,
            zorder=4,
            label=f"MEG kept n={len(kept_meg)}",
        )

    if forbidden_meg.size:
        ax.scatter(
            forbidden_meg[:, 0],
            forbidden_meg[:, 1],
            s=36,
            c=COLOR_MEG_FILTERED,
            marker="x",
            linewidths=1.6,
            alpha=1.0,
            zorder=5,
            label=f"MEG in label=5 grid n={len(forbidden_meg)}",
        )

    n_forbidden_cells = int(forbidden_mask.sum())
    n_total_cells = rows * cols
    ax.set_aspect("equal")
    ax.set_xlabel("Global X (pixel)", color="white")
    ax.set_ylabel("Global Y (pixel)", color="white")
    ax.tick_params(colors="white")
    ax.set_title(
        f"Heatmap grid ({cell_size:.0f}×{cell_size:.0f}px) with label={bmp_label} forbidden cells\n"
        f"grid={cols}×{rows}, forbidden cells={n_forbidden_cells} "
        f"({100.0 * n_forbidden_cells / n_total_cells:.2f}%), "
        f"label={bmp_label} rects={len(label5_rects)}, min_component={min_component_size}",
        color="white",
        fontsize=12,
    )

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=COLOR_FORBIDDEN, lw=8, alpha=0.85, label="forbidden grid cell"),
        Line2D([0], [0], color=COLOR_LABEL5_EDGE, lw=1.5, alpha=0.9, label=f"label={bmp_label} source rect"),
    ]
    if kept_meg.size:
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MEG_KEPT,
                   markersize=7, linestyle="", label=f"MEG kept n={len(kept_meg)}")
        )
    if forbidden_meg.size:
        legend_handles.append(
            Line2D([0], [0], marker="x", color=COLOR_MEG_FILTERED, linestyle="",
                   markersize=7, markeredgewidth=1.6, label=f"MEG filtered n={len(forbidden_meg)}")
        )
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        facecolor="black",
        labelcolor="white",
        framealpha=0.6,
        fontsize=9,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def main() -> None:
    cfg = VizConfig()
    json_path = Path(cfg.json_path)
    out_path = Path(cfg.out_dir) / cfg.out_name

    print(f"[INFO] 加载项目 JSON: {json_path}")
    project = SmearProject.load_json(str(json_path))
    bm_cfg = BM40Config(dpi=134912, Smear_type=project.smear_type)

    layer = project.get_layer(bm_cfg.dpi)
    if not layer:
        raise RuntimeError(f"缺少 dpi={bm_cfg.dpi} 层")
    tiles = list(layer.tiles.values())

    grid = build_lightweight_grid(tiles, bm_cfg.cell_size)
    forbidden_mask = build_forbidden_mask(grid, bm_cfg, tiles=tiles)
    label5_rects = collect_label5_rects(tiles, bm_cfg.bmp_label, bm_cfg.scale)
    kept_meg, forbidden_meg = classify_meg_cells(
        tiles, cfg.cell_type, grid, forbidden_mask
    )

    rows, cols = forbidden_mask.shape
    print(f"[INFO] grid={cols}×{rows}, cell_size={bm_cfg.cell_size}px")
    print(f"[INFO] label={bm_cfg.bmp_label} rects={len(label5_rects)}")
    print(
        f"[INFO] forbidden grid cells={int(forbidden_mask.sum())}/"
        f"{rows * cols} ({100.0 * forbidden_mask.sum() / (rows * cols):.2f}%)"
    )
    print(f"[INFO] MEG kept={len(kept_meg)}, MEG filtered={len(forbidden_meg)}")

    plot_forbidden_grid(
        grid=grid,
        forbidden_mask=forbidden_mask,
        label5_rects=label5_rects,
        kept_meg=kept_meg,
        forbidden_meg=forbidden_meg,
        save_path=out_path,
        bmp_label=bm_cfg.bmp_label,
        min_component_size=bm_cfg.forbidden_label5_min_component_size,
    )
    print(f"[SUCCESS] 已保存: {out_path}")


if __name__ == "__main__":
    main()
