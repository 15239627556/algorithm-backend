# plot_meg_dedup_compare.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from _paths import OUTPUT_DIR, setup_sys_path

setup_sys_path()

from project.smear_project import SmearProject
from project.tiles import Tile


BBox = Tuple[int, int, int, int]  # xmin, ymin, xmax, ymax


@dataclass(frozen=True)
class VizConfig:
    old_json: str = (
        "/home/ubuntu/VScodeProjects/项目json数据/巨核拍摄顺序不正确/"
        "202607170002/edd259c80c1546bf943b83d346c04ad0_old.json"
    )
    new_json: str = str(
        OUTPUT_DIR / "edd259c80c1546bf943b83d346c04ad0_old_dedup_meg.json"
    )

    # new_json: str = (
    #     "/home/ubuntu/VScodeProjects/项目json数据/巨核拍摄顺序不正确/"
    #     "202607170002/edd259c80c1546bf943b83d346c04ad0.json"
    # )
    out_dir: str = str(OUTPUT_DIR)
    out_name: str = "fig_meg_dedup_compare_2.png"
    dpi: Optional[int] = None
    cell_type: int = 100001  # 巨核细胞


def resolve_dpi(project: SmearProject, dpi: Optional[int]) -> int:
    if dpi is not None:
        return int(dpi)
    if not project.layers:
        raise ValueError("项目中没有任何 layer")
    return int(project.layers[0].dpi)


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


def box_centers(boxes: Sequence[BBox]) -> Tuple[List[float], List[float]]:
    cx = [0.5 * (b[0] + b[2]) for b in boxes]
    cy = [0.5 * (b[1] + b[3]) for b in boxes]
    return cx, cy


def plot_overlay(
    *,
    old_boxes: Sequence[BBox],
    new_boxes: Sequence[BBox],
    save_path: Path,
    cell_type: int,
) -> None:
    """单张叠加图：红点=未去重，青点=去重后。"""
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    if old_boxes:
        ocx, ocy = box_centers(old_boxes)
        ax.scatter(
            ocx,
            ocy,
            s=12,
            c="red",
            marker="o",
            alpha=0.85,
            linewidths=0,
            zorder=3,
            label=f"old MEG n={len(old_boxes)}",
        )

    if new_boxes:
        ncx, ncy = box_centers(new_boxes)
        ax.scatter(
            ncx,
            ncy,
            s=14,
            c="cyan",
            marker="o",
            alpha=0.95,
            linewidths=0,
            zorder=4,
            label=f"dedup MEG n={len(new_boxes)}",
        )

    if old_boxes or new_boxes:
        all_boxes = list(old_boxes) + list(new_boxes)
        xs = [b[0] for b in all_boxes] + [b[2] for b in all_boxes]
        ys = [b[1] for b in all_boxes] + [b[3] for b in all_boxes]
        margin = 1500
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.tick_params(colors="white")
    ax.set_xlabel("Global X (pixel)", color="white")
    ax.set_ylabel("Global Y (pixel)", color="white")
    ax.set_title(
        f"Overlay: old(red) vs dedup(cyan)\n"
        f"old={len(old_boxes)}, dedup={len(new_boxes)}, cell_type={cell_type}",
        color="white",
        fontsize=12,
    )
    ax.legend(loc="upper right", facecolor="black", labelcolor="white", framealpha=0.55)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def main() -> None:
    cfg = VizConfig()
    old_path = Path(cfg.old_json)
    new_path = Path(cfg.new_json)
    out_path = Path(cfg.out_dir) / cfg.out_name

    if not old_path.exists():
        print(f"[ERROR] 未去重 JSON 不存在: {old_path}")
        return
    if not new_path.exists():
        print(f"[ERROR] 已去重 JSON 不存在: {new_path}")
        return

    print(f"[INFO] 加载未去重: {old_path}")
    old_project = SmearProject.load_json(str(old_path))
    print(f"[INFO] 加载已去重: {new_path}")
    new_project = SmearProject.load_json(str(new_path))

    dpi = resolve_dpi(old_project, cfg.dpi)
    print(f"[INFO] 使用 dpi={dpi}, cell_type={cfg.cell_type}")

    old_layer = old_project.get_layer(dpi)
    new_layer = new_project.get_layer(dpi)
    if not old_layer or not old_layer.tiles:
        print(f"[ERROR] 未去重项目 dpi={dpi} 无有效 tiles")
        return
    if not new_layer or not new_layer.tiles:
        print(f"[ERROR] 已去重项目 dpi={dpi} 无有效 tiles")
        return

    old_boxes = collect_meg_boxes(list(old_layer.tiles.values()), cfg.cell_type)
    new_boxes = collect_meg_boxes(list(new_layer.tiles.values()), cfg.cell_type)
    print(f"[INFO] MEG boxes: old={len(old_boxes)}, dedup={len(new_boxes)}")

    plot_overlay(
        old_boxes=old_boxes,
        new_boxes=new_boxes,
        save_path=out_path,
        cell_type=cfg.cell_type,
    )
    print(f"[SUCCESS] 对比图已保存: {out_path}")


if __name__ == "__main__":
    main()
