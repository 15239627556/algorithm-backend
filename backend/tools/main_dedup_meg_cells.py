# main_dedup_meg_cells.py
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from project.cells import Cell
from project.smear_project import SmearProject
from project.tiles import Tile
from backend.tools.dedup_cells_across_tiles import dedup_cells_across_tiles


# ===================== 运行配置 =====================
@dataclass(frozen=True)
class DedupMegConfig:
    # BM 项目 JSON（SmearProject.load_json），含 40x layers.tiles 与 tile.cells
    json_path: str = (
        "/home/ubuntu/VScodeProjects/项目json数据/巨核细胞去重问题/"
        "3ed2f0d3b6b14c11b9837779d477d487.json"
    )
    out_dir: str = str(Path(__file__).resolve().parent / "output")
    # 去重结果输出文件名（留空则使用 <输入stem>_dedup_meg.json）
    output_json: str = ""
    dpi: int = 140750
    cell_type: int = 100001  # 巨核细胞
    iou_thresh: float = 0.2
    tile_w: int = 2448
    tile_h: int = 2048


def count_cells(tiles: List[Tile], cell_type: int | None = None) -> int:
    total = 0
    for tile in tiles:
        cells = tile.cells or []
        if cell_type is None:
            total += len(cells)
        else:
            total += sum(1 for c in cells if c.cell_type == cell_type)
    return total


def partition_cells_by_type(
    tiles: List[Tile],
    target_cell_type: int,
) -> Dict[str, List[Cell]]:
    """
    将每个 tile 中非目标类型的细胞暂存起来，tile 内仅保留目标类型细胞。
    返回 image_uid -> 其他类型细胞列表。
    """
    other_cells_by_uid: Dict[str, List[Cell]] = {}
    for tile in tiles:
        cells = tile.cells or []
        target_cells = [c for c in cells if c.cell_type == target_cell_type]
        other_cells = [c for c in cells if c.cell_type != target_cell_type]
        tile.cells = target_cells
        other_cells_by_uid[tile.image_uid] = other_cells
    return other_cells_by_uid


def merge_other_cells_back(
    tiles: List[Tile],
    other_cells_by_uid: Dict[str, List[Cell]],
) -> None:
    """将暂存的非目标类型细胞写回对应 tile。"""
    for tile in tiles:
        other_cells = other_cells_by_uid.get(tile.image_uid, [])
        if other_cells:
            tile.cells = other_cells + (tile.cells or [])


# ===================== 主逻辑 =====================
def main() -> None:
    cfg = DedupMegConfig()
    json_path = Path(cfg.json_path)
    out_dir = Path(cfg.out_dir)

    if not json_path.exists():
        print(f"[ERROR][DEDUP_MEG] 输入文件不存在: {json_path}")
        return

    print(f"[INFO][DEDUP_MEG] 加载项目 JSON: {json_path}")
    project = SmearProject.load_json(str(json_path))
    print(f"[INFO][DEDUP_MEG] 涂片类型: {project.smear_type}")

    layer_40x = project.get_layer(cfg.dpi)
    if not layer_40x or not layer_40x.tiles:
        print(f"[ERROR][DEDUP_MEG] dpi={cfg.dpi} 层无有效 tiles")
        return

    tiles_40x = list(layer_40x.tiles.values())
    print(f"[INFO][DEDUP_MEG] dpi={cfg.dpi}, tiles 数量: {len(tiles_40x)}")

    before_target = count_cells(tiles_40x, cfg.cell_type)
    before_total = count_cells(tiles_40x)
    print(
        f"[INFO][DEDUP_MEG] 去重前 cell_type={cfg.cell_type} 数量: {before_target} "
        f"(全部细胞: {before_total})"
    )
    if before_target == 0:
        print(f"[WARN][DEDUP_MEG] 未找到 cell_type={cfg.cell_type} 的细胞，跳过去重。")
        return

    other_cells_by_uid = partition_cells_by_type(tiles_40x, cfg.cell_type)

    start_time = time.perf_counter()
    dedup_cells_across_tiles(
        tiles_40x=tiles_40x,
        tile_w=cfg.tile_w,
        tile_h=cfg.tile_h,
        iou_thresh=cfg.iou_thresh,
    )
    elapsed_s = time.perf_counter() - start_time

    merge_other_cells_back(tiles_40x, other_cells_by_uid)

    after_target = count_cells(tiles_40x, cfg.cell_type)
    after_total = count_cells(tiles_40x)
    removed = before_target - after_target

    print(f"[INFO][DEDUP_MEG] 去重后 cell_type={cfg.cell_type} 数量: {after_target}")
    print(f"[INFO][DEDUP_MEG] 去重减少 cell_type={cfg.cell_type} 数量: {removed}")
    print(f"[INFO][DEDUP_MEG] 去重后全部细胞数量: {after_total}")
    print(f"[INFO][DEDUP_MEG] dedup_cells_across_tiles 运行时间: {elapsed_s:.6f} 秒")

    output_name = cfg.output_json or f"{json_path.stem}_dedup_meg.json"
    save_path = out_dir / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    project.save_json(str(save_path))
    print(f"[SUCCESS][DEDUP_MEG] 去重后的项目已保存至: {save_path}")


if __name__ == "__main__":
    main()
