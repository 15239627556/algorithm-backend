from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from project.smear_project import SmearProject
from .dedup_cells_across_tiles import dedup_cells_across_tiles


def count_cells(tiles) -> int:
    return sum(len(t.cells or []) for t in tiles)


def main() -> None:
    parser = argparse.ArgumentParser(description="40x tiles cell dedup runner")
    parser.add_argument(
        "--input-json",
        default="/home/ubuntu/Downloads/d1bde0b324bd41dc9fe8a1a4821e50bc.json",
        help="输入项目 JSON 路径",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=138430,
        help="40x 层 DPI（默认 138430）",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.2,
        help="NMS IoU 阈值（默认 0.2）",
    )
    parser.add_argument(
        "--save-json",
        default="",
        help="可选：保存去重后 JSON 的路径；留空则不保存",
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_path}")
        return

    project = SmearProject.load_json(str(input_path))
    layer_40x = project.get_layer(args.dpi)
    if not layer_40x:
        print(f"[ERROR] 未找到 dpi={args.dpi} 的扫描层")
        return

    tiles_40x = list(layer_40x.tiles.values())
    if not tiles_40x:
        print(f"[ERROR] dpi={args.dpi} 层没有有效 tiles")
        return

    before_cells = count_cells(tiles_40x)
    start_time = time.perf_counter()
    deduped_tiles = dedup_cells_across_tiles(tiles_40x=tiles_40x, iou_thresh=args.iou_thresh)
    elapsed_s = time.perf_counter() - start_time
    after_cells = count_cells(deduped_tiles)

    print(f"[INFO] 输入文件: {input_path}")
    print(f"[INFO] dpi: {args.dpi}, tiles 数量: {len(tiles_40x)}")
    print(f"[INFO] 去重前细胞数: {before_cells}")
    print(f"[INFO] 去重后细胞数: {after_cells}")
    print(f"[INFO] 去重减少细胞数: {before_cells - after_cells}")
    print(f"[INFO] dedup_cells_across_tiles 运行时间: {elapsed_s:.6f} 秒")

    if args.save_json:
        save_path = Path(args.save_json)
        project.save_json(str(save_path))
        print(f"[SUCCESS] 去重后的项目已保存: {save_path}")


if __name__ == "__main__":
    main()

