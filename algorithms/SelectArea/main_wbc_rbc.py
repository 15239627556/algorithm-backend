# main_wbc_rbc.py
"""
批量有核选区：
- BM -> WBCSamplingPipeline（main_wbc.py）
- PB -> RBCSamplingPipeline（main_rbc.py）
有 {stem}.roi.npz 时优先读 npz，否则读 json；dpi / 朝向 / tile / smear_type 来自 info。
每个项目的结果与图片写到 out_root/<项目相对路径>/。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from .config import BM40Config
from .data_structure import TaskOutput
from .main_rbc import visualize_results as visualize_rbc_results
from .main_wbc import visualize_results as visualize_wbc_results
from .main_wbc_meg import (
    _load_data,
    _pick_input_source,
    discover_projects,
    tee_stdio,
)
from .pipeline_rbc import RBCSamplingPipeline
from .pipeline_wbc import WBCSamplingPipeline
from .project_info import load_dpi_and_orientation, load_project_info


DEFAULT_DATA_ROOT = Path("/home/ubuntu/VScodeProjects/项目json数据/20260807_project")
DEFAULT_OUT_ROOT = Path(__file__).resolve().parent / "output" / "20260807_project_2"

X100_WBC_WIDTH = 496
X100_WBC_HEIGHT = 415
X100_RBC_WIDTH = 482
X100_RBC_HEIGHT = 403


def run_one_project(
    proj,
    out_root: Path,
    *,
    input_source_pref: str = "auto",
    target_cell_num_wbc: int = 200,
    skip_viz: bool = False,
) -> Dict[str, Any]:
    params = load_dpi_and_orientation(proj.json_path)
    smear_type = (params.smear_type or "").strip().upper()
    if smear_type not in ("BM", "PB"):
        raise ValueError(f"不支持的 smear_type={params.smear_type!r}，仅支持 BM/PB")

    out_dir = (out_root / proj.out_key).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    is_bm = smear_type == "BM"
    x100_w = X100_WBC_WIDTH if is_bm else X100_RBC_WIDTH
    x100_h = X100_WBC_HEIGHT if is_bm else X100_RBC_HEIGHT
    view_type = "WBC" if is_bm else "wbc"
    pipeline_name = "WBC" if is_bm else "RBC"
    result_name = "results.json" if is_bm else "results_rbc.json"

    with open(out_dir / "info_used.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_info": str(params.info_path),
                "source_json": str(proj.json_path),
                "source_roi": str(proj.roi_path) if proj.roi_path else None,
                "dpi": params.dpi,
                "heatmap_orientation": params.heatmap_orientation,
                "tile_width": params.tile_w,
                "tile_height": params.tile_h,
                "smear_type": smear_type,
                "pipeline": pipeline_name,
                "x100_rect_width": x100_w,
                "x100_rect_height": x100_h,
                "target_cell_num_WBC": target_cell_num_wbc,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    input_source = _pick_input_source(input_source_pref, proj.roi_path)
    print(f"\n{'=' * 60}")
    print(f"[PROJECT] {proj.out_key}")
    print(f"  json : {proj.json_path}")
    print(f"  info : {params.info_path}")
    print(f"  roi  : {proj.roi_path}")
    print(f"  out  : {out_dir}")
    print(
        f"  smear_type={smear_type}, pipeline={pipeline_name}, source={input_source}, "
        f"dpi={params.dpi}, tile=({params.tile_w},{params.tile_h})"
    )

    project, roi, loaded_smear, _tile_w, _tile_h = _load_data(
        proj.json_path, proj.roi_path, input_source, params.dpi
    )
    if not smear_type:
        smear_type = str(loaded_smear or "").strip().upper()

    cfg = BM40Config(
        target_cell_num_WBC=target_cell_num_wbc,
        dpi=params.dpi,
        x100_rect_width=x100_w,
        x100_rect_height=x100_h,
        View_type=view_type,
        heatmap_orientation=params.heatmap_orientation,
        Smear_type=smear_type,
        tile_w=params.tile_w,
        tile_h=params.tile_h,
    )
    print(f"[INFO] 当前 Tile 尺寸: {cfg.tile_w} x {cfg.tile_h}")

    t0 = time.time()
    if is_bm:
        pipeline = WBCSamplingPipeline(cfg)
        tasks: List[TaskOutput] = pipeline.run(project=project, roi=roi)
    else:
        pipeline = RBCSamplingPipeline(cfg)
        tasks = pipeline.run(project=project, roi=roi)
    elapsed = time.time() - t0
    print(f"[{pipeline_name}] {len(tasks)} views, {elapsed:.3f}s")

    with open(out_dir / result_name, "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in tasks], f, indent=2, ensure_ascii=False)

    if not skip_viz and pipeline.best_res is not None and pipeline.grid is not None:
        visualize = visualize_wbc_results if is_bm else visualize_rbc_results
        visualize(
            best_res=pipeline.best_res,
            tasks=tasks,
            grid_info=pipeline.grid,
            user_search_mask=pipeline.user_search_mask,
            save_path_base=out_dir,
        )

    summary: Dict[str, Any] = {
        "out_key": proj.out_key,
        "smear_type": smear_type,
        "pipeline": pipeline_name,
        "input_source": input_source,
        "dpi": params.dpi,
        "heatmap_orientation": params.heatmap_orientation,
        "views": len(tasks),
        "seconds": round(elapsed, 3),
        "ok": True,
        "error": None,
        "result_json": str(out_dir / result_name),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="批量运行 BM 骨髓有核选区 / PB 血片有核选区；有 npz 优先用 npz。",
    )
    p.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="项目 .json 文件或目录（可多个）。省略则扫描 --data-root",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"批量扫描与 out_key 相对路径基准（默认: {DEFAULT_DATA_ROOT}）",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help=f"输出根目录，每个项目写到对应子目录（默认: {DEFAULT_OUT_ROOT}）",
    )
    p.add_argument(
        "--input-source",
        choices=("auto", "json", "roi"),
        default="auto",
        help="数据源：auto=有 roi.npz 则用 npz，否则 json",
    )
    p.add_argument("--target-wbc", type=int, default=200, help="target_cell_num_WBC")
    p.add_argument("--skip-viz", action="store_true", help="只写 JSON，不画图")
    p.add_argument("--limit", type=int, default=0, help="最多跑前 N 个项目（0=全部）")
    p.add_argument(
        "--smear-type",
        choices=("ALL", "BM", "PB"),
        default="ALL",
        help="只跑指定涂片类型",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="批量运行日志路径（默认: <out-root>/batch_run_YYYYMMDD_HHMMSS.log）",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.log_file is not None:
        log_path = Path(args.log_file).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = out_root / f"batch_run_{stamp}.log"

    with tee_stdio(log_path):
        inputs: List[Path] = list(args.inputs) if args.inputs else [Path(args.data_root)]
        projects = discover_projects(inputs, data_root=Path(args.data_root))
        smear_filter = args.smear_type.upper()
        if smear_filter != "ALL":
            kept = []
            for proj in projects:
                try:
                    info = load_project_info(proj.info_path)
                    st = str(info.get("smear_type") or "").strip().upper()
                except Exception:
                    st = ""
                if st == smear_filter:
                    kept.append(proj)
            projects = kept
        if args.limit and args.limit > 0:
            projects = projects[: args.limit]

        if not projects:
            print("[ERROR] 未发现任何带 info 的 BM/PB 项目")
            sys.exit(1)

        print(f"[INFO] 共 {len(projects)} 个项目 → {out_root}")
        print(f"[INFO] argv: {list(argv) if argv is not None else sys.argv[1:]}")
        all_summaries: List[Dict[str, Any]] = []
        for i, proj in enumerate(projects, 1):
            print(f"\n[{i}/{len(projects)}]")
            try:
                s = run_one_project(
                    proj,
                    out_root,
                    input_source_pref=args.input_source,
                    target_cell_num_wbc=args.target_wbc,
                    skip_viz=args.skip_viz,
                )
                all_summaries.append(s)
            except Exception as e:
                print(f"[ERROR] {proj.out_key}: {e}")
                traceback.print_exc()
                fail_dir = (out_root / proj.out_key).resolve()
                fail_dir.mkdir(parents=True, exist_ok=True)
                with open(fail_dir / "run.log", "a", encoding="utf-8") as pf:
                    pf.write(f"[ERROR] {datetime.now().isoformat()} {e}\n")
                    pf.write(traceback.format_exc())
                    pf.write("\n")
                all_summaries.append(
                    {
                        "out_key": proj.out_key,
                        "ok": False,
                        "error": str(e),
                    }
                )

        batch_path = out_root / "batch_summary.json"
        with open(batch_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        n_ok = sum(1 for s in all_summaries if s.get("ok"))
        print(f"\n[DONE] {n_ok}/{len(all_summaries)} 成功；汇总: {batch_path}")
        print(f"[DONE] 完整日志: {log_path}")

    latest = out_root / "batch_run_latest.log"
    try:
        latest.write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[DONE] 最新日志副本: {latest}")
    except Exception as e:
        print(f"[WARN] 写入 batch_run_latest.log 失败: {e}")


if __name__ == "__main__":
    main()
