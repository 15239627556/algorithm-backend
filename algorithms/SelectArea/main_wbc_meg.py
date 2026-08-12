# main_wbc_meg.py
"""
批量串联：WBC 选区 → MEG 选区。
从与项目 JSON 同目录的 info（info.json 或 {stem}.info.json）读取 dpi / heatmap_orientation 等，
固定百倍视野 496×415，输出到 output/test_project/<相对路径>/。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, TextIO, Tuple

import numpy as np

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from project.smear_project import SmearProject
from project.roi_store import RoiDataset

from .config import BM40Config
from .data_structure import TaskOutput
from .pipeline_wbc import WBCSamplingPipeline
from .pipeline_meg import MegSamplingPipeline
from .main_wbc import visualize_results
from .main_meg import visualize_meg_results, collect_wbc_view_rects_from_results


# 百倍视野尺寸：批量测试固定值，无需再改代码
X100_RECT_WIDTH = 496
X100_RECT_HEIGHT = 415

# 输出根目录
DEFAULT_OUT_ROOT = (
    Path(__file__).resolve().parent / "output" / "test_project"
)

# 默认扫描的数据根（可用 CLI 覆盖）
DEFAULT_DATA_ROOT = Path("/home/ubuntu/VScodeProjects/项目json数据/20260807_project")


class _TeeStream:
    """同时写入终端与日志文件（捕获 print / pipeline 提示）。"""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        n = 0
        for s in self.streams:
            try:
                n = s.write(data)
                s.flush()
            except Exception:
                pass
        return n

    def flush(self) -> None:
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return getattr(self.streams[0], "encoding", "utf-8") or "utf-8"


@contextmanager
def tee_stdio(log_path: Path) -> Iterator[Path]:
    """将 stdout/stderr 同时 tee 到 log 文件；退出时恢复。"""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(old_out, log_f)  # type: ignore[assignment]
    sys.stderr = _TeeStream(old_err, log_f)  # type: ignore[assignment]
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n========== batch start {ts} ==========")
        print(f"[LOG] 运行日志: {log_path}")
        yield log_path
    finally:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"========== batch end {ts} ==========\n")
        sys.stdout = old_out
        sys.stderr = old_err
        try:
            log_f.close()
        except Exception:
            pass


@dataclass(frozen=True)
class ProjectPaths:
    """单个可运行项目的路径集合。"""
    project_dir: Path
    json_path: Path
    info_path: Path
    roi_path: Optional[Path]
    # 相对 data_root 的目录，用作输出子文件夹名；data_root 外则用 parent/stem
    out_key: str


def _resolve_info_path(json_path: Path) -> Optional[Path]:
    """同路径下查找 info：{stem}.info.json 或 info.json。"""
    candidates = [
        json_path.parent / f"{json_path.stem}.info.json",
        json_path.parent / "info.json",
    ]
    seen = set()
    for p in candidates:
        p = p.resolve()
        if p in seen:
            continue
        seen.add(p)
        if p.is_file():
            return p
    return None


def _resolve_roi_path(json_path: Path) -> Optional[Path]:
    roi = json_path.parent / f"{json_path.stem}.roi.npz"
    return roi if roi.is_file() else None


def load_project_info(info_path: Path) -> Dict[str, Any]:
    with open(info_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"info 文件应为 JSON 对象: {info_path}")
    return data


def discover_projects(
    inputs: Sequence[Path],
    data_root: Optional[Path] = None,
) -> List[ProjectPaths]:
    """
    从文件/目录列表发现 (json + info) 对。
    - 文件：指向 .json 项目文件
    - 目录：递归查找 *.json（排除 *.info.json）
    """
    data_root_res = data_root.resolve() if data_root is not None else None
    found: Dict[Path, ProjectPaths] = {}

    def _out_key_for(json_path: Path) -> str:
        parent = json_path.parent.resolve()
        if data_root_res is not None:
            try:
                rel = parent.relative_to(data_root_res)
                if str(rel) == ".":
                    return json_path.stem
                return str(rel).replace("\\", "/")
            except ValueError:
                pass
        # 无 data_root 或越界：用 上一级/本级 名称
        if parent.name:
            return f"{parent.parent.name}/{parent.name}".strip("/") if parent.parent.name else parent.name
        return json_path.stem

    def _try_add(json_path: Path) -> None:
        json_path = json_path.resolve()
        name = json_path.name
        if not name.endswith(".json") or name.endswith(".info.json"):
            return
        if name in ("results.json", "results_meg.json", "results_wbc.json"):
            return
        info_path = _resolve_info_path(json_path)
        if info_path is None:
            print(f"[SKIP] 无 info 文件，跳过: {json_path}")
            return
        key = json_path
        if key in found:
            return
        found[key] = ProjectPaths(
            project_dir=json_path.parent,
            json_path=json_path,
            info_path=info_path,
            roi_path=_resolve_roi_path(json_path),
            out_key=_out_key_for(json_path),
        )

    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_file():
            _try_add(path)
        elif path.is_dir():
            for jp in sorted(path.rglob("*.json")):
                _try_add(jp)
        else:
            print(f"[WARN] 路径不存在: {path}")

    projects = sorted(found.values(), key=lambda p: p.out_key)
    return projects


def _pick_input_source(
    prefer: str,
    roi_path: Optional[Path],
) -> str:
    env = os.getenv("SELECT_AREA_INPUT_SOURCE", "").strip().lower()
    if env in ("json", "roi"):
        return env
    prefer = (prefer or "auto").strip().lower()
    if prefer == "auto":
        return "roi" if roi_path is not None else "json"
    return prefer


def _load_data(
    json_path: Path,
    roi_path: Optional[Path],
    input_source: str,
    dpi: int,
) -> Tuple[Any, Any, str, int, int]:
    """返回 (project, roi, smear_type, tile_w, tile_h)。"""
    project = None
    roi = None
    if input_source == "roi":
        if roi_path is None:
            raise ValueError(f"input_source=roi 但未找到 .roi.npz: {json_path}")
        roi = RoiDataset.load(str(roi_path))
        smear_type = roi.smear_type
        if not roi.tiles:
            raise ValueError(f"ROI 无 Tile: {roi_path}")
        first = roi.tiles[0]
        return project, roi, smear_type, int(first.w), int(first.h)

    if input_source == "json":
        project = SmearProject.load_json(str(json_path))
        smear_type = project.smear_type
        layer = project.get_layer(dpi)
        if layer is None or not layer.tiles:
            raise ValueError(f"项目缺少 dpi={dpi} 的 Tile: {json_path}")
        first = next(iter(layer.tiles.values()))
        return project, roi, smear_type, int(first.w), int(first.h)

    raise ValueError(f"不支持的 input_source: {input_source!r}")


def _info_int(info: Dict[str, Any], key: str, default: Optional[int] = None) -> int:
    if key not in info or info[key] is None:
        if default is None:
            raise KeyError(f"info 缺少字段 {key!r}")
        return int(default)
    return int(info[key])


def run_one_project(
    proj: ProjectPaths,
    out_root: Path,
    *,
    input_source_pref: str = "auto",
    target_cell_num_wbc: int = 200,
    target_cell_num_meg: int = 90,
    skip_viz: bool = False,
) -> Dict[str, Any]:
    """跑单个项目的 WBC + MEG，结果写入 out_root / proj.out_key。"""
    info = load_project_info(proj.info_path)
    dpi = _info_int(info, "dpi")
    heatmap_orientation = _info_int(info, "heatmap_orientation", default=1)
    # tile 优先用 info，缺省再从数据推断
    info_tile_w = info.get("tile_width")
    info_tile_h = info.get("tile_height")

    smear_hint = info.get("smear_type")
    target_types = str(info.get("target_cell_types") or "WBC,MEG").upper()
    run_meg = "MEG" in target_types

    out_dir = (out_root / proj.out_key).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存所用 info 快照，便于复现
    with open(out_dir / "info_used.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_info": str(proj.info_path),
                "source_json": str(proj.json_path),
                "dpi": dpi,
                "heatmap_orientation": heatmap_orientation,
                "x100_rect_width": X100_RECT_WIDTH,
                "x100_rect_height": X100_RECT_HEIGHT,
                "target_cell_num_WBC": target_cell_num_wbc,
                "target_cell_num_MEG": target_cell_num_meg,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    input_source = _pick_input_source(input_source_pref, proj.roi_path)
    print(f"\n{'=' * 60}")
    print(f"[PROJECT] {proj.out_key}")
    print(f"  json : {proj.json_path}")
    print(f"  info : {proj.info_path}")
    print(f"  out  : {out_dir}")
    print(f"  dpi={dpi}, heatmap_orientation={heatmap_orientation}, source={input_source}")

    project, roi, smear_type, tile_w, tile_h = _load_data(
        proj.json_path, proj.roi_path, input_source, dpi
    )
    if info_tile_w is not None:
        tile_w = int(info_tile_w)
    if info_tile_h is not None:
        tile_h = int(info_tile_h)
    if smear_hint:
        smear_type = str(smear_hint)

    # ---------- WBC ----------
    wbc_cfg = BM40Config(
        target_cell_num_WBC=target_cell_num_wbc,
        dpi=dpi,
        x100_rect_width=X100_RECT_WIDTH,
        x100_rect_height=X100_RECT_HEIGHT,
        View_type="WBC",
        heatmap_orientation=heatmap_orientation,
        Smear_type=smear_type,
        tile_w=tile_w,
        tile_h=tile_h,
    )
    t0 = time.time()
    wbc_pipeline = WBCSamplingPipeline(wbc_cfg)
    wbc_tasks: List[TaskOutput] = wbc_pipeline.run(project=project, roi=roi)
    t_wbc = time.time() - t0
    print(f"[WBC] {len(wbc_tasks)} views, {t_wbc:.3f}s")

    wbc_json_path = out_dir / "results.json"
    with open(wbc_json_path, "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in wbc_tasks], f, indent=2, ensure_ascii=False)

    if not skip_viz and wbc_pipeline.best_res is not None and wbc_pipeline.grid is not None:
        visualize_results(
            best_res=wbc_pipeline.best_res,
            tasks=wbc_tasks,
            grid_info=wbc_pipeline.grid,
            user_search_mask=wbc_pipeline.user_search_mask,
            save_path_base=out_dir,
        )

    summary: Dict[str, Any] = {
        "out_key": proj.out_key,
        "dpi": dpi,
        "heatmap_orientation": heatmap_orientation,
        "wbc_views": len(wbc_tasks),
        "wbc_seconds": round(t_wbc, 3),
        "meg_views": 0,
        "meg_seconds": 0.0,
        "ok": True,
        "error": None,
    }

    if not run_meg:
        print("[MEG] target_cell_types 不含 MEG，跳过")
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    if not wbc_tasks:
        print("[MEG] 无 WBC 视野，跳过 MEG")
        summary["error"] = "no_wbc_tasks"
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    # ---------- MEG ----------
    meg_cfg = BM40Config(
        target_cell_num_MEG=target_cell_num_meg,
        dpi=dpi,
        x100_rect_width=X100_RECT_WIDTH,
        x100_rect_height=X100_RECT_HEIGHT,
        View_type="MEG",
        heatmap_orientation=heatmap_orientation,
        Smear_type=smear_type,
        tile_w=tile_w,
        tile_h=tile_h,
    )
    wbc_rects = collect_wbc_view_rects_from_results(wbc_json_path)
    t1 = time.time()
    meg_pipeline = MegSamplingPipeline(meg_cfg)
    meg_tasks: List[TaskOutput] = meg_pipeline.run_meg(
        project=project,
        wbc_rects=wbc_rects,
        roi=roi,
    )
    t_meg = time.time() - t1
    print(f"[MEG] {len(meg_tasks)} views, {t_meg:.3f}s")

    meg_json_path = out_dir / "results_meg.json"
    with open(meg_json_path, "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in meg_tasks], f, indent=2, ensure_ascii=False)

    if not skip_viz and meg_pipeline.grid is not None and meg_tasks:
        visualize_meg_results(
            tasks=meg_tasks,
            grid_info=meg_pipeline.grid,
            save_path_base=out_dir,
            wbc_rects=wbc_rects,
        )

    summary["meg_views"] = len(meg_tasks)
    summary["meg_seconds"] = round(t_meg, 3)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="批量运行 WBC+MEG 选区；dpi/heatmap_orientation 从 info 读取。",
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
        help=f"输出根目录（默认: {DEFAULT_OUT_ROOT}）",
    )
    p.add_argument(
        "--input-source",
        choices=("auto", "json", "roi"),
        default="auto",
        help="数据源：auto=有 roi.npz 则用 roi（可用环境变量 SELECT_AREA_INPUT_SOURCE 覆盖）",
    )
    p.add_argument("--target-wbc", type=int, default=200, help="target_cell_num_WBC")
    p.add_argument("--target-meg", type=int, default=90, help="target_cell_num_MEG")
    p.add_argument("--skip-viz", action="store_true", help="只写 JSON，不画图")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多跑前 N 个项目（0=全部）",
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
        if args.limit and args.limit > 0:
            projects = projects[: args.limit]

        if not projects:
            print("[ERROR] 未发现任何带 info 的项目 JSON")
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
                    target_cell_num_meg=args.target_meg,
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

    # with 退出后日志文件已关闭，再写 fixed 路径副本
    latest = out_root / "batch_run_latest.log"
    try:
        latest.write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[DONE] 最新日志副本: {latest}")
    except Exception as e:
        print(f"[WARN] 写入 batch_run_latest.log 失败: {e}")


if __name__ == "__main__":
    main()
