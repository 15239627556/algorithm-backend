# main_wbc_meg.py
"""WBC 选区 → MEG 选区。用法与 main_wbc.py 相同：改路径后直接运行。"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from project.smear_project import SmearProject
from project.roi_store import RoiDataset

from .config import BM40Config
from .pipeline_wbc import WBCSamplingPipeline
from .pipeline_meg import MegSamplingPipeline
from .main_wbc import visualize_results
from .main_meg import visualize_meg_results, collect_wbc_view_rects_from_results
from .project_info import load_dpi_and_orientation


@dataclass(frozen=True)
class VizConfig:
    # 输入与输出路径配置（按项目随意改）
    json_path: str = ""
    roi_path: str = (
        "/home/ubuntu/VScodeProjects/项目json数据/6楼-海德星/BM/巨核数量不够/"
        "20261113037/7d30ffa48f3f4e9793ea83ef0f8776c7.roi.npz"
    )
    out_dir: str = (
        "/home/ubuntu/VScodeProjects/项目json数据/6楼-海德星/BM/巨核数量不够/"
        "20261113037/output"
    )
    # 默认优先 NPZ；可用 SELECT_AREA_INPUT_SOURCE=json 强制走 JSON
    input_source: str = "roi"
    target_cell_num_WBC: int = 200
    target_cell_num_MEG: int = 30 * 3
    wbc_x100_rect_width: int = 405
    wbc_x100_rect_height: int = 304
    meg_x100_rect_width: int = 977
    meg_x100_rect_height: int = 817


def main() -> None:
    viz_cfg = VizConfig()
    os.makedirs(viz_cfg.out_dir, exist_ok=True)

    project = None
    roi = None
    input_source = os.getenv("SELECT_AREA_INPUT_SOURCE", viz_cfg.input_source).strip().lower()
    slide_path = viz_cfg.json_path if input_source == "json" else viz_cfg.roi_path
    if not slide_path:
        raise ValueError(f"input_source={input_source!r} 时必须配置对应的 json_path / roi_path")

    info = load_dpi_and_orientation(Path(slide_path))
    print(
        f"[INFO] 从 info 读取: dpi={info.dpi}, heatmap_orientation={info.heatmap_orientation}, "
        f"tile=({info.tile_w},{info.tile_h}), smear_type={info.smear_type}, info={info.info_path}"
    )

    if input_source == "roi":
        if not viz_cfg.roi_path:
            raise ValueError("input_source='roi' 时必须配置 roi_path")
        roi = RoiDataset.load(viz_cfg.roi_path)
        smear_type = info.smear_type or roi.smear_type
        if not roi.tiles:
            raise ValueError("ROI 数据集不含 Tile")
        print(f"[INFO] 成功加载 ROI 数据集: {viz_cfg.roi_path}")
    elif input_source == "json":
        project = SmearProject.load_json(str(slide_path))
        smear_type = info.smear_type or project.smear_type
        layer = project.get_layer(info.dpi)
        if layer is None or not layer.tiles:
            raise ValueError(f"项目中缺少 dpi={info.dpi} 的有效 Tile")
        print(f"[INFO] 成功加载项目: {smear_type}")
    else:
        raise ValueError(f"不支持的输入来源: {input_source!r}（仅支持 'json' 或 'roi'）")

    out_dir = Path(viz_cfg.out_dir)

    # ---------- WBC ----------
    wbc_cfg = BM40Config(
        target_cell_num_WBC=viz_cfg.target_cell_num_WBC,
        dpi=info.dpi,
        x100_rect_width=viz_cfg.wbc_x100_rect_width,
        x100_rect_height=viz_cfg.wbc_x100_rect_height,
        View_type="WBC",
        heatmap_orientation=info.heatmap_orientation,
        Smear_type=smear_type,
        tile_w=info.tile_w,
        tile_h=info.tile_h,
    )
    print(f"[INFO] Tile={wbc_cfg.tile_w}x{wbc_cfg.tile_h}")

    t0 = time.time()
    wbc_pipeline = WBCSamplingPipeline(wbc_cfg)
    wbc_tasks = wbc_pipeline.run(project=project, roi=roi)
    print(f"[WBC] {len(wbc_tasks)} views, {time.time() - t0:.3f}s")

    wbc_json_path = out_dir / "results.json"
    with open(wbc_json_path, "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in wbc_tasks], f, indent=2, ensure_ascii=False)

    if wbc_pipeline.best_res and wbc_pipeline.grid:
        visualize_results(
            best_res=wbc_pipeline.best_res,
            tasks=wbc_tasks,
            grid_info=wbc_pipeline.grid,
            user_search_mask=wbc_pipeline.user_search_mask,
            save_path_base=out_dir,
            pre_expand_res=wbc_pipeline.pre_expand_res,
        )

    if not wbc_tasks:
        print("[MEG] 无 WBC 视野，跳过 MEG")
        return

    # ---------- MEG ----------
    meg_cfg = BM40Config(
        target_cell_num_MEG=viz_cfg.target_cell_num_MEG,
        dpi=info.dpi,
        x100_rect_width=viz_cfg.meg_x100_rect_width,
        x100_rect_height=viz_cfg.meg_x100_rect_height,
        View_type="MEG",
        heatmap_orientation=info.heatmap_orientation,
        Smear_type=smear_type,
        tile_w=info.tile_w,
        tile_h=info.tile_h,
    )
    wbc_rects = collect_wbc_view_rects_from_results(wbc_json_path)
    t1 = time.time()
    meg_pipeline = MegSamplingPipeline(meg_cfg)
    meg_tasks = meg_pipeline.run_meg(
        project=project,
        wbc_rects=wbc_rects,
        roi=roi,
        heatmap_grid=wbc_pipeline.grid,
    )
    print(f"[MEG] {len(meg_tasks)} views, {time.time() - t1:.3f}s")

    with open(out_dir / "results_meg.json", "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in meg_tasks], f, indent=2, ensure_ascii=False)

    if meg_pipeline.grid is not None and meg_tasks:
        visualize_meg_results(
            tasks=meg_tasks,
            grid_info=meg_pipeline.grid,
            save_path_base=out_dir,
            wbc_rects=wbc_rects,
        )


if __name__ == "__main__":
    main()
