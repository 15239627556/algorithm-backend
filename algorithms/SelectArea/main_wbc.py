# main.py
from __future__ import annotations

import os
import sys
from pathlib import Path


root_dir = Path(__file__).resolve().parents[2] 
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
from project.smear_project import SmearProject
from project.roi_store import RoiDataset

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 导入核心组件
from .data_structure import SelectionResult, TaskOutput
from .config import BM40Config
from .pipeline_wbc import WBCSamplingPipeline


# ===================== 可视化配置 =====================
@dataclass(frozen=True)
class VizConfig:
    # 输入与输出路径配置
    # json_path: str = "/home/ubuntu/VScodeProjects/项目json数据/data2025063005.json"  # scale=4.0
    # json_path: str = "/home/ubuntu/VScodeProjects/项目json数据/83a1a79fefba4f9dab89c0a7ee48ad6b.json" # scale=1.0
    json_path: str = "/home/ubuntu/VScodeProjects/项目json数据/新数据类型/20260805002/b5caee92b7554d01b4b6dec96ef8fb8c.json"
    roi_path: str | None = "/home/ubuntu/VScodeProjects/项目json数据/新数据类型/20260805002/b5caee92b7554d01b4b6dec96ef8fb8c.roi.npz"
    # 默认优先使用 NPZ；可用 SELECT_AREA_INPUT_SOURCE=json 强制走旧 JSON 路径。
    input_source: str = "roi"
    out_dir: str = "/home/ubuntu/VScodeProjects/algorithm-backend/algorithms/SelectArea/output"

    def get_color(self, region_name: str) -> str:
        """
        根据区域名称返回对应的可视化颜色
        """
        if "Primary_Region" in region_name:
            return "red"  # 初始拍摄区域（主区域）用红色
        
        # 为不同的补拍区域分配循环颜色
        # 补拍区域命名格式通常为 Supplementary_1, Supplementary_2 ...
        colors = ["lime", "cyan", "magenta", "orange", "yellow", "hotpink"]
        try:
            # 提取补拍序号后缀
            parts = region_name.split("_")
            if len(parts) > 1:
                idx = int(parts[-1])
                return colors[(idx - 1) % len(colors)]
        except (ValueError, IndexError):
            pass
        
        return "blue" # 默认颜色

# ===================== 可视化绘制核心 =====================
def visualize_results(
    best_res: "SelectionResult", 
    tasks: List["TaskOutput"], 
    grid_info: Any, 
    save_path_base: Path,
    user_search_mask: np.ndarray = None  
):
    """
    可视化结果，并在图 2 上绘制用户约束区域（如有）。
    """
    if not tasks:
        print("[WARNING] 无采样任务，跳过可视化生成。")
        return

    # 1. 确保目录存在
    save_path_base.mkdir(parents=True, exist_ok=True)

    # 2. 准备热力图数据
    heatmap_data = grid_info.finalize(fill_value=np.nan)
    
    # 计算归一化范围
    valid_scores = heatmap_data[~np.isnan(heatmap_data)]
    vmin, vmax = (np.nanmin(valid_scores), np.nanmax(valid_scores)) if valid_scores.size > 0 else (0, 1)

    # ---------------------------------------------------------
    # 图 3：纯分值图 (Heatmap Only)
    # ---------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    im3 = ax3.imshow(heatmap_data, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
    plt.colorbar(im3, ax=ax3, label='Score (Grayscale)')
    ax3.set_title("Figure 3: Score Grid Heatmap (Grayscale)")
    fig3.savefig(save_path_base / "heatmap.png", dpi=200, bbox_inches='tight')
    plt.close(fig3)

    # ---------------------------------------------------------
    # 图 2：灰度分值图 + 选区结果 + 用户约束区域
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    ax2.imshow(heatmap_data, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
    
    # A. 绘制用户约束区域 (User Choice Area) - 青色虚线框
    if user_search_mask is not None and np.any(user_search_mask > 0):
        # 寻找掩码中为 1 的坐标范围
        y_indices, x_indices = np.where(user_search_mask > 0)
        cmin, cmax = x_indices.min(), x_indices.max()
        rmin, rmax = y_indices.min(), y_indices.max()
        
        # 绘制矩形（注意：imshow 中坐标是 [col, row]）
        user_rect = patches.Rectangle(
            (cmin - 0.5, rmin - 0.5), cmax - cmin + 1, rmax - rmin + 1,
            linewidth=2, edgecolor='cyan', facecolor='none', 
            linestyle='--', label='User Search Area', zorder=9
        )
        ax2.add_patch(user_rect)

    # B. 绘制最终选区多边形 (Selection Result) - 黄色实线
    poly_grid = best_res.vertices_grid
    polygon = patches.Polygon(
        poly_grid, linewidth=2, edgecolor='yellow', facecolor='none', 
        linestyle='-', label='Final Selection', zorder=10
    )
    ax2.add_patch(polygon)
    
    ax2.set_title("Figure 2: Final Selection & User Constraint on Heatmap")
    ax2.legend(loc='upper right') # 显示图例以区分两个框
    
    fig2.savefig(save_path_base / "fig2_selection.png", dpi=200, bbox_inches='tight')
    plt.close(fig2)

    # ---------------------------------------------------------
    # 图 4：百倍视野框 + 细胞点 - 物理坐标，强制截取
    # ---------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(14, 12))
    ax4.set_facecolor('black') 
    fig4.patch.set_facecolor('black')
    
    # 获取颜色映射 (如果 VizConfig 不在作用域，请确保 main.py 中已定义)
    try:
        viz_cfg = VizConfig() 
    except NameError:
        # 兼容性兜底逻辑
        class SimpleVizCfg:
            def get_color(self, name):
                return "red" if "Initial" in name else "lime"
        viz_cfg = SimpleVizCfg()
    
    # 记录所有框的边界以进行视口截取 (Bounding Box)
    task_xs, task_ys = [], []

    for task in tasks:
        color = viz_cfg.get_color(task.region_name)
        task_xs.extend([task.view_xmin, task.view_xmax])
        task_ys.extend([task.view_ymin, task.view_ymax])

        # 1. 绘制視野框 (全局物理坐标)
        width = task.view_xmax - task.view_xmin
        height = task.view_ymax - task.view_ymin
        rect = patches.Rectangle(
            (task.view_xmin, task.view_ymin), width, height,
            linewidth=1.5, edgecolor=color, facecolor='none', alpha=0.9, zorder=5
        )
        ax4.add_patch(rect)

        # 在视野框中心标注拍摄顺序，字体颜色与框线一致
        ax4.text(
            task.view_xmin + width / 2,
            task.view_ymin + height / 2,
            str(task.task_index),
            color=color,
            ha='center',
            va='center',
            fontsize=5,
            fontweight='bold',
            zorder=7,
        )

        # 2. 绘制视野内的细胞点
        if task.cell_list:
            c_xs = [(c.cell_xmin + c.cell_xmax) / 2 for c in task.cell_list]
            c_ys = [(c.cell_ymin + c.cell_ymax) / 2 for c in task.cell_list]
            ax4.scatter(c_xs, c_ys, s=4, c=color, marker='.', edgecolors='none', zorder=6)

    # 3. 核心：截取有拍摄区域的范围，并留出 margin
    if task_xs and task_ys:
        margin = 1500 # 边距像素
        ax4.set_xlim(min(task_xs) - margin, max(task_xs) + margin)
        ax4.set_ylim(min(task_ys) - margin, max(task_ys) + margin)
    
    ax4.set_aspect('equal')
    ax4.invert_yaxis() 
    ax4.set_title(f"Figure 4: WBC Views & Cells (Clipped View)", color='white', fontsize=15)
    ax4.set_xlabel("Global X (pixel)", color='white')
    ax4.set_ylabel("Global Y (pixel)", color='white')
    ax4.tick_params(colors='white')

    fig4.savefig(save_path_base / "fig4_tasks.png", dpi=200, bbox_inches='tight', facecolor='black')
    plt.close(fig4)

    print(f"[SUCCESS] 三张可视化图片已保存至: {save_path_base}")




# main.py 修改后的主逻辑部分
def main() -> None:
    viz_cfg = VizConfig()
    project = None
    roi = None
    input_source = os.getenv("SELECT_AREA_INPUT_SOURCE", viz_cfg.input_source).strip().lower()
    if input_source == "roi":
        if not viz_cfg.roi_path:
            raise ValueError("input_source='roi' 时必须配置 roi_path")
        roi = RoiDataset.load(viz_cfg.roi_path)
        smear_type = roi.smear_type
        print(f"[INFO] 成功加载 ROI 数据集: {viz_cfg.roi_path}")
    elif input_source == "json":
        project = SmearProject.load_json(str(Path(viz_cfg.json_path)))
        smear_type = project.smear_type
        print(f"[INFO] 成功加载项目: {smear_type}")
    else:
        raise ValueError(f"不支持的输入来源: {input_source!r}（仅支持 'json' 或 'roi'）")


    # user_choice_area = {"x_min": 150000, "y_min": 30000, "x_max": 200000, "y_max": 80000}  # 示例用户选区
    # bm_cfg = BM40Config(user_choice_area=user_choice_area, target_cell_num_WBC=300)
    bm_cfg = BM40Config(target_cell_num_WBC=200, 
                        dpi=144750,
                        x100_rect_width=605,
                        x100_rect_height=445,
                        View_type="WBC", 
                        heatmap_orientation=1,
                        Smear_type=smear_type)
    # pipeline = WBCSamplingPipeline(bm_cfg)
    
    import time
    start_time = time.time()
    pipeline = WBCSamplingPipeline(bm_cfg)
    final_task_list = pipeline.run(project=project, roi=roi)
    end_time = time.time()
    print(f"[INFO] 算法执行时间: {end_time - start_time} 秒")
    print(f"[INFO] 算法执行完成，生成了 {len(final_task_list)} 个拍摄视野")

    # 转换为标准的字典列表 [{}, {}...]
    json_ready_results = [task.to_dict() for task in final_task_list]
    import json
    with open(viz_cfg.out_dir + "/results.json", "w", encoding="utf-8") as f:
        json.dump(json_ready_results, f, indent=2, ensure_ascii=False)
        
    print(f"[INFO] 转换完成，共 {len(json_ready_results)} 条任务数据")



    # 3. 执行可视化：同样直接传入 project
    if pipeline.best_res and pipeline.grid:
        visualize_results(
            best_res=pipeline.best_res,
            tasks=final_task_list,
            grid_info=pipeline.grid,
            user_search_mask=pipeline.user_search_mask,
            save_path_base=Path(viz_cfg.out_dir)
        )


if __name__ == "__main__":
    main()