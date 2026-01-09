# main.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 导入核心组件
from data_structure import Tile, SelectionResult, TaskOutput
from config import BM40Config
from pipeline import WBCSamplingPipeline

# ===================== 可视化配置 =====================
@dataclass(frozen=True)
class VizConfig:
    # 输入与输出路径配置
    json_path: str = "/home/ubuntu/VScodeProjects/项目json数据/data2025063001.json"
    out_dir: str = "/home/ubuntu/VScodeProjects/algorithm-backend/algorithms/SelectArea/output"
    out_png_name: str = "sampling_result_viz.png"

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
    tiles: List["Tile"], 
    best_res: "SelectionResult", 
    tasks: List["TaskOutput"], 
    grid_info: Any, # HeatmapGrid
    save_path_base: Path
):
    """
    根据要求生成三张独立的可视化结果图。
    """
    if not tasks:
        print("[WARNING] 无采样任务，跳过可视化生成。")
        return

    # 1. 确保目录存在
    save_path_base.mkdir(parents=True, exist_ok=True)

    # 2. 准备热力图数据
    heatmap_data = grid_info.finalize(fill_value=np.nan)
    
    # 计算归一化范围，增强灰度图对比度
    valid_scores = heatmap_data[~np.isnan(heatmap_data)]
    if valid_scores.size > 0:
        vmin = np.nanmin(valid_scores)
        vmax = np.nanmax(valid_scores)
    else:
        vmin, vmax = 0, 1

    # ---------------------------------------------------------
    # 图 3：纯分值图 (Heatmap Only) - 灰度图
    # ---------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    # 使用 cmap='gray' 转换为灰度
    im3 = ax3.imshow(heatmap_data, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
    plt.colorbar(im3, ax=ax3, label='Score (Grayscale)')
    ax3.set_title("Figure 3: Score Grid Heatmap (Grayscale)")
    ax3.set_xlabel("Grid Column")
    ax3.set_ylabel("Grid Row")
    
    # 修正路径拼接：使用 / 运算符
    fig3.savefig(save_path_base / "heatmap.png", dpi=200, bbox_inches='tight')
    plt.close(fig3)

    # ---------------------------------------------------------
    # 图 2：灰度分值图 + 选区结果 (Selection on Heatmap)
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    ax2.imshow(heatmap_data, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
    
    # 绘制选区多边形 (vertices_grid 为 [col, row] 序列)
    # 增加线宽至 4 确保可见
    poly_grid = best_res.vertices_grid
    polygon = patches.Polygon(
        poly_grid, linewidth=2, edgecolor='yellow', facecolor='none', 
        linestyle='-', label='Selection Area', zorder=10
    )
    ax2.add_patch(polygon)
    ax2.set_title("Figure 2: Final Selection Area on Grayscale Heatmap")
    ax2.set_xlabel("Grid Column")
    ax2.set_ylabel("Grid Row")
    
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



# ===================== 数据加载适配 =====================
def _build_tiles_from_json(obj: Dict[str, Any]) -> List[Tile]:
    """
    从 JSON 对象解析 Tile 列表
    """
    layers = obj.get("layers", [])
    if not layers:
        raise KeyError("JSON 数据中缺少 'layers' 层级")

    tiles_dict = layers[0].get("tiles", {})
    tiles: List[Tile] = []
    for tile_id, d in tiles_dict.items():
        if "image_uid" not in d:
            d["image_uid"] = tile_id
        tiles.append(Tile.from_dict(d))

    # 按坐标排序
    tiles.sort(key=lambda t: ((t.y or 0), (t.x or 0)))
    return tiles

# ===================== 主程序执行 =====================
def main() -> None:
    viz_cfg = VizConfig()
    json_path = Path(viz_cfg.json_path)

    if not json_path.exists():
        print(f"[ERROR] 找不到指定的 JSON 文件: {json_path}")
        return

    # 1. 加载数据
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    tiles = _build_tiles_from_json(obj)
    print(f"[INFO] 成功加载 {len(tiles)} 个 Tiles")

    # 2. 初始化配置与 Pipeline
    bm_cfg = BM40Config(cell_size=896)
    pipeline = WBCSamplingPipeline(bm_cfg)

    # 3. 运行算法流程
    # run() 方法返回 TaskOutput 列表
    # 同时在检查模式下，pipeline 实例会持有中间结果（grid, best_res 等）
    final_task_list = pipeline.run(tiles)
    print(f"[INFO] 算法执行完成，生成了 {len(final_task_list)} 个拍摄视野")

    # 4. 执行可视化
    # 此时直接通过 pipeline 实例获取中间结果进行绘图
    if pipeline.best_res and pipeline.grid:
        visualize_results(
            tiles=tiles,
            best_res=pipeline.best_res,
            tasks=final_task_list,
            grid_info=pipeline.grid,
            save_path_base=Path(VizConfig.out_dir) # 传入输出目录 Path
        )
    else:
        print("[WARNING] Pipeline 未能生成有效的中间结果，跳过可视化。")

if __name__ == "__main__":
    main()