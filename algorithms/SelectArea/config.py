# config.py
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import cv2

@dataclass
class BM40Config:
    # --- 硬件与视野参数 ---
    # 40x tile 原始物理像素大小 (全局坐标系下的基准宽高)
    tile_w: int = 2448
    tile_h: int = 2048
    

    # --- 热力图核心配置 ---
    cell_size: float = 896.0               # 热力图每个格子的物理跨度 (单位：像素，需与 Tile 坐标单位一致)
    heatmap_penalty_value: float = -100.0  # 热力图无效区域（无数据覆盖）的默认填充值
    scale = 4.0                            # x 坐标放大倍数（根据你当前数据调整）
  
    # --- 头部裁剪算法参数 ---
    head_crop_height_threshold_ratio: float = 0.1  # 有效列判定阈值：该列有效像素占比 (原 rows // 10 -> 0.1)
    head_crop_width_ratio: float = 0.2             # 头部裁剪宽度比例：有效区域跨度的占比 (原 available_w // 5 -> 0.2)

    # --- 选区算法参数 ---
    angles: Tuple[int, ...] = (0, 15, 75)   # 旋转角度候选
    search_area_scales: Tuple[float, ...] = (4, 5, 6, 8, 10, 15, 20, 30, 40, 60, 100)  # 面积跨度候选（相对于标准 Tile 面积的倍数）
    window_aspect_ratios: Tuple[Tuple[float, float], ...] = (
        (1.25, 1.0), (2.0, 1.0), (3.5, 1.0), (5.0, 1.0),
        (1.0, 1.25), (1.0, 2.0), (1.0, 3.5), (1.0, 5.0)
    )  # 窗口宽高比候选 (w, h)
    
    heatmap_orientation: int = 1    # 0: 头部在右，1: 头部在左
    target_cell_num: int = 300      # 用户设定的目标细胞数量
    target_ratio: float = 3.0       # 选区时用 target_cell_num * ratio
    select_ratio_init: float = 0.2  # 根据细胞数量目标筛选候选区域,初始比例
    kernel_margin: int = 2          # 均值滤波时卷积核预留边缘
    filter_border_type: int = cv2.BORDER_CONSTANT   # 边界填充策略
    bmp_label: int = 5                              # 骨髓小粒规避参数
    init_task_select_ratio: float = 0.3             # 生成初始拍摄任务时，阈值搜索允许误差比例
    edge_avoidance_radius: int = 5                # 边缘规避半径（单位：格子数）
    edge_penalty_magnitude: float = -10.0          # 边缘惩罚幅度

    # ---百倍视野选区覆盖算法参数---
    setcover_pad: int = 100         # 百倍视野选区覆盖算法时，搜索区域扩展像素数
    x100_rect_width: int = 384      # 百倍视野尺寸-宽
    x100_rect_height: int = 283     # 百倍视野尺寸-高


    # ---输出配置---
    Initial_name: str = "Initial"  # 初始选区名称
    Extra_name: str = "Extra"      # 补拍选区名称
    

