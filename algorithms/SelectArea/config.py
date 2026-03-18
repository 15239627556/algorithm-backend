# config.py
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
import numpy as np
import cv2

@dataclass
class BM40Config:
    # ========= 允许外部设置的参数 =========
    user_choice_area: Optional[Dict[str, int]] = None   # 用户框选的搜索区域 (x_min, y_min, x_max, y_max) 全局坐标系下的物理坐标
    target_cell_num: int = 300      # 用户设定的目标细胞数量(wbc)     
    x100_rect_width: int = 384      # 百倍视野尺寸-宽 
    x100_rect_height: int = 283     # 百倍视野尺寸-高    
    heatmap_orientation: int = 1    # 0: 头部在右，1: 头部在左
    target_cell_num_MEG: int = 300  # 用户设定的目标细胞数量(meg)     



    # ========内部默认参数 =========

    # --- 硬件与视野参数 ---
    tile_w: int = field(default=2448, init=False)
    tile_h: int = field(default=2048, init=False)
    
    # --- 热力图核心配置 ---
    cell_size: float = field(default=224.0, init=False)               # 热力图每个格子的物理跨度 (单位：像素，需与 Tile 坐标单位一致)
    heatmap_penalty_value: float = field(default=-100.0, init=False)  # 热力图无效区域（无数据覆盖）的默认填充值
    # scale: float = field(default=4.0, init=False)                     # x 坐标放大倍数
    scale: float = field(default=1.0, init=False)
  
    # --- 头部裁剪算法参数 ---
    head_crop_height_threshold_ratio: float = field(default=0.1, init=False)   # 有效列判定阈值：该列有效像素占比 (原 rows // 10 -> 0.1)
    head_crop_width_ratio: float = field(default=0.2, init=False)              # 头部裁剪宽度比例：有效区域跨度的占比 (原 available_w // 5 -> 0.2)

    # --- 选区算法参数 ---
    angles: Tuple[int, ...] = field(default=(0, 15, 75), init=False)   # 旋转角度候选
    search_area_scales: Tuple[float, ...] = field(
        default=(4, 5, 6, 8, 10, 15, 20, 30, 40, 60, 100), init=False
    )   # 面积跨度候选（相对于标准 Tile 面积的倍数）
    window_aspect_ratios: Tuple[Tuple[float, float], ...] = field(
        default=((1.25, 1.0), (2.0, 1.0), (3.5, 1.0), (5.0, 1.0),
                 (1.0, 1.25), (1.0, 2.0), (1.0, 3.5), (1.0, 5.0)), init=False
    )   # 窗口宽高比候选 (w, h)
    
    target_ratio: float = field(default=3.0, init=False)                     # 选区时用 target_cell_num * ratio
    select_ratio_init: float = field(default=0.2, init=False)                # 根据细胞数量目标筛选候选区域,初始比例
    kernel_margin: int = field(default=2, init=False)                        # 均值滤波时卷积核预留边缘
    filter_border_type: int = field(default=cv2.BORDER_CONSTANT, init=False) # 边界填充策略
    bmp_label: int = field(default=5, init=False)                            # 骨髓小粒规避参数
    init_task_select_ratio: float = field(default=0.3, init=False)           # 生成初始拍摄任务时，阈值搜索允许误差比例
    edge_avoidance_radius: int = field(default=5, init=False)                # 边缘规避半径（单位：格子数）
    edge_penalty_magnitude: float = field(default=-10.0, init=False)         # 边缘惩罚幅度

    # --- 百倍视野选区覆盖算法参数 ---
    setcover_pad: int = field(default=100, init=False)     # 百倍视野选区覆盖算法时，搜索区域扩展像素数

    # --- 输出配置 ---
    Initial_name: str = field(default="Initial", init=False) # 初始选区名称
    Extra_name: str = field(default="Extra", init=False)     # 补拍选区名称
    View_type: str = field(default="WBC", init=False)        # 视野类型标识
    Smear_type: str = field(default="BM", init=False)        # 涂片类型标识


    WBC_cell_type: int = field(default=100000, init=False)
    MEG_cell_type: int = field(default=100001, init=False)  
