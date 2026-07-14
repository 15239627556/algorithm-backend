# config.py
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
import numpy as np
import cv2

@dataclass
class BM40Config:
    # ========= 允许外部设置的参数 =========
    user_choice_area: Optional[Dict[str, int]] = None   # 用户框选的搜索区域 (x_min, y_min, x_max, y_max) 全局坐标系下的物理坐标
    target_cell_num_WBC: int = 300  # 骨髓-血片-用户设定的目标有核细胞数量(wbc)    
    target_cell_num_MEG: int = 300  # 骨髓-用户设定的目标巨核细胞数量(meg)  
    x100_rect_width: int = 384      # 百倍视野尺寸-宽 
    x100_rect_height: int = 283     # 百倍视野尺寸-高    
    x100_rect_size_scale: float = 0.9  # set-cover 覆盖判定缩放（中心点定位）；落盘仍用标称 x100 尺寸
    heatmap_orientation: int = 1    # 0: 头部在右，1: 头部在左
    dpi: int = 138430                   # 扫描层 DPI（默认40倍DPI）
    View_type: str = "WBC"              # 视野类型标识（写入 TaskOutput）
    Smear_type: str = "BM"              # 涂片类型标识（写入 TaskOutput，建议与 project.smear_type 一致）
    view_path_order_mode: str = "band_snake"  # 组内拍摄顺序：band_snake | tsp | scanline
    view_path_band_height: int = 5000  # band_snake 的 Y 方向分带高度（像素）


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
    angles_bm: Tuple[int, ...] = field(default=(0, 15, -15), init=False)  # BM 旋转角：轴对齐 + 小倾角
    angles_pb: Tuple[int, ...] = field(default=(0,), init=False)          # PB 旋转角：仅轴对齐
    search_area_scales_bm: Tuple[float, ...] = field(
        default=(4, 5, 6, 8, 10, 15, 20, 30, 40, 60, 100), init=False
    )   # 骨髓选区面积跨度候选（相对于标准 Tile 面积的倍数）
    search_area_scales_pb: Tuple[float, ...] = field(
        default=(20, 30, 40, 60, 100, 150, 200, 300, 400, 500, 600, 800), init=False
    )   # 血片选区面积跨度候选（有核细胞稀疏，需更大窗口）
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
    edge_avoidance_radius: int = field(default=20, init=False)                # 边缘规避半径（单位：格子数）
    edge_penalty_magnitude: float = field(default=-10.0, init=False)         # 边缘惩罚幅度

    # --- 百倍视野选区覆盖算法参数 ---
    setcover_pad: int = field(default=100, init=False)     # 百倍视野选区覆盖算法时，搜索区域扩展像素数

    # --- 输出配置 ---
    Initial_name: str = field(default="Initial", init=False) # 初始选区名称
    Extra_name: str = field(default="Extra", init=False)     # 补拍选区名称

 
    MEG_cell_type: int = field(default=100001, init=False)  # MEG 细胞类型
    WBC_cell_type: int = field(default=100000, init=False)  # WBC 细胞类型 
    RBC_cell_type: int = field(default=100002, init=False)  # RBC 细胞类型 

    def get_search_area_scales(self) -> Tuple[float, ...]:
        """按涂片类型返回搜索窗口面积跨度。"""
        if self.Smear_type.upper() == "PB":
            return self.search_area_scales_pb
        return self.search_area_scales_bm

    def get_angles(self) -> Tuple[int, ...]:
        """按涂片类型返回选区旋转角候选。"""
        if self.Smear_type.upper() == "PB":
            return self.angles_pb
        return self.angles_bm
