# Algorithm Backend

血液涂片（骨髓/外周血/脑脊液等）智能分析算法后端服务，基于 **FastAPI** 构建，通过 **NVIDIA Triton / multi_pipeline** 调用深度学习推理，支持 X40 全片扫描任务管理与 X100 高倍细胞识别、ROI 任务规划、图像增强滤镜等功能。

当前版本：**v40**

## 功能概览

| 模块 | 说明 |
|------|------|
| **X40 任务模式** | 创建扫描任务、逐块上传拼图 tile、更新坐标、检查上传完整性、获取 ROI 内细胞检测结果 |
| **X100 细胞分析** | 单张细胞图识别，或在任务上下文中按坐标裁剪分析；支持 WBC / MEG / RBC / PLAT 等细胞类型组合 |
| **ROI 任务规划** | 基于 Set Cover 算法生成最少视场覆盖方案，支持 WBC/MEG/RBC 等任务类型的 X100 采样点推荐 |
| **玻片分析** | 全片级指标分析（如 cellularity 细胞量评估） |
| **图像滤镜** | x100 / x40 图像增强（OpenCV 与深度学习超分辨率两种模式） |
| **细胞去重与过滤** | 跨 tile 去重、边缘不完整细胞过滤、高 DPI 下细长靠边细胞过滤等 |

## 技术栈

- **Web 框架**：FastAPI + Uvicorn
- **推理服务**：Triton Inference Server（gRPC）+ multi_pipeline HTTP 推理
- **深度学习**：PyTorch / TorchVision
- **图像处理**：OpenCV、Pillow、NumPy
- **算法优化**：PySCIPOpt（Set Cover）、Pandas、Matplotlib

## 项目结构

```
algorithm-backend/
├── app.py                      # FastAPI 应用入口
├── config.py                   # Triton 端点、端口、线程池等配置
├── requirements.txt
├── backend/
│   ├── routes/
│   │   ├── task.py             # 涂片分析任务 API
│   │   └── ImgFilter.py        # 图像滤镜 API
│   ├── services/
│   │   └── task_service.py     # 任务业务逻辑
│   └── tools/                  # Triton 客户端、模型控制、过滤工具等
├── project/                    # 涂片项目数据模型（Layer / Tile / Cell）
├── algorithms/
│   └── SelectArea/             # ROI 选取、Set Cover、跨 tile 去重等算法
├── uploads/                    # 任务数据与推理结果存储目录
└── logs/                       # 应用/访问/错误日志
```

## 环境要求

- Python 3.10+
- 可访问的 Triton Inference Server 及 multi_pipeline 服务
- GPU 环境（推理由 Triton 侧提供）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 Triton

编辑 `config.py` 中的 `TRITON_IPS` 列表，指向 Triton 容器/主机地址。各端点默认端口：

| 服务 | 端口 |
|------|------|
| HTTP | 8000 |
| gRPC | 8001 |
| Metrics | 8002 |
| multi_pipeline | 9000 |

### 3. 启动服务

**开发模式（单进程）：**

```bash
python app.py
```

**生产模式（Uvicorn 多 worker）：**

```bash
uvicorn app:app --host 0.0.0.0 --port 3090 --workers 2
```

服务默认监听 `0.0.0.0:3090`。启动后访问 `/` 可打开 Swagger 文档（`is_doc=True` 时启用）。

### 4. 健康检查

```bash
curl http://localhost:3090/health
# {"status":"ok"}
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `APP_HOST` / `FLASK_HOST` | `0.0.0.0` | 监听地址 |
| `APP_PORT` / `FLASK_PORT` | `3090` | 监听端口 |
| `THREAD_POOL_SIZE` | `256` | anyio 线程池大小（阻塞推理/IO 并发） |
| `TRITON_GPU_ID` | `0` | 定点使用的 Triton 端点下标 |
| `TRITON_ROUND_ROBIN` | `0` | 设为 `1` 时多端点轮询 |
| `UPDATE_COORDINATES_ASYNC_FINISH` | `1` | 坐标更新后后台异步完成合并/去重 |
| `PIPELINE_HTTP_TIMEOUT_S` | `600` | multi_pipeline 推理超时（秒） |

## API 接口

### 涂片分析 `/api/v1/smear_analysis`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/create_task` | 创建 X40 扫描任务 |
| POST | `/upload_tile` | 上传拼图块（支持 multipart 或二进制流） |
| POST | `/update_coordinates` | 更新拼图块坐标信息 |
| POST | `/check_missing_tiles` | 检查 tile 是否上传完毕 |
| POST | `/check_task_status` | 获取任务状态 |
| POST | `/get_task_result` | 获取任务 ROI 内检测结果 |
| POST | `/roi_selection` | 获取 X100 任务列表（采样点推荐） |
| POST | `/generate_views` | Set Cover 生成最少视场覆盖框 |
| POST | `/analyze_slide` | 玻片级分析（如 cellularity） |
| POST | `/analyze_cell_image` | X100 细胞图像分析（任务模式或单张识别） |

### 图像滤镜 `/img_filter`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/x100_img_filter` | x100 OpenCV 增强滤镜 |
| POST | `/x40_img_filter_pt` | x40 超分辨率滤镜（普通模式） |
| POST | `/x40_img_filter` | x40 超分辨率滤镜（深度学习模式） |

### 常用参数说明

- **smear_type**：涂片类型，`BM`（骨髓）/ `PB`（外周血）/ `CF`（脑脊液）
- **dpi**：分辨率档位，`144750` / `357378` / `714756`（±10% 容差），用于选择对应推理 pipeline
- **target_cell_types**：目标细胞类型，如 `WBC,MEG`、`WBC,RBC,PLAT`

## 典型工作流

### X40 全片扫描

```
create_task → upload_tile（逐块） → update_coordinates
    → check_missing_tiles → get_task_result
```

### X100 细胞识别

```
create_task → upload_tile → update_coordinates
    → roi_selection → analyze_cell_image
```

## 日志

日志写入 `logs/` 目录：

| 文件 | 内容 |
|------|------|
| `app.log` | 应用与业务 INFO/WARNING |
| `error.log` | ERROR 及以上 |
| `access.log` | HTTP 访问摘要 |

采用 RotatingFileHandler 自动轮转，避免单文件无限增长。

## 许可证

本项目采用 [MIT License](LICENSE) 开源。
