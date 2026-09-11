# Algorithm Backend

血液涂片（骨髓 / 外周血 / 脑脊液等）智能分析算法后端服务，基于 **FastAPI** 构建。业务层负责任务编排、ROI 规划与结果聚合；推理通过 **NVIDIA Triton Inference Server**（模型加载 / gRPC）与 **multi_pipeline HTTP 服务**（图像解码 + pipeline 推理）完成。

当前版本：**v52**（见 `config.py` 中 `sufa_version`）

## 功能概览

| 模块 | 说明 |
|------|------|
| **X40 任务模式** | 创建扫描任务、逐块上传 tile、更新坐标、检查完整性、获取 ROI 内细胞检测结果 |
| **X100 细胞分析** | 任务模式（task_id + 坐标裁剪）或单张识别；支持 WBC / MEG / RBC / PLAT 等组合 |
| **ROI 任务规划** | Set Cover 生成最少视场覆盖；WBC / MEG / RBC 等 X100 采样点推荐 |
| **玻片分析** | 全片级指标，当前支持 `cellularity`（骨髓增生程度） |
| **全局图分析** | 单张原图 ROI 选区（`roi`）与头部朝向（`dir`）识别 |
| **模型管控** | 按 DPI + 涂片类型 + 目标细胞类型按需加载 / 卸载 Triton 模型 |
| **图像滤镜** | x100 / x40 OpenCV 增强；x40 深度学习超分辨率 |
| **后处理** | 跨 tile 去重、边缘不完整细胞过滤、高 DPI 细长靠边细胞过滤等 |

## 技术栈

- **Web 框架**：FastAPI + Uvicorn
- **推理**：Triton Inference Server（HTTP 模型仓库 + gRPC）+ multi_pipeline HTTP
- **深度学习**：PyTorch / TorchVision（去重 NMS 等本地算法）
- **图像处理**：OpenCV、Pillow、NumPy
- **优化算法**：PySCIPOpt（Set Cover）、Pandas

## 架构说明

```
客户端
  │
  ▼
FastAPI（app.py）
  ├── task_service        任务状态、tile 合并、ROI 缓存、玻片分析
  ├── triton_client       调用 multi_pipeline / Triton
  └── model_control       模型解析（MODEL_TABLE）、加载 / 卸载、显存预算
  │
  ├── Triton HTTP/gRPC    模型 load/unload、部分 gRPC 推理
  └── multi_pipeline      POST /{dpi}/infer、/global/infer、滤镜 infer 等
```

**multi_pipeline 主要路由（本服务会调用）：**

| 路由 | 用途 |
|------|------|
| `POST /147246/infer` 等 | 按 DPI 的细胞检测 pipeline |
| `POST /147246/infer_ca` | 骨髓增生程度（cellularity）专用 |
| `POST /global/infer` | 全局图 ROI / 头部朝向 |
| `POST /opencv_enhance/infer` | OpenCV 滤镜（裸流） |
| `POST /image_enhance/infer` | 深度学习超分滤镜（裸流） |

模型清单与 DPI 区间定义见 `backend/tools/MESSAGE_DICT.py` 中 `MODEL_TABLE`。

## 项目结构

```
algorithm-backend/
├── app.py                      # FastAPI 入口、日志、线程池、路由挂载
├── config.py                   # Triton 端点、端口、开关、版本号
├── gunicorn.conf.py            # 可选 Gunicorn + gevent 配置
├── requirements.txt
├── backend/
│   ├── routes/
│   │   ├── task.py             # 涂片分析 / 任务 / 模型 / 全局图 API
│   │   └── ImgFilter.py        # 图像滤镜 API
│   ├── services/
│   │   └── task_service.py     # 任务与玻片分析业务逻辑
│   └── tools/
│       ├── triton_client.py    # multi_pipeline / Triton 客户端
│       ├── model_control.py    # 模型加载与解析
│       ├── MESSAGE_DICT.py     # 返回码、细胞类型、MODEL_TABLE
│       ├── x100_image_infer.py # X100 单张 / 任务模式推理
│       └── combo_validator.py  # DPI + smear_type + 细胞类型组合校验
├── project/                    # Layer / Tile / Cell 数据模型与 ROI 存储
├── algorithms/SelectArea/      # ROI 选取、Set Cover、跨 tile 去重
├── uploads/                    # 任务数据与推理结果
└── logs/                       # app / error / access 日志
```

## 环境要求

- Python 3.10+
- 可访问的 Triton Inference Server 与 multi_pipeline 服务
- GPU 由 Triton / multi_pipeline 侧提供；本服务以 CPU + HTTP 为主

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 Triton

编辑 `config.py` 中的 `TRITON_IPS`，填写 Triton / multi_pipeline 可达的主机名或 IP。各端点默认端口：

| 服务 | 默认端口 | 环境变量 |
|------|----------|----------|
| Triton HTTP | 8000 | `TRITON_HTTP_PORT` |
| Triton gRPC | 8001 | `TRITON_GRPC_PORT` |
| Triton Metrics | 8002 | `TRITON_METRICS_PORT` |
| multi_pipeline | 9000 | `MULTI_PIPELINE_PORT` |

### 3. 启动服务

**开发 / 直接运行（默认 2 worker）：**

```bash
python app.py
```

**Uvicorn 命令行：**

```bash
uvicorn app:app --host 0.0.0.0 --port 3090 --workers 2
```

默认监听 `0.0.0.0:3090`。`config.is_doc=True` 时 Swagger 在 `/`，ReDoc 在 `/redoc`。

### 4. 健康检查

```bash
curl http://localhost:3090/health
# {"status":"ok"}
```

## 环境变量

### 应用

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `APP_HOST` / `FLASK_HOST` | `0.0.0.0` | 监听地址 |
| `APP_PORT` / `FLASK_PORT` | `3090` | 监听端口 |
| `THREAD_POOL_SIZE` | `256` | anyio 线程池（阻塞推理 / IO 并发上限） |

### Triton / 推理

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TRITON_GPU_ID` | `0` | 定点使用的端点下标（对应 `TRITON_IPS` 顺序） |
| `TRITON_ROUND_ROBIN` | `0` | `1` 时多端点轮询发请求 |
| `TRITON_HTTP_PORT` / `TRITON_GRPC_PORT` / `MULTI_PIPELINE_PORT` | 见上 | 覆盖默认端口 |
| `TRITON_LOAD_TIMEOUT` | `600` | Triton 模型 load 超时（秒） |
| `PIPELINE_HTTP_TIMEOUT_S` | `600` | multi_pipeline 推理读超时（秒） |
| `PIPELINE_HTTP_CONNECT_TIMEOUT_S` | `10` | multi_pipeline 连接超时（秒） |
| `PIPELINE_147246_INFER_URL` | （空） | 可选，覆盖 147246 infer 完整 URL |
| `MAX_MEMORY` | `11` | Triton 侧显存预算（GB） |
| `RESERVED_MEMORY` | `1` | 预留显存（GB） |

### 业务开关

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CAMERA` | `flir` | 相机类型；`flir` 时使用 FLIR 专用 WBC 分类器 |
| `RBC_SWITCH` | 关闭 | `1`/`true` 等开启红细胞相关逻辑 |
| `PLT_SWITCH` | 关闭 | 血小板相关逻辑 |
| `CELLULARITY_FILE_PATH_PREFIX` | `/mnt/d/data/result` | 增生分析原图目录前缀 |
| `UPDATE_COORDINATES_ASYNC_FINISH` | `1` | 坐标更新后后台异步合并 / 去重 |
| `ROI_DATASET_CACHE_TTL_SEC` | 见代码 | ROI 数据集内存缓存 TTL |

## 统一返回格式

多数业务接口使用 JSON 包装：

| 字段 | 说明 |
|------|------|
| `ret_code` | `200` 成功；`201` 客户端错误；`100`–`105` 任务状态；`202` ROI 错误 |
| `ret_desc` | 返回码描述 |
| `result` | 业务数据（分析类接口） |
| `reason` | 部分接口在错误时附加说明 |

滤镜接口（`/img_filter/*`）成功时直接返回增强后的图片二进制流，失败时返回 JSON `{ "message": "..." }`。

## API 接口

### 涂片分析 `/api/v1/smear_analysis`

#### 任务与检测

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/create_task` | 创建 X40 扫描任务 |
| POST | `/upload_tile` | 上传拼图块（multipart 或二进制流） |
| POST | `/update_coordinates` | 更新拼图块全局坐标 |
| POST | `/check_missing_tiles` | 检查 tile 是否上传完毕 |
| POST | `/check_task_status` | 获取任务状态 |
| POST | `/get_task_result` | 分页获取 ROI 内细胞检测结果 |

**`upload_tile` 两种上传方式：**

1. **multipart/form-data**：`task_id`、`row_index`、`col_index`、`tile_image`
2. **二进制流**：Body 为图像字节，Query 传 `task_id`、`row_index`、`col_index`，`Content-Type: application/octet-stream`

#### ROI 与 X100

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/roi_selection` | 获取 X100 任务列表（采样点推荐） |
| POST | `/generate_views` | Set Cover 生成最少视场覆盖框 |
| POST | `/analyze_cell_image` | X100 细胞图像分析（任务模式或单张识别） |

#### 玻片 / 全局图分析

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/analyze_slide` | 玻片级分析（JSON Body） |
| POST | `/global_image_analysis` | 全局图分析（multipart） |

**`/analyze_slide`**

- Body：`task_id`（必填）、`analyze_names`（必填，如 `["cellularity"]`）
- 字段映射：`cellularity` → 增生程度（`red_pixel_count / wbc_pixel_count`）
- 成功响应示例：

```json
{
  "ret_code": 200,
  "ret_desc": "API success",
  "result": {
    "cellularity": 0.42,
    "wbc_pixel_count": 1000,
    "red_pixel_count": 420
  }
}
```

**`/global_image_analysis`**

- Form：`image_file`（原图）、`smear_type`（`BM` / `PB`）、`task`（`roi` / `dir` / `roi,dir`，逗号或分号分隔）
- Header（可选）：`x-client-seq`（仅日志）
- 字段映射：`roi` → ROI 选区；`dir` → 头部朝向
- 内部模型：`GLOBAL-IMAGE-ANALYSIS`、`GLOBAL-IMAGE-HEAD-DIR`；转发 multi_pipeline `POST /global/infer`（文件字段名为 `image`）
- 成功响应示例：

```json
{
  "ret_code": 200,
  "ret_desc": "API success",
  "result": { }
}
```

`result` 内容由 pipeline 返回，含 `roi` / `dir` 等推理字段（已过滤 timing、models 等内部字段）。

#### 模型管控

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/load_models` | 按 DPI + smear_type + target_cell_types 加载模型 |
| POST | `/unload_models` | 卸载对应组合模型 |

Body 字段：`smear_type`、`dpi`、`target_cell_types`；可选 `gpu_id`、`all_gpus`。

### 图像滤镜 `/img_filter`

| 方法 | 路径 | 说明 | multi_pipeline |
|------|------|------|----------------|
| POST | `/x100_img_filter` | x100 OpenCV 增强 | `/opencv_enhance/infer` |
| POST | `/x40_img_filter_pt` | x40 超分（普通模式） | `/opencv_enhance/infer` |
| POST | `/x40_img_filter` | x40 超分（深度学习） | `/image_enhance/infer` |

请求均为 multipart，字段 `image_file`；响应为 JPEG/PNG 图片流。

### 其他

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 服务健康检查 |
| GET | `/uploads/...` | 静态文件（任务上传目录） |

## 常用参数

| 参数 | 说明 |
|------|------|
| **smear_type** | 涂片类型：`BM`（骨髓）、`PB`（外周血）、`CF` / `CSF`（脑脊液） |
| **dpi** | 分辨率档位：`144750` / `357378` / `714756`（±10% 容差映射到实际 pipeline DPI） |
| **target_cell_types** | 目标细胞，如 `WBC,MEG`、`WBC,RBC,PLAT` |
| **task_type**（roi_selection） | 如 `BM_WBC`、`BM_MEG`、`BM_RBC` 等 |

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

### 骨髓增生分析

```
create_task → … → roi_selection（得到最大任务框）
    → analyze_slide（analyze_names: ["cellularity"]）
```

### 全局图 ROI / 朝向

```
global_image_analysis（image_file + smear_type + task）
```

## 日志

日志目录 `logs/`：

| 文件 | 内容 |
|------|------|
| `app.log` | 应用与 `backend.*` INFO/WARNING |
| `error.log` | ERROR 及以上 |
| `access.log` | HTTP 访问一行摘要 |

RotatingFileHandler 自动轮转。`stdout` 会 tee 到 `app.log`。

## 许可证

本项目采用 [MIT License](LICENSE) 开源。
