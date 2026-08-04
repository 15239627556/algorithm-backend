# project.py
from __future__ import annotations

import sys
from pathlib import Path

# 允许 `python project/smear_project.py` 直接运行（相对导入需要包上下文）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import pickle
from typing import List, Optional

import orjson

from project.layers import Layer


# =========================
# 核心 Project 类
# =========================

class SmearProject:
    """
    表示一次完整的涂片扫描项目：
    - 支持 40x 平扫层
    - 支持 100x 采样层
    - 存储瓦片级与细胞级信息
    - smear_type: 涂片类型（例如 BM, PB, CF）
    - dpi: int 默认创建的 40x 层的分辨率
    和接口的对应关系：
    - /upload_tile  -> add_tile()
    - /get_task_result -> get_cells_in_roi()
    """

    def __init__(self, smear_type: str) -> None:
        self.smear_type = smear_type
        self.layers: List[Layer] = []

    # ---------- 图层管理 ----------
    def add_layer(self, dpi: int) -> Layer:
        """添加一个倍率层，例如新建 100x 采样层。"""
        self.layers.append(Layer(dpi))
        return self.layers[-1]

    def get_layer(self, dpi) -> Optional[Layer]:
        """获取指定倍率层"""
        for layer in self.layers:
            if layer.dpi == dpi:
                return layer
        layer = Layer(dpi)
        self.layers.append(layer)
        return layer

    def list_layers(self) -> List[Layer]:
        """列出所有倍率层"""
        return self.layers

    def save_pickle(self, root_dir: str, task_id: str) -> str:
        """
        使用 pickle 持久化（不推荐跨语言，但简单粗暴）。
        文件名: <root_dir>/<task_id>.smear.pkl
        """
        os.makedirs(root_dir, exist_ok=True)
        path = os.path.join(root_dir, f"{task_id}.smear.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load_pickle(cls, root_dir: str, task_id: str) -> "SmearProject":
        path = os.path.join(root_dir, f"{task_id}.smear.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    def to_dict(self) -> dict:
        return {
            "version": 1,  # 🔥 强烈建议：未来可做数据升级
            "smear_type": self.smear_type,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SmearProject":
        project = cls(
            smear_type=data["smear_type"],
        )
        project.layers = [Layer.from_dict(ld) for ld in data.get("layers", [])]
        return project

    def save_json(self, path: str) -> str:
        """
        使用 JSON 持久化（推荐，可跨版本 / 跨语言）。
        编码为 UTF-8；与标准库 json 产出格式兼容，可由任意 JSON 解析器读取。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(self.to_dict()))
        return "save success"

    @classmethod
    def load_json(cls, path: str) -> "SmearProject":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        data = orjson.loads(path.read_bytes())
        return cls.from_dict(data)

    def save_pickle(self, path: str) -> str:
        """Python 专用快路径（选区等）；原子写入。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        return "save success"

    @classmethod
    def load_pickle(cls, path: str) -> "SmearProject":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"pickle 内容不是 SmearProject: {type(obj)!r}")
        return obj


if __name__ == '__main__':
    import time

    # 以脚本运行时类名为 __main__.SmearProject，pickle 里是 project.smear_project.SmearProject
    from project.smear_project import SmearProject as _SmearProject

    t1 = time.time()
    project = _SmearProject.load_pickle(r'D:\f00a5a56a9a44250880c9014b07ae843.roi.pkl')
    t2 = time.time()
    print(f"load pickle time: {t2 - t1} seconds")
    # layer = project.get_layer(dpi)
    # for tile in layer.iter_tiles():
    #     print(tile.image_uid)
    #     print(f"Tile ({tile.global_x}, {tile.global_y}): {len(tile.cells)} cells")
