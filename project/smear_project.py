# project.py
from __future__ import annotations

import os
from pathlib import Path
import json
import pickle
from typing import List, Optional

from .layers import Layer


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
            dpi=data["layers"][0]["dpi"] if data.get("layers") else 0,
        )
        project.layers = [Layer.from_dict(ld) for ld in data.get("layers", [])]
        return project

    def save_json(self, path: str) -> str:
        """
        使用 JSON 持久化（推荐，可跨版本 / 跨语言）。
        文件名: <root_dir>/<task_id>.smear.json
        """
        path = Path(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return "save success"

    @classmethod
    def load_json(cls, path: str) -> "SmearProject":
        path = Path(path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


if __name__ == '__main__':
    dpi = 40
    project = SmearProject.load_json('../backend/uploads/b2364dafea904bea8fb978bfd218e1b9.json')
    print(project.list_layers())
    # layer = project.get_layer(dpi)
    # for tile in layer.iter_tiles():
    #     print(tile.image_uid)
    #     print(f"Tile ({tile.global_x}, {tile.global_y}): {len(tile.cells)} cells")
