"""选区（ROI）专用 numpy 存储：预计算全局坐标 + 按 cell_type 索引 + 全图热力图。"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import orjson

from project.tiles import Tile

if TYPE_CHECKING:
    from algorithms.SelectArea.config import BM40Config
    from algorithms.SelectArea.heatmaps import HeatmapGrid
    from project.smear_project import SmearProject

ROI_NPZ_VERSION = 2
DEFAULT_HEATMAP_CELL_SIZE = 224.0

CELL_DTYPE = np.dtype([
    ("xmin", "i4"),
    ("ymin", "i4"),
    ("xmax", "i4"),
    ("ymax", "i4"),
    ("cell_type", "i4"),
    ("class_confidence", "f4"),
    ("bbox_confidence", "f4"),
])


@dataclass
class RoiDataset:
    """选区热路径数据结构：tile 元数据 + 全局坐标细胞表 + 可选预计算全图热力图。"""

    smear_type: str
    dpi: int
    tiles: list[Tile] = field(default_factory=list)
    cells: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=CELL_DTYPE))
    type_index: dict[int, tuple[int, int]] = field(default_factory=dict)
    heatmap_values: Optional[np.ndarray] = None
    heatmap_weights: Optional[np.ndarray] = None
    heatmap_origin_x: float = 0.0
    heatmap_origin_y: float = 0.0
    heatmap_cell_size: float = DEFAULT_HEATMAP_CELL_SIZE
    wbc_cell_matrix: Optional[np.ndarray] = None

    @classmethod
    def from_project(cls, project: SmearProject, dpi: int) -> RoiDataset:
        layer = project.get_layer(dpi)
        if layer is None:
            return cls(smear_type=project.smear_type, dpi=dpi)

        tiles: list[Tile] = []
        cell_rows: list[tuple] = []

        for tile in layer.tiles.values():
            tiles.append(Tile(
                image_uid=tile.image_uid,
                x=tile.x,
                y=tile.y,
                w=tile.w,
                h=tile.h,
                image_path=tile.image_path,
                meta=dict(tile.meta) if tile.meta else {},
                cells=[],
            ))
            tx = 0 if tile.x is None else int(tile.x)
            ty = 0 if tile.y is None else int(tile.y)
            for cell in tile.cells:
                cell_rows.append((
                    cell.cell_xmin + tx,
                    cell.cell_ymin + ty,
                    cell.cell_xmax + tx,
                    cell.cell_ymax + ty,
                    cell.cell_type,
                    cell.class_confidence,
                    cell.bbox_confidence,
                ))

        if cell_rows:
            cells = np.array(cell_rows, dtype=CELL_DTYPE)
            order = np.argsort(cells["cell_type"], kind="stable")
            cells = cells[order]
        else:
            cells = np.empty(0, dtype=CELL_DTYPE)

        type_index = _build_type_index(cells)
        roi = cls(
            smear_type=project.smear_type,
            dpi=dpi,
            tiles=tiles,
            cells=cells,
            type_index=type_index,
        )
        roi.ensure_precomputed_heatmap()
        return roi

    def has_precomputed_heatmap(self, cell_size: float) -> bool:
        return (
            self.heatmap_values is not None
            and self.heatmap_weights is not None
            and abs(self.heatmap_cell_size - cell_size) < 1e-3
        )

    def ensure_precomputed_heatmap(self, cell_size: float = DEFAULT_HEATMAP_CELL_SIZE) -> None:
        """落盘前预计算全图 score 热力图（与 BM40Config 默认 cell_size 对齐）。"""
        if self.has_precomputed_heatmap(cell_size) or not self.tiles:
            return
        from algorithms.SelectArea.config import BM40Config
        from algorithms.SelectArea.heatmaps import build_score_heatmap

        cfg = BM40Config()
        grid = build_score_heatmap(self.tiles, config=cfg)
        self.heatmap_values = grid.values
        self.heatmap_weights = grid.weights
        self.heatmap_origin_x = grid.origin_x
        self.heatmap_origin_y = grid.origin_y
        self.heatmap_cell_size = grid.cell_size
        self._precompute_wbc_cell_matrix(cfg.WBC_cell_type)

    def _precompute_wbc_cell_matrix(self, wbc_cell_type: int) -> None:
        if self.heatmap_values is None:
            return
        from algorithms.SelectArea.heatmaps import HeatmapGrid

        grid = HeatmapGrid(
            origin_x=self.heatmap_origin_x,
            origin_y=self.heatmap_origin_y,
            cell_size=self.heatmap_cell_size,
            values=self.heatmap_values,
            weights=self.heatmap_weights,
        )
        cells = self.cells_xyxy_by_type(wbc_cell_type)
        if cells.size == 0:
            self.wbc_cell_matrix = np.zeros_like(self.heatmap_values, dtype=np.float32)
            return
        rows, cols = self.heatmap_values.shape
        matrix = np.zeros((rows, cols), dtype=np.float32)
        centers = 0.5 * (cells[:, 0:2] + cells[:, 2:4])
        g_cols = ((centers[:, 0] - grid.origin_x) // grid.cell_size).astype(np.int32)
        g_rows = ((centers[:, 1] - grid.origin_y) // grid.cell_size).astype(np.int32)
        in_bounds = (g_rows >= 0) & (g_rows < rows) & (g_cols >= 0) & (g_cols < cols)
        flat_idx = g_rows[in_bounds] * cols + g_cols[in_bounds]
        matrix = np.bincount(flat_idx, minlength=rows * cols).reshape(rows, cols).astype(np.float32)
        self.wbc_cell_matrix = matrix

    def build_cell_matrix(
        self,
        config: BM40Config,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        all_cells_array: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """优先使用预计算 WBC 密度矩阵；局部选区时裁剪。"""
        from algorithms.SelectArea.heatmaps import HeatmapGrid, crop_heatmap_grid

        if (
            self.wbc_cell_matrix is not None
            and bounds is None
            and self.has_precomputed_heatmap(config.cell_size)
        ):
            return self.wbc_cell_matrix

        if all_cells_array is None:
            all_cells_array = self.cells_xyxy_by_type(config.WBC_cell_type)

        if bounds is not None and self.wbc_cell_matrix is not None and self.has_precomputed_heatmap(config.cell_size):
            grid = HeatmapGrid(
                origin_x=self.heatmap_origin_x,
                origin_y=self.heatmap_origin_y,
                cell_size=self.heatmap_cell_size,
                values=self.wbc_cell_matrix,
                weights=np.ones_like(self.wbc_cell_matrix),
            )
            sub = crop_heatmap_grid(grid, bounds)
            return sub.values

        grid = self.build_heatmap_grid(config, bounds=bounds)
        rows, cols = grid.values.shape
        matrix = np.zeros((rows, cols), dtype=np.float32)
        if all_cells_array.size == 0:
            return matrix
        centers = 0.5 * (all_cells_array[:, 0:2] + all_cells_array[:, 2:4])
        g_cols = ((centers[:, 0] - grid.origin_x) // grid.cell_size).astype(np.int32)
        g_rows = ((centers[:, 1] - grid.origin_y) // grid.cell_size).astype(np.int32)
        in_bounds = (g_rows >= 0) & (g_rows < rows) & (g_cols >= 0) & (g_cols < cols)
        flat_idx = g_rows[in_bounds] * cols + g_cols[in_bounds]
        return np.bincount(flat_idx, minlength=rows * cols).reshape(rows, cols).astype(np.float32)

    def build_heatmap_grid(
        self,
        config: BM40Config,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> HeatmapGrid:
        """优先使用预计算热力图；user_choice_area 时裁剪子网格。"""
        from algorithms.SelectArea.heatmaps import HeatmapGrid, build_score_heatmap, crop_heatmap_grid

        if not self.has_precomputed_heatmap(config.cell_size):
            tiles = self.tiles
            if bounds is not None:
                from algorithms.SelectArea.heatmaps import filter_tiles_by_bounds
                tiles = filter_tiles_by_bounds(tiles, bounds)
            return build_score_heatmap(tiles, config=config, bounds=bounds)

        full = HeatmapGrid(
            origin_x=self.heatmap_origin_x,
            origin_y=self.heatmap_origin_y,
            cell_size=self.heatmap_cell_size,
            values=self.heatmap_values,
            weights=self.heatmap_weights,
        )
        if bounds is None:
            return full
        return crop_heatmap_grid(full, bounds)

    def cells_xyxy_by_type(self, cell_type: int) -> np.ndarray:
        """返回指定 cell_type 的全局坐标 (N, 4) float32。"""
        bounds = self.type_index.get(int(cell_type))
        if not bounds:
            return np.empty((0, 4), dtype=np.float32)
        start, end = bounds
        if start >= end:
            return np.empty((0, 4), dtype=np.float32)
        sub = self.cells[start:end]
        return np.column_stack([
            sub["xmin"], sub["ymin"], sub["xmax"], sub["ymax"],
        ]).astype(np.float32, copy=False)

    def cells_xyxy_list_by_type(self, cell_type: int) -> list[list[float]]:
        """MEG 等仍使用 list 的路径。"""
        arr = self.cells_xyxy_by_type(cell_type)
        if arr.size == 0:
            return []
        return arr.tolist()

    def cells_in_roi(
        self,
        xmin: int | float,
        ymin: int | float,
        xmax: int | float,
        ymax: int | float,
    ) -> np.ndarray:
        """向量化 ROI 相交过滤，返回 cells 结构化数组子集（已是全局坐标）。"""
        cells = self.cells
        if cells.size == 0:
            return cells
        mask = np.ones(cells.shape[0], dtype=bool)
        if xmin is not None:
            mask &= cells["xmax"] >= int(xmin)
        if ymin is not None:
            mask &= cells["ymax"] >= int(ymin)
        if xmax is not None and not math.isinf(float(xmax)):
            mask &= cells["xmin"] <= int(xmax)
        if ymax is not None and not math.isinf(float(ymax)):
            mask &= cells["ymin"] <= int(ymax)
        return cells[mask]

    def cell_records_to_dicts(self, records: np.ndarray) -> list[dict]:
        """结构化 cell 记录 → API 字段（仅转换 records，避免全量 materialize）。"""
        if records.size == 0:
            return []
        out: list[dict] = []
        append = out.append
        for rec in records:
            ct = int(rec["cell_type"])
            append({
                "cell_xmin": int(rec["xmin"]),
                "cell_ymin": int(rec["ymin"]),
                "cell_xmax": int(rec["xmax"]),
                "cell_ymax": int(rec["ymax"]),
                "cell_type": ct,
                "cell_type_name": _cell_type_name_zh(ct),
                "class_confidence": float(rec["class_confidence"]),
                "bbox_confidence": float(rec["bbox_confidence"]),
                "extra": {},
            })
        return out

    def save(self, path: str) -> str:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        tmp = path_obj.with_suffix(f".{os.getpid()}.tmp.npz")

        self.ensure_precomputed_heatmap()

        tiles_meta = [
            {
                "image_uid": t.image_uid,
                "x": t.x,
                "y": t.y,
                "w": t.w,
                "h": t.h,
                "image_path": t.image_path,
                "meta": t.meta,
            }
            for t in self.tiles
        ]
        type_types, type_starts, type_ends = _type_index_to_arrays(self.type_index)

        payload = dict(
            version=np.array([ROI_NPZ_VERSION], dtype=np.int32),
            dpi=np.array([self.dpi], dtype=np.int32),
            smear_type=np.asarray(self.smear_type),
            cells=self.cells,
            type_types=type_types,
            type_starts=type_starts,
            type_ends=type_ends,
            tiles_meta=np.frombuffer(orjson.dumps(tiles_meta), dtype=np.uint8),
        )
        if self.heatmap_values is not None:
            payload.update(
                heatmap_values=self.heatmap_values,
                heatmap_weights=self.heatmap_weights,
                heatmap_origin_x=np.array([self.heatmap_origin_x], dtype=np.float64),
                heatmap_origin_y=np.array([self.heatmap_origin_y], dtype=np.float64),
                heatmap_cell_size=np.array([self.heatmap_cell_size], dtype=np.float64),
            )
        if self.wbc_cell_matrix is not None:
            payload["wbc_cell_matrix"] = self.wbc_cell_matrix

        np.savez_compressed(tmp, **payload)
        os.replace(tmp, path_obj)
        return "save success"

    @classmethod
    def load(cls, path: str) -> RoiDataset:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(path)

        with np.load(path_obj, allow_pickle=False) as data:
            version = int(data["version"][0])
            if version not in (1, ROI_NPZ_VERSION):
                raise ValueError(f"unsupported roi npz version: {version}")

            smear_type = str(data["smear_type"].item())
            dpi = int(data["dpi"][0])
            cells = data["cells"]
            type_index = _arrays_to_type_index(
                data["type_types"], data["type_starts"], data["type_ends"],
            )
            tiles_meta = orjson.loads(data["tiles_meta"].tobytes())

            heatmap_values = data["heatmap_values"] if "heatmap_values" in data else None
            heatmap_weights = data["heatmap_weights"] if "heatmap_weights" in data else None
            heatmap_origin_x = float(data["heatmap_origin_x"][0]) if "heatmap_origin_x" in data else 0.0
            heatmap_origin_y = float(data["heatmap_origin_y"][0]) if "heatmap_origin_y" in data else 0.0
            heatmap_cell_size = (
                float(data["heatmap_cell_size"][0])
                if "heatmap_cell_size" in data
                else DEFAULT_HEATMAP_CELL_SIZE
            )
            wbc_cell_matrix = data["wbc_cell_matrix"] if "wbc_cell_matrix" in data else None

        tiles = [
            Tile(
                image_uid=item["image_uid"],
                x=item.get("x"),
                y=item.get("y"),
                w=item["w"],
                h=item["h"],
                image_path=item.get("image_path"),
                meta=item.get("meta") or {},
                cells=[],
            )
            for item in tiles_meta
        ]
        return cls(
            smear_type=smear_type,
            dpi=dpi,
            tiles=tiles,
            cells=cells,
            type_index=type_index,
            heatmap_values=heatmap_values,
            heatmap_weights=heatmap_weights,
            heatmap_origin_x=heatmap_origin_x,
            heatmap_origin_y=heatmap_origin_y,
            heatmap_cell_size=heatmap_cell_size,
            wbc_cell_matrix=wbc_cell_matrix,
        )


def _build_type_index(cells: np.ndarray) -> dict[int, tuple[int, int]]:
    if cells.size == 0:
        return {}
    type_index: dict[int, tuple[int, int]] = {}
    for cell_type in np.unique(cells["cell_type"]):
        ct = int(cell_type)
        start = int(np.searchsorted(cells["cell_type"], ct, side="left"))
        end = int(np.searchsorted(cells["cell_type"], ct, side="right"))
        type_index[ct] = (start, end)
    return type_index


def _type_index_to_arrays(type_index: dict[int, tuple[int, int]]):
    if not type_index:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, empty
    types = sorted(type_index)
    starts = np.array([type_index[t][0] for t in types], dtype=np.int32)
    ends = np.array([type_index[t][1] for t in types], dtype=np.int32)
    return np.array(types, dtype=np.int32), starts, ends


def _arrays_to_type_index(
    type_types: np.ndarray,
    type_starts: np.ndarray,
    type_ends: np.ndarray,
) -> dict[int, tuple[int, int]]:
    return {
        int(ct): (int(start), int(end))
        for ct, start, end in zip(type_types, type_starts, type_ends)
    }


_CELL_TYPE_NAME_CACHE: dict[int, str] | None = None


def _cell_type_name_zh(cell_type: int) -> str:
    global _CELL_TYPE_NAME_CACHE
    if _CELL_TYPE_NAME_CACHE is None:
        from backend.tools.MESSAGE_DICT import (
            CELL_TYPE_CSF,
            CELL_TYPES_MEG,
            CELL_TYPES_X100,
            CELL_TYPES_X40,
        )
        cache: dict[int, str] = {}
        for mapping in (CELL_TYPES_X40, CELL_TYPES_X100, CELL_TYPES_MEG, CELL_TYPE_CSF):
            for key, val in mapping.items():
                if isinstance(val, (tuple, list)) and len(val) >= 2:
                    cache[int(key)] = str(val[1])
        _CELL_TYPE_NAME_CACHE = cache
    return _CELL_TYPE_NAME_CACHE.get(int(cell_type), "")
