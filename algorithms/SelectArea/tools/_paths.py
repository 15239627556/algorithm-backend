"""SelectArea 调试/可视化工具的路径常量。"""
from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
SELECT_AREA_DIR = TOOLS_DIR.parent
ROOT_DIR = SELECT_AREA_DIR.parent.parent
OUTPUT_DIR = SELECT_AREA_DIR / "output"


def setup_sys_path() -> None:
    root = str(ROOT_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
