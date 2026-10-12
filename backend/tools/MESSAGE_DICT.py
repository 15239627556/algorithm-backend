# -*- coding: utf-8 -*-
import json
import logging
import os
from enum import Enum

_logger = logging.getLogger(__name__)


class RetCode(Enum):
    """
    返回码枚举（数值）
    """
    TASK_FINISHED = 100  # 任务已完成
    TASK_RUNNING = 101  # 任务进行中
    TASK_TIMEOUT = 102  # 任务超时
    IMAGE_ALREADY_UPLOADED = 103  # 图片已上传，请勿重复上传
    IMAGE_DAMAGED = 104  # 图片损坏，请检查上传图片是否完整
    TASK_IN_PROGRESS = 105  # 当前有任务正在进行

    API_SUCCESS = 200  # 接口调用成功
    CLIENT_ERROR = 201  # 参数错误、文件类型错误、相关条目不存在等客户端错误
    ROI_ERROR = 202  # ROI选区错误


class RetDesc(Enum):
    """
    返回码枚举（描述）
    """
    TASK_FINISHED = "Task completed"
    TASK_RUNNING = "Task in progress"
    TASK_TIMEOUT = "Task timeout"
    IMAGE_ALREADY_UPLOADED = "Image already uploaded, do not upload again"
    IMAGE_DAMAGED = "Image damaged, please check if uploaded image is complete"
    TASK_IN_PROGRESS = "Task in progress"

    API_SUCCESS = "API success"
    CLIENT_ERROR = "Client error: invalid params, file type error, or entry not found"
    ROI_ERROR = "ROI selection error"


class TaskType(Enum):
    """
    任务类型枚举
    """
    BM_WBC = 0
    PB_WBC = 0
    CF_WBC = 0
    BM_MEG = 1
    BM_RBC = 2
    PB_RBC = 2


CELL_TYPES_X40 = {
    100000: ("Unclassified_WBC", "未分类白细胞"),
    100001: ("Unclassified_MEG", "未分类巨核细胞"),
    100002: ("Unclassified_RBC", "未分类红细胞"),
    100003: ("Broken_Cell", "破碎细胞"),
    100004: ("Unclassified_PLT", "未分类血小板"),
    100005: ("RBC", "已分类红细胞"),
    100006: ("PLT", "已分类血小板"),
    100007: ("CSF", "未分类脑脊液细胞"),
}

CELL_TYPES_X100 = {
    200000: ("Myeloblast", "原始粒细胞"),
    200001: ("Promyelocyte", "早幼粒细胞"),
    200002: ("Neutrophilic_Myelocyte", "中性中幼粒细胞"),
    200003: ("Metamyelocyte", "中性晚幼粒细胞"),
    200004: ("Band_Neutrophil", "中性杆状核粒细胞"),
    200005: ("Segmented_Neutrophil", "中性分叶核粒细胞"),
    200006: ("Eosinophilic_Myelocyte", "嗜酸性中幼粒细胞"),
    200007: ("Eosinophilic_Metamyelocyte", "嗜酸性晚幼粒细胞"),
    200008: ("Band_Eosinophil", "嗜酸性杆状核粒细胞"),
    200009: ("Segmented_Eosinophil", "嗜酸性分叶核粒细胞"),
    200010: ("Basophil", "嗜碱性粒细胞"),
    200011: ("Proerythroblast", "原始红细胞"),
    200012: ("Early_Erythroblast", "早幼红细胞"),
    200013: ("Intermediate_Erythroblast", "中幼红细胞"),
    200014: ("Late_Erythroblast", "晚幼红细胞"),
    200015: ("Early_Megaloblastic_Erythroblast", "早巨幼红细胞"),
    200016: ("Intermediate_Megaloblastic_Erythroblast", "中巨幼红细胞"),
    200017: ("Late_Megaloblastic_Erythroblast", "晚巨幼红细胞"),
    200018: ("Lymphoblast", "原始淋巴细胞"),
    200019: ("Prolymphocyte", "幼稚淋巴细胞"),
    200020: ("Mature_Lymphocyte", "成熟淋巴细胞"),
    200021: ("Reactive_Lymphocyte", "反应性异型淋巴细胞"),
    200022: ("Monoblast", "原始单核细胞"),
    200023: ("Promonocyte", "幼稚单核细胞"),
    200024: ("Monocyte", "单核细胞"),
    200025: ("Plasmablast", "原始浆细胞"),
    200026: ("Immature_Plasma_Cell", "幼稚浆细胞"),
    200027: ("Mature_Plasma_Cell", "成熟浆细胞"),
    200028: ("Histiocyte", "组织细胞"),
    200029: ("Broken_Cell", "破碎细胞及杂质"),
    200030: ("Phagocyte", "吞噬细胞"),
    200031: ("Mast_Cell", "肥大细胞"),
    200032: ("Erythrocyte", "成熟红细胞"),
    200033: ("Mitotic_Cell", "分裂相"),
    200034: ("Platelet", "血小板"),
}

CELL_TYPES_MEG = {
    300000: ("Unclassified_MEG", "颗粒型巨核细胞"),
    300001: ("Unclassified_RBC", "产板型巨核细胞"),
    300002: ("Broken_Cell", "巨核细胞裸核"),
    300003: ("Broken_Cell", "幼稚型巨核细胞"),
    300004: ("Broken_Cell", "非巨核细胞"),
    300005: ("Broken_Cell", "分类不明/无法分类巨核细胞"),
}

CELL_TYPE_CSF = {
    400000: ("Neutrophil", "中性粒细胞"),
    400001: ("Monocyte", "单核细胞"),
    400002: ("Macrophage", "吞噬细胞"),
    400003: ("Basophil", "嗜碱性粒细胞"),
    400004: ("Eosinophil", "嗜酸性粒细胞"),
    400005: ("Plasma_Cell", "浆细胞"),
    400006: ("Mature_Lymphocyte", "成熟淋巴细胞"),
    400007: ("Broken_Cell", "破碎细胞"),
    400008: ("Tumor_Cell", "肿瘤细胞"),
    400009: ("Cryptococcus", "隐球菌"),
    400010: ("Leukemic_Cell", "白血病细胞"),
    400011: ("Lymphoma_Cell", "淋巴瘤细胞"),
}

CELL_TYPE_RBC = {
    500000: ("RBC", "正常成熟红细胞"),
    500003: ("ERBC", "球形红细胞"),
    500004: ("ERBC", "椭圆形红细胞"),
    500005: ("ERBC", "靶形红细胞"),
    500006: ("ERBC", "镰状红细胞"),
    500007: ("ERBC", "泪滴形红细胞"),
    500008: ("ERBC", "口形红细胞"),
    500009: ("ERBC", "裂红细胞"),
    500010: ("ERBC", "棘形红细胞"),
    500011: ("ERBC", "皱缩红细胞"),
    500012: ("ERBC", "咬痕红细胞"),
    500013: ("ERBC", "水泡红细胞"),
    500014: ("ERBC", "盔形红细胞"),
    500015: ("ERBC", "新月形红细胞"),
    500017: ("ERBC", "不规则红细胞"),
    500019: ("ERBC", "大红细胞"),
    500020: ("ERBC", "小红细胞"),
    500021: ("ERBC", "巨大红细胞"),
    500023: ("ERBC", "低色素"),
    500024: ("ERBC", "高色素"),
    500025: ("ERBC", "啫多色性红细胞"),
    500027: ("ERBC", "啫碱性点彩红细胞"),
    500028: ("ERBC", "豪-乔小体"),
    500029: ("ERBC", "Heinz小体"),
    500030: ("ERBC", "卡波环"),
    500031: ("ERBC", "Pappenheimer小体"),
    500032: ("ERBC", "寄生虫"),
    500034: ("ERBC", "缗钱状红细胞"),
    500035: ("ERBC", "凝集红细胞"),
}

CELL_TYPE_PLT = {
    600000: ("PLAT", "正常成熟血小板"),
    600001: ("EPLAT", "片状"),
    600002: ("EPLAT", "小簇"),
    600003: ("EPLAT", "大簇"),
    600004: ("EFC", "杂质"),
    600005: ("EPLAT", "畸形血小板"),
    600006: ("EPLAT", "颗粒减少血小板"),
    600007: ("EPLAT", "灰色血小板"),
    600008: ("EPLAT", "大血小板"),
    600009: ("EPLAT", "小血小板"),
    600010: ("EPLAT", "巨大血小板"),
}

# 根据细胞ID返回计数类别
# 不参与计数：Excluded from the count
# 有核细胞：Nucleated_Cell
# 巨核细胞：Megakaryocyte
def get_counting_cell_type(cell_id, smear_type="BM"):
    if cell_id in CELL_TYPES_X40:
        if cell_id == 100000:
            return "WBC"
        elif cell_id == 100001:
            return "MEG"
        return "EFC"
    elif cell_id in CELL_TYPES_X100:
        if smear_type == "BM":
            if cell_id in list(range(200000, 200028)):
                return "WBC"
            else:
                return "EFC"
        elif smear_type == "PB":
            if cell_id in list(range(200000, 200011)) + list(range(200018, 200028)):
                return "WBC"
            else:
                return "EFC"
    elif cell_id in CELL_TYPES_MEG:
        if cell_id in list(range(300000, 300004)):
            return "MEG"
        else:
            return "EFC"
    elif cell_id in CELL_TYPE_CSF:
        if cell_id == 400007:
            return 'EFC'
        return "WBC"
    elif cell_id in CELL_TYPE_RBC:
        return CELL_TYPE_RBC[cell_id][0]
    elif cell_id in CELL_TYPE_PLT:
        return CELL_TYPE_PLT[cell_id][0]
    else:
        return "EFC"

# 目前开放的DPI
dpi_list = [40, 50, 100]
# 允许上传的图片格式
allow_extensions = ['jpg', 'jpeg', 'gif', 'png']

# =============================================================================
# 模型目录只来自仓库根目录 manifest.json 的 models，不再在这里手写一份。
# dpi 为空：dpi_range 视为不限制；actual_dpi 取 depends_on 里第一个有 dpi 的模型。
# dpi_range 为 [0, 极大值]、max_src 为 9999999 这类占位，转换成 None（不限制）。
# output: -CLS 分类，-DET 定位，-CSCORE 评分；roi/dir/cellularity 为 result；其余有 dpi 的算定位。
# task=cscore 的 targets 为不限制；显存用 memory-usage-gb。
# =============================================================================
_OUTPUT_KIND = {
    "bboxes": "detection",
    "tops": "classification",
    "scores": "score",
    "result": "score",
}

DEFAULT_MODEL_VRAM_GB = 1.5
DPI_NOT_SUITABLE = "DPI不合适"

# manifest 里用大数表示不限制，避免写进 MODEL_TABLE 后把切块上限抬到假尺寸。
_UNLIMITED_SRC_PX = 1_000_000
_UNLIMITED_DPI = 1_000_000_000
_UNRESTRICTED_TASKS = frozenset({"cscore"})
_TASK_TARGET = {
    "wbc": "WBC",
    "meg": "MEG",
    "rbc": "RBC",
    "plat": "PLAT",
    "cellularity": "CELLULARITY",
    "roi": "roi",
    "dir": "dir",
}

MODEL_TABLE: list[dict] = []


def _default_manifest_path() -> str:
    # backend/tools/MESSAGE_DICT.py → 仓库根目录/manifest.json
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, "manifest.json")


def _optional_int(raw) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _output_of(name: str, tasks: list[str], has_own_dpi: bool) -> str:
    upper = name.upper()
    task_keys = {str(task).strip().lower() for task in tasks}
    if upper.endswith("-CLS"):
        return "tops"
    if upper.endswith("-DET"):
        return "bboxes"
    if upper.endswith("-CSCORE") or "cscore" in task_keys:
        return "scores"
    if task_keys & {"roi", "dir", "cellularity"} or upper.startswith("GLOBAL"):
        return "result"
    # CSF-35000-WBC 这类定位模型没有 -DET 后缀，但有自己的 dpi。
    if has_own_dpi:
        return "bboxes"
    return "result"


def _targets_of(tasks: list[str]) -> str:
    labels: list[str] = []
    for task in tasks:
        key = str(task).strip().lower()
        if not key or key in _UNRESTRICTED_TASKS:
            continue
        labels.append(_TASK_TARGET.get(key, key.upper()))
    if not labels:
        return "不限制"
    return "/".join(labels)


def _smear_types_of(slide_type) -> str:
    if isinstance(slide_type, str):
        parts = [part.strip() for part in slide_type.split("/") if part.strip()]
    elif isinstance(slide_type, list):
        parts = [str(item).strip() for item in slide_type if str(item).strip()]
    else:
        parts = []
    return "/".join(parts)


def _src_limit(raw) -> int | None:
    value = _optional_int(raw)
    if value is None or value >= _UNLIMITED_SRC_PX:
        return None
    return value


def _dpi_by_name(models: list) -> dict[str, int]:
    found: dict[str, int] = {}
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        dpi = _optional_int(item.get("dpi"))
        if name and dpi:
            found[name] = dpi
    return found


def _actual_dpi_of(item: dict, dpi_by_name: dict[str, int]) -> int | None:
    own = _optional_int(item.get("dpi"))
    if own:
        return own
    for dep in item.get("depends_on") or []:
        dep_name = str(dep).strip()
        if dep_name in dpi_by_name:
            return dpi_by_name[dep_name]
    return None


def _dpi_range_of(item: dict) -> tuple[int, int] | None:
    # 没有自己的 dpi（分类、全局图）不参与档位区间，避免 [0, 极大值] 冲掉定位模型的范围。
    if not _optional_int(item.get("dpi")):
        return None
    raw = item.get("dpi_range")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    low = _optional_int(raw[0])
    high = _optional_int(raw[1])
    if low is None or high is None:
        return None
    if low <= 0 and high >= _UNLIMITED_DPI:
        return None
    return (low, high)


def _vram_of(item: dict) -> float:
    raw = item.get("memory-usage-gb")
    try:
        if raw is None or raw == "":
            return DEFAULT_MODEL_VRAM_GB
        return float(raw)
    except (TypeError, ValueError):
        _logger.warning("Invalid memory-usage-gb for %s: %r", item.get("name"), raw)
        return DEFAULT_MODEL_VRAM_GB


def _rows_from_manifest_models(models: list) -> list[dict]:
    dpi_by_name = _dpi_by_name(models)
    rows: list[dict] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        tasks = item.get("task") or []
        if isinstance(tasks, str):
            tasks = [tasks]
        rows.append(
            {
                "name": name,
                "dpi_range": _dpi_range_of(item),
                "actual_dpi": _actual_dpi_of(item, dpi_by_name),
                "targets": _targets_of(list(tasks)),
                "smear_types": _smear_types_of(item.get("slide_type")),
                "output": _output_of(name, list(tasks), _optional_int(item.get("dpi")) is not None),
                "max_src_w": _src_limit(item.get("max_src_w")),
                "max_src_h": _src_limit(item.get("max_src_h")),
                "vram_gb": _vram_of(item),
            }
        )
    return rows


def load_model_table_from_manifest(manifest_path: str | None = None) -> int:
    """用 manifest.json 的 models 重建 MODEL_TABLE。文件不存在或读失败时保持原表。"""
    path = manifest_path or _default_manifest_path()
    if not os.path.isfile(path):
        _logger.info("manifest.json not found, MODEL_TABLE unchanged: %s", path)
        return 0
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        _logger.exception("Failed to read manifest.json: %s", path)
        return 0
    rows = _rows_from_manifest_models(payload.get("models") or [])
    MODEL_TABLE.clear()
    MODEL_TABLE.extend(rows)
    _logger.info("MODEL_TABLE loaded from manifest.json (%s entries): %s", len(rows), path)
    return len(rows)


load_model_table_from_manifest()


def model_dpi_ranges() -> dict[int, tuple[int, int]]:
    """actual_dpi → 并集后的 DPI 适用范围。"""
    ranges: dict[int, tuple[int, int]] = {}
    for row in MODEL_TABLE:
        dr = row.get("dpi_range")
        if not dr or dr == "不限制":
            continue
        actual_raw = row.get("actual_dpi")
        if actual_raw is None:
            continue
        actual = int(actual_raw)
        low, high = int(dr[0]), int(dr[1])
        if actual in ranges:
            lo, hi = ranges[actual]
            ranges[actual] = (min(lo, low), max(hi, high))
        else:
            ranges[actual] = (low, high)
    return ranges


def model_max_src_by_actual_dpi() -> dict[int, tuple[int, int] | None]:
    """
    actual_dpi → 该档位 detection/score 模型的原图尺寸上限 (max_w, max_h)。
    若该档位所有相关模型均不限尺寸，则值为 None。
    """
    limits: dict[int, tuple[int, int]] = {}
    has_limited: dict[int, bool] = {}
    for row in MODEL_TABLE:
        if row.get("output") not in ("bboxes", "scores"):
            continue
        actual_raw = row.get("actual_dpi")
        if actual_raw is None:
            continue
        actual = int(actual_raw)
        mw, mh = row.get("max_src_w"), row.get("max_src_h")
        if mw is None or mh is None:
            has_limited.setdefault(actual, False)
            continue
        has_limited[actual] = True
        cur = limits.get(actual)
        if cur is None:
            limits[actual] = (int(mw), int(mh))
        else:
            limits[actual] = (max(cur[0], int(mw)), max(cur[1], int(mh)))
    out: dict[int, tuple[int, int] | None] = {}
    for actual in set(limits) | set(has_limited):
        out[actual] = limits[actual] if has_limited.get(actual) else None
    return out



