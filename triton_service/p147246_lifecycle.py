from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict


Trace = Dict[str, Any]


def _wall_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def new_trace(
    *,
    image_seq: int,
    route: str,
    client_seq: str | None = None,
    filename: str | None = None,
    enable_meg: bool | None = None,
) -> Trace:
    trace: Trace = {
        "image_seq": image_seq,
        "route": route,
        "client_seq": client_seq,
        "filename": filename,
        "enable_meg": enable_meg,
        "status": "running",
        "stages": {},
    }
    return trace


def set_field(trace: Trace | None, key: str, value: Any) -> None:
    if trace is not None and value is not None:
        trace[key] = value


def set_batch(trace: Trace | None, **kwargs: Any) -> None:
    if trace is None:
        return
    batch = trace.setdefault("batch", {})
    for key, value in kwargs.items():
        if value is not None:
            batch[key] = value


def set_result(trace: Trace | None, **kwargs: Any) -> None:
    if trace is None:
        return
    result = trace.setdefault("result", {})
    for key, value in kwargs.items():
        if value is not None:
            result[key] = value


def stage_start(trace: Trace | None, name: str, *, scope: str = "image") -> None:
    if trace is None:
        return
    trace.setdefault("stages", {})[name] = {
        "scope": scope,
        "start_wall": _wall_now(),
        "_start_ns": time.perf_counter_ns(),
    }


def stage_end(trace: Trace | None, name: str) -> None:
    if trace is None:
        return
    stage = trace.setdefault("stages", {}).setdefault(name, {})
    end_ns = time.perf_counter_ns()
    start_ns = stage.pop("_start_ns", None)
    stage["end_wall"] = _wall_now()
    if start_ns is not None:
        stage["ms"] = round((end_ns - int(start_ns)) / 1_000_000.0, 3)


def stage_instant(trace: Trace | None, name: str, *, scope: str = "image") -> None:
    if trace is None:
        return
    trace.setdefault("stages", {})[name] = {
        "scope": scope,
        "wall": _wall_now(),
        "_instant_ns": time.perf_counter_ns(),
    }


def clean_trace(trace: Trace) -> Trace:
    for stage in trace.get("stages", {}).values():
        stage.pop("_start_ns", None)
        stage.pop("_instant_ns", None)
    return trace


def finish_ok(trace: Trace | None, *, request_total_ms: float | None = None) -> None:
    if trace is None:
        return
    trace["status"] = "ok"
    if request_total_ms is not None:
        trace["request_total_ms"] = request_total_ms


def finish_error(trace: Trace | None, error: BaseException, *, request_total_ms: float | None = None) -> None:
    if trace is None:
        return
    trace["status"] = "error"
    trace["error_type"] = type(error).__name__
    trace["error"] = str(error)
    if request_total_ms is not None:
        trace["request_total_ms"] = request_total_ms


def to_json_line(trace: Trace) -> str:
    return json.dumps(clean_trace(trace), ensure_ascii=False, separators=(",", ":"))
