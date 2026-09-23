# -*- coding: utf-8 -*-
"""
推理服务熔断：任一 GPU 超时重试耗尽后，掐断全部图片推理，
并对 config 中每张卡的独立容器调用 /admin/force_exit。
之后每秒探测全部 /health，全部 ready 才恢复接图。
Web 进程启动时清除落盘熔断，避免重启 Web 后沿用 open 状态反复 force_exit。

熔断状态落盘（backend/tmp/pipeline_circuit.json），uvicorn workers>1 时各进程共享。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any
from urllib.parse import urlparse

import requests

from config import TRITON_ENDPOINTS

logger = logging.getLogger(__name__)

HEALTH_CONNECT_TIMEOUT_S = float(os.environ.get("HEALTH_CONNECT_TIMEOUT_S", "3"))
HEALTH_READ_TIMEOUT_S = float(os.environ.get("HEALTH_READ_TIMEOUT_S", "5"))
HEALTH_POLL_INTERVAL_S = float(os.environ.get("HEALTH_POLL_INTERVAL_S", "1"))
FORCE_EXIT_CONNECT_TIMEOUT_S = float(os.environ.get("FORCE_EXIT_CONNECT_TIMEOUT_S", "3"))
FORCE_EXIT_READ_TIMEOUT_S = float(os.environ.get("FORCE_EXIT_READ_TIMEOUT_S", "5"))
FORCE_EXIT_RETRY_SEC = float(os.environ.get("FORCE_EXIT_RETRY_SEC", "15"))
# force_exit 先回响应再 os._exit，立刻打 /health 仍可能是旧进程的 ok。
FORCE_EXIT_WAIT_SEC = float(os.environ.get("FORCE_EXIT_WAIT_SEC", "5"))
# 额外 force_exit 次数上限（不含首次）。避免健康检查错过掉线窗口后每 15s 杀一次。
FORCE_EXIT_MAX_RETRIES = int(os.environ.get("FORCE_EXIT_MAX_RETRIES", "2"))
# 平扫 / 单张识别在熔断时先等待再返回，配合客户端重试，避免同一张图打满。
CIRCUIT_OPEN_WAIT_S = float(os.environ.get("CIRCUIT_OPEN_WAIT_S", "12"))

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STATE_DIR = os.path.join(_ROOT, "backend", "tmp")
_STATE_PATH = os.path.join(_STATE_DIR, "pipeline_circuit.json")
_LOCK_PATH = os.path.join(_STATE_DIR, "pipeline_circuit.lock")


class PipelineUnavailable(RuntimeError):
    """熔断开启或推理服务重启中，暂不接收图片推理。"""


class _FileLock:
    """跨进程互斥，覆盖 Windows / POSIX。"""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fh: Any = None

    def __enter__(self) -> "_FileLock":
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fh = open(self.path, "a+b")
        if self._fh.tell() == 0:
            self._fh.write(b"\0")
            self._fh.flush()
        self._fh.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
        else:
            import fcntl

            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._fh is None:
            return
        try:
            self._fh.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


_thread_lock = threading.Lock()
_poller_thread: threading.Thread | None = None


def _empty_state() -> dict[str, Any]:
    return {
        "open": False,
        "opened_at": 0.0,
        "force_exit_at": 0.0,
        "force_exit_count": 0,
        "trigger_gpu_id": None,
        "seen_down": {},
    }


def _read_state_unlocked() -> dict[str, Any]:
    try:
        with open(_STATE_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            st = _empty_state()
            st.update(data)
            if not isinstance(st.get("seen_down"), dict):
                st["seen_down"] = {}
            return st
    except (OSError, ValueError):
        pass
    return _empty_state()


def _write_state_unlocked(st: dict[str, Any]) -> None:
    os.makedirs(_STATE_DIR, exist_ok=True)
    tmp = _STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(st, fh, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, _STATE_PATH)


def _load_state() -> dict[str, Any]:
    with _FileLock(_LOCK_PATH):
        return _read_state_unlocked()


def _endpoint_name(gpu_id: int) -> str:
    if 0 <= gpu_id < len(TRITON_ENDPOINTS):
        return TRITON_ENDPOINTS[gpu_id].get("name") or f"triton_gpu{gpu_id}"
    return f"triton_gpu{gpu_id}"


def _pipeline_base_url(gpu_id: int) -> str:
    if gpu_id < 0 or gpu_id >= len(TRITON_ENDPOINTS):
        return ""
    return (TRITON_ENDPOINTS[gpu_id].get("pipeline_base_url") or "").rstrip("/")


def force_exit_url(gpu_id: int) -> str:
    base = _pipeline_base_url(gpu_id)
    return f"{base}/admin/force_exit" if base else ""


def _resolve_gpu_id_from_url(url: str | None) -> int | None:
    if not url:
        return None
    raw = url if "://" in url.lower() else f"http://{url}"
    netloc = (urlparse(raw).netloc or "").lower()
    if not netloc:
        return None
    for gpu_id, ep in enumerate(TRITON_ENDPOINTS):
        base = (ep.get("pipeline_base_url") or "").rstrip("/")
        if not base:
            continue
        if "://" not in base.lower():
            base = f"http://{base}"
        if (urlparse(base).netloc or "").lower() == netloc:
            return gpu_id
    return None


def is_circuit_open(gpu_id: int | None = None) -> bool:
    return bool(_load_state().get("open"))


def reset_circuit_on_startup() -> None:
    """Web 进程启动时清除落盘熔断，恢复接图，避免沿用上次 open 状态反复 force_exit。"""
    with _FileLock(_LOCK_PATH):
        st = _read_state_unlocked()
        was_open = bool(st.get("open"))
        _write_state_unlocked(_empty_state())
    if was_open:
        logger.warning("Web 启动：发现残留熔断状态，已清除并恢复接收图片推理")
    else:
        logger.info("Web 启动：熔断状态已复位")


def circuit_snapshot(gpu_id: int | None = None) -> dict[str, Any]:
    st = _load_state()
    open_now = bool(st.get("open"))
    opened_at = float(st.get("opened_at") or 0.0)
    trigger = st.get("trigger_gpu_id")
    if trigger is not None:
        try:
            trigger = int(trigger)
        except (TypeError, ValueError):
            trigger = None
    seen = st.get("seen_down") or {}
    endpoints = []
    for gid in range(len(TRITON_ENDPOINTS)):
        endpoints.append(
            {
                "gpu_id": gid,
                "name": _endpoint_name(gid),
                "open": open_now,
                "seen_down": bool(seen.get(str(gid), seen.get(gid))),
                "force_exit_url": force_exit_url(gid),
            }
        )
    return {
        "open": open_now,
        "trigger_gpu_id": trigger,
        "trigger_name": _endpoint_name(trigger) if trigger is not None else None,
        "open_age_sec": (
            round(max(0.0, time.time() - opened_at), 3) if open_now and opened_at else 0.0
        ),
        "endpoints": endpoints,
    }


def assert_inference_allowed(gpu_id: int | None = None) -> None:
    if is_circuit_open():
        _ensure_poller()
        raise PipelineUnavailable("推理服务重启中，暂不接收图片推理")


def wait_while_circuit_open() -> None:
    """熔断中阻塞一段时间再让调用方返回错误，避免客户端立即重试。"""
    delay = max(0.0, CIRCUIT_OPEN_WAIT_S)
    if delay <= 0:
        return
    logger.warning("熔断中，等待 %.0fs 后再拒绝推理请求", delay)
    time.sleep(delay)


def check_pipeline_health(endpoint: dict, gpu_id: int) -> dict[str, Any]:
    """请求该卡 multi_pipeline_server /health，对齐其 pipelines 就绪判定。"""
    base = (endpoint.get("pipeline_base_url") or "").rstrip("/")
    name = endpoint.get("name") or f"triton_gpu{gpu_id}"
    item: dict[str, Any] = {
        "gpu_id": gpu_id,
        "name": name,
        "pipeline_base_url": base,
    }
    if not base:
        item["status"] = "unreachable"
        item["error"] = "pipeline_base_url is empty"
        return item

    url = f"{base}/health"
    try:
        resp = requests.get(
            url,
            timeout=(HEALTH_CONNECT_TIMEOUT_S, HEALTH_READ_TIMEOUT_S),
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        item["status"] = "unreachable"
        item["error"] = str(e)
        return item
    except ValueError as e:
        item["status"] = "unreachable"
        item["error"] = f"invalid JSON response: {e}"
        return item

    item.update(
        {
            "status": data.get("status", "unknown"),
            "runtime_profile": data.get("runtime_profile"),
            "pipelines": data.get("pipelines"),
            "watchdog": data.get("watchdog"),
            "triton_grpc_url": data.get("triton_grpc_url"),
            "triton_http_url": data.get("triton_http_url"),
        }
    )
    return item


def collect_inference_health() -> tuple[list[dict[str, Any]], bool]:
    """检查 config 中全部 Triton 端点。返回 (inference 列表, 是否全部 ready 且未熔断)。"""
    inference = [
        check_pipeline_health(ep, gpu_id) for gpu_id, ep in enumerate(TRITON_ENDPOINTS)
    ]
    circuit_open = is_circuit_open()
    if circuit_open:
        _ensure_poller()
    for item in inference:
        item["circuit_open"] = circuit_open
        if circuit_open:
            item["circuit"] = {
                "open": True,
                "force_exit_url": force_exit_url(int(item["gpu_id"])),
            }
    ready = (
        bool(inference)
        and not circuit_open
        and all(item.get("status") == "ok" for item in inference)
    )
    return inference, ready


def trip_on_timeout(*, gpu_id: int | None = None, url: str | None = None) -> None:
    """任一卡三次超时：熔断全部推理，并对每张卡独立容器 force_exit。"""
    if gpu_id is None:
        gpu_id = _resolve_gpu_id_from_url(url)
    do_force_exit = False
    with _FileLock(_LOCK_PATH):
        st = _read_state_unlocked()
        if st.get("open"):
            already_open = True
        else:
            already_open = False
            now = time.time()
            st["open"] = True
            st["opened_at"] = now
            st["force_exit_at"] = now
            st["force_exit_count"] = 1
            st["trigger_gpu_id"] = gpu_id
            st["seen_down"] = {str(gid): False for gid in range(len(TRITON_ENDPOINTS))}
            _write_state_unlocked(st)
            do_force_exit = True
    if already_open:
        logger.warning("推理服务已处于整组熔断重启中，跳过重复 force_exit")
        _ensure_poller()
        return
    names = ", ".join(_endpoint_name(gid) for gid in range(len(TRITON_ENDPOINTS))) or "(none)"
    logger.error(
        "推理服务连续超时（trigger=%s url=%s pid=%s），熔断全部请求并重启全部卡: %s",
        _endpoint_name(gpu_id) if gpu_id is not None else "-",
        url or "-",
        os.getpid(),
        names,
    )
    if do_force_exit:
        _force_exit_all()
    _ensure_poller()


def _close_circuit() -> None:
    with _FileLock(_LOCK_PATH):
        st = _empty_state()
        _write_state_unlocked(st)
    logger.info("全部推理服务 /health 已 ready，恢复接收图片推理")


def _force_exit_all() -> None:
    if not TRITON_ENDPOINTS:
        logger.error("TRITON_ENDPOINTS 为空，无法调用 /admin/force_exit")
        return
    for gpu_id in range(len(TRITON_ENDPOINTS)):
        _force_exit_one(gpu_id)


def _force_exit_one(gpu_id: int) -> None:
    name = _endpoint_name(gpu_id)
    url = force_exit_url(gpu_id)
    if not url:
        logger.error("force_exit 跳过 %s：pipeline_base_url 为空", name)
        return
    try:
        resp = requests.post(
            url,
            timeout=(FORCE_EXIT_CONNECT_TIMEOUT_S, FORCE_EXIT_READ_TIMEOUT_S),
        )
    except requests.RequestException as e:
        logger.error("force_exit %s (%s) 失败: %s", name, url, e)
        return

    payload: dict[str, Any] = {}
    try:
        data = resp.json()
        if isinstance(data, dict):
            payload = data
    except ValueError:
        pass
    status = payload.get("status")
    exit_code = payload.get("exit_code")
    logger.error(
        "force_exit %s (%s) -> HTTP %s status=%s exit_code=%s body=%s",
        name,
        url,
        resp.status_code,
        status,
        exit_code,
        (resp.text or "")[:300],
    )
    if resp.status_code >= 400 or status not in (None, "exiting"):
        logger.error(
            "force_exit %s 响应异常，仍等待容器退出后探测 /health",
            name,
        )


def _wait_after_force_exit() -> None:
    """等容器真正退出后再探测；按共享 force_exit_at 计算剩余等待，避免双 worker 重复空等。"""
    st = _load_state()
    last = float(st.get("force_exit_at") or 0.0)
    remain = last + max(0.0, FORCE_EXIT_WAIT_SEC) - time.time()
    if remain <= 0:
        return
    logger.info(
        "force_exit 已发出，再等待 %.1f 秒后检查 /health（避免打到尚未退出的旧进程）",
        remain,
    )
    time.sleep(remain)


def _ensure_poller() -> None:
    global _poller_thread
    with _thread_lock:
        if _poller_thread is not None and _poller_thread.is_alive():
            return
        thread = threading.Thread(
            target=_recovery_loop,
            name="pipeline-health-poller",
            daemon=True,
        )
        _poller_thread = thread
        thread.start()


def _seen_flag(seen: dict[str, Any], gid: int) -> bool:
    return bool(seen.get(str(gid), seen.get(gid)))


def _recovery_loop() -> None:
    """熔断后立刻轮询 /health：必须先见到掉线，再全部 ready 才恢复接图。

    不再先空等 FORCE_EXIT_WAIT_SEC，否则容易错过掉线窗口，把旧进程的 ok
    当成已恢复。health 一直 ok 时最多再 force_exit 有限次，然后保持熔断
    （拒绝推理），直到见到重启完成，或 Web 进程启动清状态。
    """
    logger.info(
        "开始探测全部推理服务 /health，pid=%s interval=%.0fs",
        os.getpid(),
        HEALTH_POLL_INTERVAL_S,
    )
    while True:
        started = time.monotonic()
        st = _load_state()
        if not st.get("open"):
            return

        try:
            items = [
                check_pipeline_health(ep, gid) for gid, ep in enumerate(TRITON_ENDPOINTS)
            ]
        except Exception:
            logger.exception("health 探测失败")
            items = []

        retry_exit = False
        still_up: list[int] = []
        max_exits = 1 + max(0, FORCE_EXIT_MAX_RETRIES)
        with _FileLock(_LOCK_PATH):
            st = _read_state_unlocked()
            if not st.get("open"):
                return
            seen = dict(st.get("seen_down") or {})
            if not items:
                for gid in range(len(TRITON_ENDPOINTS)):
                    seen[str(gid)] = True
            for item in items:
                gid = int(item["gpu_id"])
                if item.get("status") != "ok":
                    seen[str(gid)] = True
            st["seen_down"] = seen
            gids = list(range(len(TRITON_ENDPOINTS)))
            all_seen_down = bool(gids) and all(_seen_flag(seen, gid) for gid in gids)
            all_ready = bool(items) and all(item.get("status") == "ok" for item in items)
            last_exit = float(st.get("force_exit_at") or 0.0)
            exit_count = int(st.get("force_exit_count") or 1)
            still_up = [gid for gid in gids if not _seen_flag(seen, gid)]
            if all_seen_down and all_ready:
                _write_state_unlocked(_empty_state())
                logger.info("全部推理服务 /health 已 ready，恢复接收图片推理")
                return
            if (
                (not all_seen_down)
                and last_exit
                and (time.time() - last_exit) >= FORCE_EXIT_RETRY_SEC
                and exit_count < max_exits
            ):
                st["force_exit_at"] = time.time()
                st["force_exit_count"] = exit_count + 1
                retry_exit = True
            elif (
                (not all_seen_down)
                and last_exit
                and (time.time() - last_exit) >= FORCE_EXIT_RETRY_SEC
                and exit_count >= max_exits
                and not st.get("retry_exhausted")
            ):
                st["retry_exhausted"] = True
                logger.error(
                    "force_exit 已达上限 %s 次，仍有卡未见掉线，停止反复重启并保持熔断: %s",
                    max_exits,
                    ", ".join(_endpoint_name(gid) for gid in still_up) or "all",
                )
            _write_state_unlocked(st)

        if retry_exit:
            logger.error(
                "force_exit 后仍有卡 health=ready，再次重启 (%s/%s): %s",
                exit_count + 1,
                max_exits,
                ", ".join(_endpoint_name(gid) for gid in still_up) or "all",
            )
            _force_exit_all()
            continue

        elapsed = time.monotonic() - started
        time.sleep(max(0.0, HEALTH_POLL_INTERVAL_S - elapsed))
