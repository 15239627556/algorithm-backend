# model_control.py
"""
Triton 模型动态加载/卸载。需 Triton 以 --model-control-mode=explicit 启动。
显存最多保留 max_models 个模型，按加载顺序 FIFO 淘汰。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import urllib.request
from typing import List, Tuple

logger = logging.getLogger(__name__)

# TRITON_HTTP_URL: 模型加载/卸载用 HTTP API，默认从 TRITON_URL 推导 (端口改为 8000)
_TRITON_URL = os.environ.get("TRITON_URL", "192.168.31.188:18001")


def _get_http_base_url() -> str:
    url = os.environ.get("TRITON_HTTP_URL")
    if url:
        return url.rstrip("/")
    if "://" in _TRITON_URL:
        base = _TRITON_URL.split("//")[1].split(":")[0]
        return f"http://{base}:8000"
    host = _TRITON_URL.split(":")[0]
    return f"http://{host}:8000"


_loaded_order: List[str] = []
_max_models = 3
_model_lock = threading.Lock()

LOAD_TIMEOUT = int(os.environ.get("TRITON_LOAD_TIMEOUT", "600"))


def _http_post(url: str, body: bytes = None, timeout: int = LOAD_TIMEOUT) -> Tuple[bool, str]:
    try:
        req = urllib.request.Request(url, data=body or b"{}", method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, resp.read().decode() if resp.length else ""
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode()
        except Exception:
            err_body = str(e)
        return False, f"HTTP {e.code}: {err_body}"
    except Exception as e:
        return False, str(e)


def get_loaded_models() -> List[str]:
    """获取当前已加载的模型列表（通过 Triton repository index，ready=true）"""
    base = _get_http_base_url()
    url = f"{base}/v2/repository/index"
    body = json.dumps({"ready": True}).encode()
    ok, msg = _http_post(url, body, timeout=30)
    if not ok:
        logger.warning("get_loaded_models failed: %s", msg)
        return []
    try:
        data = json.loads(msg)
        return [m["name"] for m in data if m.get("state") == "READY"]
    except Exception as e:
        logger.warning("parse get_loaded_models response: %s", e)
        return []


def load_model(model_name: str, timeout: int = LOAD_TIMEOUT) -> Tuple[bool, str]:
    """加载单个模型"""
    base = _get_http_base_url()
    url = f"{base}/v2/repository/models/{model_name}/load"
    ok, msg = _http_post(url, b"{}", timeout=timeout)
    if ok:
        logger.info("Model %s loaded.", model_name)
    else:
        logger.warning("Load model %s failed: %s", model_name, msg)
    return ok, msg


def unload_model(model_name: str, timeout: int = 60) -> Tuple[bool, str]:
    """卸载单个模型"""
    base = _get_http_base_url()
    url = f"{base}/v2/repository/models/{model_name}/unload"
    ok, msg = _http_post(url, b"{}", timeout=timeout)
    if ok:
        logger.info("Model %s unloaded.", model_name)
    else:
        logger.warning("Unload model %s failed: %s", model_name, msg)
    return ok, msg


def ensure_model_loaded(model_name: str, max_models: int = 3) -> Tuple[bool, str]:
    """
    确保模型已加载。若超过 max_models 个模型在显存中，按加载顺序卸载最旧的。
    返回 (成功, 错误信息)
    """
    global _loaded_order
    with _model_lock:
        actually_loaded = set(get_loaded_models())
        _loaded_order[:] = [m for m in _loaded_order if m in actually_loaded]

        if model_name in actually_loaded:
            if model_name in _loaded_order:
                _loaded_order.remove(model_name)
            _loaded_order.append(model_name)
            return True, ""

        while len(actually_loaded) >= max_models:
            to_unload = _loaded_order.pop(0) if _loaded_order else (get_loaded_models()[0] if get_loaded_models() else None)
            if not to_unload:
                break
            unload_ok, _ = unload_model(to_unload)
            if unload_ok:
                actually_loaded.discard(to_unload)
                if to_unload in _loaded_order:
                    _loaded_order.remove(to_unload)
            else:
                if to_unload not in _loaded_order:
                    _loaded_order.insert(0, to_unload)
                break
            actually_loaded = set(get_loaded_models())

        ok, msg = load_model(model_name)
        if ok:
            _loaded_order.append(model_name)
        return ok, msg


def warmup_model(model_name: str) -> None:
    """模型预热：创建任务或单张识别前调用，确保模型已加载。失败时记录日志，不抛异常。"""
    ok, msg = ensure_model_loaded(model_name, max_models=3)
    if not ok:
        logger.warning("Model warmup failed for %s: %s", model_name, msg)
