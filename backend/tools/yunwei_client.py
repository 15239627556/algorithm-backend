# -*- coding: utf-8 -*-
"""
运维平台接入：Eureka 注册 + Spring Cloud Config。

本进程就绪（/health 返回 200）后：
1. POST {注册中心}/eureka/apps/{APP}
2. GET  {注册中心}/eureka/apps ，取 config 实例的 homePageUrl
3. GET  {homePageUrl}/{vip}/{profile}/{label}
之后每 30 秒心跳；配置每 30 秒按 version 刷新。退出时先 OUT_OF_SERVICE，
等待 yunwei_drain_seconds，再 DELETE。

多 worker 时只用一把文件锁，避免同一个 instanceId 被反复注销。
X-Micro-Token 只在 config.yunwei_micro_token 非空时附带。
"""
from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import random
import socket
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import urlparse

import requests

import config as cfg

logger = logging.getLogger("backend.tools.yunwei_client")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOCK_PATH = os.path.join(_ROOT, "logs", "yunwei_register.lock")

# 注册中心里统一配置服务本身的应用名，用来拿到配置服务地址。
_CONFIG_SERVER_APP = "config"
_RENEW_SECONDS = 30
_LEASE_SECONDS = 90
_REGISTER_RETRY_SECONDS = 5
_HEARTBEAT_BACKOFF = (1, 2, 4, 8, 16, 30)
_CONFIG_ALERT_SECONDS = 300

# 环境变量优先于配置中心。只热更新运行中会重新读取的项。
_SPECS: tuple[dict[str, Any], ...] = (
    {
        "attr": "cellularity_file_path_prefix",
        "env": "CELLULARITY_FILE_PATH_PREFIX",
        "kind": "str",
        "patch": (("backend.services.task_service", "cellularity_file_path_prefix"),),
    },
    {
        "attr": "camera",
        "env": "CAMERA",
        "kind": "str",
        "patch": (("backend.tools.model_control", "camera"),),
    },
    {"attr": "rbc_switch", "env": "RBC_SWITCH", "kind": "bool", "patch": ()},
    {"attr": "plt_switch", "env": "PLT_SWITCH", "kind": "bool", "patch": ()},
    {
        "attr": "max_memory",
        "env": "MAX_MEMORY",
        "kind": "int",
        "patch": (("backend.tools.model_control", "max_memory"),),
    },
    {
        "attr": "reserved_memory",
        "env": "RESERVED_MEMORY",
        "kind": "int",
        "patch": (("backend.tools.model_control", "reserved_memory"),),
    },
    {
        "attr": "save_heatmap",
        "env": "SAVE_HEATMAP",
        "kind": "bool",
        "patch": (("backend.services.task_service", "default_save_heatmap"),),
    },
    {"attr": "TRITON_ROUND_ROBIN", "env": "TRITON_ROUND_ROBIN", "kind": "bool", "patch": ()},
    {"attr": "TRITON_GPU_ID", "env": "TRITON_GPU_ID", "kind": "int", "patch": ()},
)

_client: "YunweiClient | None" = None
_client_lock = threading.Lock()
_atexit_registered = False


class CipherConfigError(RuntimeError):
    """配置值仍以 {cipher} 开头，服务端没有解密。"""

    def __init__(self, keys: list[str]) -> None:
        self.keys = keys
        super().__init__("cipher")


class HttpStatus(RuntimeError):
    def __init__(
        self,
        status: int,
        *,
        method: str = "",
        url: str = "",
        request_body: str = "",
        response_body: str = "",
    ) -> None:
        self.status = status
        self.method = method
        self.url = url
        self.request_body = request_body
        self.response_body = response_body
        super().__init__(f"http {status}")


class _TryLock:
    """跨进程非阻塞锁。持有者退出后操作系统会释放。"""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fh: Any = None

    def acquire(self) -> bool:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        fh = open(self.path, "a+b")
        try:
            if fh.seek(0, os.SEEK_END) == 0:
                fh.write(b"\0")
                fh.flush()
            fh.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        fh = self._fh
        if fh is None:
            return
        self._fh = None
        try:
            fh.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            fh.close()


def eureka_zone(base: str) -> str:
    """注册中心根地址补成 .../eureka/。"""
    text = (base or "").strip().rstrip("/")
    if not text:
        raise ValueError("empty eureka address")
    if "/eureka" not in text.lower():
        text += "/eureka"
    return text + "/"


def config_url(home: str, app: str, profile: str, label: str) -> str:
    return f"{home.rstrip('/')}/{app}/{profile}/{label}"


def merge_property_sources(env: dict[str, Any]) -> dict[str, Any]:
    """propertySources 从后往前合并，先出现的源覆盖后面的源。"""
    merged: dict[str, Any] = {}
    sources = env.get("propertySources") or []
    if not isinstance(sources, list):
        raise RuntimeError("propertySources")
    for source in reversed(sources):
        if not isinstance(source, dict):
            continue
        raw = source.get("source") or {}
        if not isinstance(raw, dict):
            continue
        for key, value in raw.items():
            merged[str(key)] = value
    return merged


def fingerprint(env: dict[str, Any], merged: dict[str, Any]) -> str:
    version = env.get("version")
    if version:
        return "v:" + str(version)
    raw = json.dumps(merged, sort_keys=True, ensure_ascii=False, default=str)
    return "h:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def reject_cipher(merged: dict[str, Any]) -> None:
    bad = [
        key
        for key, value in merged.items()
        if isinstance(value, str) and value.startswith("{cipher}")
    ]
    if bad:
        raise CipherConfigError(bad)


def _norm(text: str) -> str:
    return text.strip().lower().replace("-", "_").replace(".", "_")


def _as_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def port_num(inst: dict[str, Any]) -> int:
    port = inst.get("port")
    raw: Any = port
    if isinstance(port, dict):
        raw = port.get("$", port.get("port", 80))
    try:
        number = int(raw)
    except (TypeError, ValueError):
        return 80
    return number if number > 0 else 80


def instance_home(inst: dict[str, Any]) -> str:
    home = str(inst.get("homePageUrl") or "").strip()
    if home.startswith("http://") or home.startswith("https://"):
        return home if home.endswith("/") else home + "/"
    ip = str(inst.get("ipAddr") or inst.get("hostName") or "").strip()
    if not ip:
        return ""
    return f"http://{ip}:{port_num(inst)}/"


def iter_applications(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    body = payload.get("applications", payload)
    if not isinstance(body, dict):
        return []
    return _as_list(body.get("application", []))


def select_config_home(payload: Any, app_name: str) -> str:
    """从 Eureka 应用列表里随机挑一个 UP 的 config 实例。"""
    want = (app_name or "").strip().upper()
    homes: list[str] = []
    for app in iter_applications(payload):
        if str(app.get("name", "")).strip().upper() != want:
            continue
        for inst in _as_list(app.get("instance", [])):
            if str(inst.get("status", "UP")).strip().upper() != "UP":
                continue
            home = instance_home(inst)
            if home:
                homes.append(home)
    if not homes:
        raise LookupError(want)
    return random.choice(homes)


_SKIP_ADAPTER = (
    "wsl",
    "hyper-v",
    "vethernet",
    "docker",
    "virtualbox",
    "vmware",
    "loopback",
    "bluetooth",
    "npcap",
    "tunnel",
)


def _in_wsl() -> bool:
    if os.name != "posix":
        return False
    chunks: list[str] = []
    for path in ("/proc/sys/kernel/osrelease", "/proc/version"):
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                chunks.append(handle.read().lower())
        except OSError:
            continue
    text = "\n".join(chunks)
    return "microsoft" in text or "wsl" in text


def _host_tools_available() -> bool:
    if os.name == "nt" or _in_wsl():
        return True
    return os.path.isfile("/mnt/c/Windows/System32/ipconfig.exe")


def _ipv4_prefix(ip: str, host: str) -> int:
    left = ip.split(".")
    right = host.split(".")
    if len(left) != 4 or len(right) != 4:
        return 0
    matched = 0
    for one, other in zip(left, right):
        if one != other:
            break
        matched += 1
    return matched


def choose_advertise_ip(rows: list[tuple[str, str]], registry_host: str) -> str:
    """从 Windows 网卡里挑注册中心能访问到的 IPv4，跳过 WSL 虚拟网卡。"""
    usable: list[str] = []
    for ip, alias in rows:
        low = alias.lower()
        if any(word in low for word in _SKIP_ADAPTER):
            continue
        if ip.startswith("127.") or ip.startswith("169.254."):
            continue
        usable.append(ip)
    if not usable:
        return ""
    return max(usable, key=lambda item: _ipv4_prefix(item, registry_host))


def _decode_host_output(raw: bytes) -> str:
    if not raw:
        return ""
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    if b"\x00" in raw[:80]:
        return raw.decode("utf-16-le", errors="replace")
    for encoding in ("utf-8", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _run_host_command(args: list[str]) -> tuple[str, str]:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "", exc.__class__.__name__
    text = _decode_host_output(completed.stdout or b"")
    err = _decode_host_output(completed.stderr or b"").strip()
    if completed.returncode not in (0, None) and not text.strip():
        return "", err or f"exit {completed.returncode}"
    return text, err


def _host_command_paths(name: str) -> list[str]:
    if os.name == "nt":
        return [name]
    roots = (
        "/mnt/c/Windows/System32/",
        "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/",
        "",
    )
    paths: list[str] = []
    for root in roots:
        if root and not root.endswith(("/", "\\")):
            continue
        candidate = f"{root}{name}" if root else name
        if root and not os.path.isfile(candidate):
            continue
        if candidate not in paths:
            paths.append(candidate)
    return paths


def _parse_ipconfig(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    alias = ""
    for raw in text.splitlines():
        if not raw.strip():
            continue
        if not raw[:1].isspace():
            alias = raw.strip().rstrip(":")
            continue
        if "IPv4" not in raw or ":" not in raw:
            continue
        ip = raw.split(":")[-1].split("(")[0].strip()
        parts = ip.split(".")
        if len(parts) == 4 and all(part.isdigit() for part in parts):
            rows.append((ip, alias))
    return rows


def _windows_ipv4_rows() -> tuple[list[tuple[str, str]], str]:
    """向宿主机要 IPv4。先 ipconfig，再 PowerShell。失败时带回原因。"""
    errors: list[str] = []
    for path in _host_command_paths("ipconfig.exe"):
        text, err = _run_host_command([path])
        rows = _parse_ipconfig(text)
        if rows:
            return rows, ""
        if err or not text.strip():
            errors.append(f"{path}: {err or 'empty'}")
    script = (
        "Get-NetIPAddress -AddressFamily IPv4 | "
        "Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } | "
        "ForEach-Object { $_.IPAddress + '|' + $_.InterfaceAlias }"
    )
    for path in _host_command_paths("powershell.exe"):
        text, err = _run_host_command(
            [path, "-NoProfile", "-NonInteractive", "-Command", script]
        )
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if "|" not in line:
                continue
            ip, alias = line.split("|", 1)
            ip = ip.strip()
            parts = ip.split(".")
            if len(parts) == 4 and all(part.isdigit() for part in parts):
                rows.append((ip, alias.strip()))
        if rows:
            return rows, ""
        errors.append(f"{path}: {err or 'empty'}")
    return [], "; ".join(errors) or "no windows command"


def _route_ip(registry: str) -> str:
    host = urlparse(registry).hostname or "8.8.8.8"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((host, 80))
        return sock.getsockname()[0]
    finally:
        sock.close()


def detect_advertise_ip(explicit: str, registry: str) -> str:
    """注册必须用宿主机可达 IP，不能是 127.0.0.1，也不能是 WSL 内部地址。"""
    ip = (explicit or "").strip()
    if ip:
        if ip.startswith("127.") or ip == "::1":
            raise RuntimeError("loopback")
        return ip
    host = urlparse(registry).hostname or ""
    if _host_tools_available():
        rows, reason = _windows_ipv4_rows()
        chosen = choose_advertise_ip(rows, host) if rows else ""
        if chosen:
            logger.info("注册 IP 取自宿主机网卡 %s", chosen)
            return chosen
        logger.warning("未读到宿主机网卡，原因=%s", reason)
    ip = _route_ip(registry)
    if not ip or ip.startswith("127.") or ip == "::1":
        raise RuntimeError("loopback")
    logger.warning("注册 IP 回退为本机路由地址 %s", ip)
    return ip


def advertise_port() -> int:
    """注册中心上的端口固定为宿主机转发端口。"""
    return int(cfg.yunwei_advertise_port)


def build_instance(
    *,
    app: str,
    ip: str,
    port: int,
    version: str,
    zone: str,
) -> dict[str, Any]:
    vip = app.lower()
    now_ms = int(time.time() * 1000)
    base = f"http://{ip}:{port}"
    return {
        "instance": {
            "instanceId": f"{ip}:{vip}:{port}",
            "hostName": ip,
            "app": app,
            "appGroupName": "APP",
            "ipAddr": ip,
            "vipAddress": vip,
            "secureVipAddress": vip,
            "status": "UP",
            "overriddenstatus": "UNKNOWN",
            "port": {"$": int(port), "@enabled": "true"},
            "dataCenterInfo": {
                "@class": "com.netflix.appinfo.InstanceInfo$DefaultDataCenterInfo",
                "name": "MyOwn",
            },
            "leaseInfo": {
                "renewalIntervalInSecs": _RENEW_SECONDS,
                "durationInSecs": _LEASE_SECONDS,
            },
            "metadata": {
                "version": version,
                "zone": zone or "default",
                "protocol": "http",
                "language": "python",
                "management.port": str(port),
            },
            "homePageUrl": base + "/",
            "healthCheckUrl": base + "/health",
            "statusPageUrl": base + "/health",
            "lastUpdatedTimestamp": now_ms,
            "lastDirtyTimestamp": now_ms,
            "actionType": "ADDED",
        }
    }


def _env_set(name: str | None) -> bool:
    if not name:
        return False
    raw = os.environ.get(name)
    return raw is not None and raw.strip() != ""


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off", ""):
        return False
    raise ValueError("bool")


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("int")
    if isinstance(value, str) and not value.strip():
        raise ValueError("int")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("int")
        return int(value)
    return int(str(value).strip(), 10)


def _as_str(value: Any) -> str:
    if isinstance(value, (dict, list)):
        raise ValueError("str")
    return str(value)


def _convert(kind: str, value: Any) -> Any:
    if kind == "bool":
        return _as_bool(value)
    if kind == "int":
        return _as_int(value)
    return _as_str(value)


def _lookup_spec(merged: dict[str, Any], spec: dict[str, Any]) -> tuple[str, Any] | None:
    names = {_norm(spec["attr"])}
    exact: tuple[str, Any] | None = None
    tail: tuple[str, Any] | None = None
    for key, value in merged.items():
        if _norm(key) in names:
            exact = (key, value)
            break
        if tail is None and _norm(key.split(".")[-1]) in names:
            tail = (key, value)
    return exact or tail


def plan_updates(merged: dict[str, Any]) -> tuple[list[tuple[str, Any, tuple]], list[str]]:
    """返回 (待写入计划, 被环境变量挡住的键)。转换失败时整份配置不应用。"""
    planned: list[tuple[str, Any, tuple]] = []
    held: list[str] = []
    for spec in _SPECS:
        found = _lookup_spec(merged, spec)
        if found is None:
            continue
        key, raw = found
        if _env_set(spec["env"]):
            held.append(key)
            continue
        try:
            value = _convert(spec["kind"], raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"bad config key {key}") from exc
        planned.append((spec["attr"], value, spec["patch"]))
    return planned, held


def apply_plan(planned: list[tuple[str, Any, tuple]]) -> None:
    for attr, value, patches in planned:
        setattr(cfg, attr, value)
        for mod_name, alias in patches:
            mod = sys.modules.get(mod_name)
            if mod is not None and hasattr(mod, alias):
                setattr(mod, alias, value)


class YunweiClient:
    def __init__(self, version: str) -> None:
        self.version = (version or "").strip() or cfg.sufa_version
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._config_thread: threading.Thread | None = None
        self._lock = _TryLock(_LOCK_PATH)
        self._owner = False
        self._registered = False
        self._auth_failed = False
        self._config_denied = False
        self._draining = False
        self._config_blocked = False
        self._eureka_status = ""
        self._instance_id = ""
        self._config_home = ""
        self._merged: dict[str, Any] = {}
        self._fp = ""
        self._config_version = ""
        self._cfg_lock = threading.Lock()
        self._warn_at: dict[str, float] = {}
        self._sessions = threading.local()
        self._first_config = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._owner_loop,
            name="yunwei-register",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(cfg.yunwei_drain_seconds + 30)

    def get(self, key: str, default: Any = None) -> Any:
        with self._cfg_lock:
            if key in self._merged:
                return self._merged[key]
            wanted = _norm(key)
            for name, value in self._merged.items():
                if _norm(name) == wanted:
                    return value
        return default

    def _session(self) -> requests.Session:
        session = getattr(self._sessions, "session", None)
        if session is None:
            session = requests.Session()
            session.trust_env = False
            self._sessions.session = session
        return session

    def _headers(self, content_json: bool) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if content_json:
            headers["Content-Type"] = "application/json"
        token = cfg.yunwei_micro_token
        if token:
            headers["X-Micro-Token"] = token
        return headers

    def _auth(self) -> tuple[str, str] | None:
        if cfg.yunwei_user:
            return (cfg.yunwei_user, cfg.yunwei_password)
        return None

    def _platform(
        self,
        method: str,
        url: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        timeout: tuple[float, float] = (1, 3),
    ) -> requests.Response:
        return self._session().request(
            method,
            url,
            headers=self._headers(json_body is not None),
            auth=self._auth(),
            json=json_body,
            params=params,
            timeout=timeout,
        )

    def _warn(self, key: str, interval: float, message: str, *args: Any) -> None:
        now = time.monotonic()
        if now - self._warn_at.get(key, 0.0) < interval:
            return
        self._warn_at[key] = now
        logger.warning(message, *args)

    def _owner_loop(self) -> None:
        announced = False
        try:
            while not self._stop.is_set():
                if not self._lock.acquire():
                    if not announced:
                        logger.info("其他进程已负责运维平台注册")
                        announced = True
                    if self._stop.wait(5):
                        return
                    continue
                self._owner = True
                try:
                    logger.info(
                        "本进程注册运维平台 app=%s version=%s",
                        cfg.sufa_name,
                        self.version,
                    )
                    if not self._wait_health():
                        return
                    if not self._register_until_ok():
                        return
                    self._config_thread = threading.Thread(
                        target=self._config_loop,
                        name="yunwei-config",
                        daemon=True,
                    )
                    self._config_thread.start()
                    self._first_config.wait(15)
                    self._heartbeat_loop()
                    return
                finally:
                    try:
                        self._on_stop()
                    finally:
                        self._lock.release()
                        self._owner = False
        except Exception:
            logger.exception("运维平台注册线程退出")

    def _wait_health(self) -> bool:
        host = cfg.APP_HOST
        if host in ("0.0.0.0", "::", ""):
            host = "127.0.0.1"
        url = f"http://{host}:{cfg.APP_PORT}/health"
        while not self._stop.is_set():
            try:
                resp = self._session().get(url, timeout=(1, 3))
                if resp.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            if self._stop.wait(1):
                return False
        return False

    def _register_until_ok(self) -> bool:
        while not self._stop.is_set():
            try:
                if self._register_once():
                    return True
            except requests.RequestException:
                self._warn("register", 30, "注册中心不可达，5 秒后重试")
            except RuntimeError:
                self._warn("register", 30, "注册地址或本机 IP 无效，5 秒后重试")
            if self._auth_failed or self._stop.wait(_REGISTER_RETRY_SECONDS):
                return False
        return False

    def _register_once(self) -> bool:
        ip = detect_advertise_ip(cfg.yunwei_ip, cfg.yunwei_plat)
        body = build_instance(
            app=cfg.sufa_name,
            ip=ip,
            port=advertise_port(),
            version=self.version,
            zone=cfg.yunwei_zone,
        )
        self._instance_id = str(body["instance"]["instanceId"])
        url = eureka_zone(cfg.yunwei_plat) + "apps/" + cfg.sufa_name
        request_body = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        resp = self._platform("POST", url, json_body=body, timeout=(1, 3))
        if resp.status_code in (200, 204):
            self._registered = True
            self._eureka_status = "UP"
            logger.info(
                "已注册 app=%s instance=%s health=%s zone=%s",
                cfg.sufa_name,
                self._instance_id,
                body["instance"].get("healthCheckUrl"),
                eureka_zone(cfg.yunwei_plat),
            )
            return True
        response_body = _response_text(resp)
        if resp.status_code in (401, 403):
            self._auth_failed = True
            logger.error(
                "注册被拒绝 status=%s url=%s request_body=%s response_body=%s",
                resp.status_code,
                url,
                request_body,
                response_body,
            )
            return False
        self._warn(
            "register-status",
            30,
            "注册失败 status=%s url=%s request_body=%s response_body=%s",
            resp.status_code,
            url,
            request_body,
            response_body,
        )
        return False

    def _instance_url(self) -> str:
        return (
            eureka_zone(cfg.yunwei_plat)
            + "apps/"
            + cfg.sufa_name
            + "/"
            + self._instance_id
        )

    def _heartbeat_loop(self) -> None:
        backoff_idx = 0
        while not self._stop.is_set():
            outcome = self._heartbeat_once()
            if outcome == "ok":
                backoff_idx = 0
                self._sync_runtime_status()
                delay = _RENEW_SECONDS + random.uniform(0, 2)
            elif outcome == "auth":
                self._auth_failed = True
                logger.error("心跳认证失败，停止重试")
                return
            elif outcome == "missing":
                logger.warning("心跳返回 404，重新注册")
                try:
                    self._register_once()
                except (requests.RequestException, RuntimeError):
                    self._warn("register", 30, "404 后重新注册失败")
                if self._auth_failed:
                    return
                delay = 1
            else:
                delay = _HEARTBEAT_BACKOFF[min(backoff_idx, len(_HEARTBEAT_BACKOFF) - 1)]
                backoff_idx += 1
            if self._stop.wait(delay):
                return

    def _heartbeat_once(self) -> str:
        if not self._instance_id:
            return "error"
        try:
            resp = self._platform("PUT", self._instance_url(), timeout=(1, 3))
        except requests.RequestException:
            self._warn("heartbeat", 30, "心跳请求失败")
            return "error"
        if resp.status_code in (200, 204):
            return "ok"
        if resp.status_code == 404:
            return "missing"
        if resp.status_code in (401, 403):
            return "auth"
        self._warn("heartbeat", 30, "心跳失败 status=%s", resp.status_code)
        return "error"

    def _sync_runtime_status(self) -> None:
        if self._draining or not self._registered:
            return
        want = "DOWN" if self._config_blocked or _circuit_open() else "UP"
        if want == self._eureka_status:
            return
        try:
            self._put_status(want)
        except (requests.RequestException, HttpStatus):
            self._warn("status", 60, "同步 Eureka 状态失败")
            return
        self._eureka_status = want

    def _put_status(self, status: str) -> None:
        resp = self._platform(
            "PUT",
            self._instance_url() + "/status",
            params={"value": status},
            timeout=(1, 3),
        )
        if resp.status_code not in (200, 204):
            raise HttpStatus(resp.status_code)

    def _config_loop(self) -> None:
        fail_since: float | None = None
        alerted = False
        backoff = 1.0
        first = True
        while not self._stop.is_set():
            ok = self._refresh_config()
            if first:
                self._first_config.set()
                first = False
            if self._draining:
                return
            if self._config_denied:
                return
            if ok:
                fail_since = None
                alerted = False
                backoff = 1.0
                delay = _RENEW_SECONDS + random.uniform(0, 5)
            else:
                now = time.monotonic()
                fail_since = now if fail_since is None else fail_since
                if not alerted and now - fail_since >= _CONFIG_ALERT_SECONDS:
                    logger.error("统一配置连续失败已超过 5 分钟，仍保留上一份配置")
                    alerted = True
                delay = min(backoff, 30.0)
                backoff = min(backoff * 2, 30.0)
            if self._stop.wait(delay):
                return

    def _refresh_config(self) -> bool:
        try:
            if not self._config_home:
                self._config_home = self._lookup_config_home()
            env = self._fetch_config(self._config_home)
            merged = merge_property_sources(env)
            reject_cipher(merged)
            self._commit(env, merged)
            self._config_blocked = False
            if not self._draining:
                self._sync_runtime_status()
            return True
        except CipherConfigError as exc:
            self._config_blocked = True
            logger.error("统一配置仍为密文，未应用 keys=%s", ",".join(exc.keys))
            if not self._draining:
                self._sync_runtime_status()
            return False
        except LookupError:
            self._config_home = ""
            self._warn("config", 60, "注册中心没有可用的 config 实例")
            return False
        except HttpStatus as exc:
            message = (
                "拉取统一配置失败 status=%s request=%s %s request_body=%s response_body=%s"
            )
            args = (
                exc.status,
                exc.method or "-",
                exc.url or "-",
                exc.request_body or "-",
                exc.response_body or "-",
            )
            if exc.status in (401, 403):
                self._config_denied = True
                logger.error(message, *args)
                return False
            if exc.status == 404 or exc.status >= 500:
                self._config_home = ""
            self._warn("config", 60, message, *args)
            return False
        except requests.RequestException:
            self._config_home = ""
            self._warn("config", 60, "统一配置地址不可达")
            return False
        except RuntimeError as exc:
            self._warn("config", 60, "统一配置无法应用: %s", exc)
            return False

    def _lookup_config_home(self) -> str:
        url = eureka_zone(cfg.yunwei_plat) + "apps"
        resp = self._platform("GET", url, timeout=(1, 5))
        if resp.status_code in (401, 403):
            self._auth_failed = True
            raise self._http_status(resp, "GET", url)
        if resp.status_code != 200:
            raise self._http_status(resp, "GET", url)
        return select_config_home(resp.json(), _CONFIG_SERVER_APP)

    def _fetch_config(self, home: str) -> dict[str, Any]:
        url = config_url(
            home,
            cfg.yunwei_config_app,
            cfg.yunwei_profile,
            cfg.yunwei_label,
        )
        resp = self._platform("GET", url, timeout=(1, 5))
        if resp.status_code != 200:
            raise self._http_status(resp, "GET", url)
        payload = resp.json()
        if not isinstance(payload, dict):
            raise RuntimeError("config payload")
        return payload

    @staticmethod
    def _http_status(resp: requests.Response, method: str, url: str, request_body: str = "") -> HttpStatus:
        return HttpStatus(
            resp.status_code,
            method=method,
            url=url,
            request_body=request_body or "-",
            response_body=_response_text(resp),
        )

    def _commit(self, env: dict[str, Any], merged: dict[str, Any]) -> None:
        fp = fingerprint(env, merged)
        with self._cfg_lock:
            if fp == self._fp and self._merged:
                return
            planned, held = plan_updates(merged)
            apply_plan(planned)
            old = dict(self._merged)
            old_version = self._config_version
            self._merged = dict(merged)
            self._fp = fp
            version = env.get("version")
            self._config_version = str(version) if version else fp[:18]
        changed = sorted(
            key for key in (set(old) | set(merged)) if old.get(key) != merged.get(key)
        )
        shown = changed if old_version else sorted(merged)
        logger.info(
            "统一配置已加载 app=%s profile=%s label=%s version=%s->%s keys=%s env_hold=%s",
            cfg.yunwei_config_app,
            cfg.yunwei_profile,
            cfg.yunwei_label,
            old_version or "-",
            self._config_version,
            ",".join(shown),
            ",".join(held),
        )

    def _on_stop(self) -> None:
        if not self._owner:
            return
        self._draining = True
        self._first_config.set()
        thread = self._config_thread
        if thread and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=8)
        self._unregister()

    def _unregister(self) -> None:
        if not self._registered:
            return
        self._registered = False
        if self._auth_failed:
            self._delete_quiet()
            return
        try:
            self._put_status("OUT_OF_SERVICE")
            self._eureka_status = "OUT_OF_SERVICE"
            logger.info("实例已标记 OUT_OF_SERVICE，%s 秒后注销", cfg.yunwei_drain_seconds)
        except (requests.RequestException, HttpStatus):
            logger.warning("设置 OUT_OF_SERVICE 失败，仍会注销")
        deadline = time.time() + max(0, cfg.yunwei_drain_seconds)
        while time.time() < deadline:
            time.sleep(0.2)
        self._delete_quiet()

    def _delete_quiet(self) -> None:
        if not self._instance_id:
            return
        try:
            resp = self._platform("DELETE", self._instance_url(), timeout=(1, 3))
        except requests.RequestException:
            logger.warning("注销实例失败")
            return
        if resp.status_code in (200, 204, 404):
            logger.info("已注销 instance=%s", self._instance_id)
            return
        logger.warning("注销实例失败 status=%s", resp.status_code)


def _response_text(resp: requests.Response, limit: int = 500) -> str:
    """取出响应正文，优先按 UTF-8 解码，避免中文原因变成问号。"""
    raw = resp.content or b""
    if not raw:
        return "-"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode(resp.encoding or "utf-8", errors="replace")
    text = " ".join(text.split())
    if len(text) > limit:
        return text[:limit] + "..."
    return text or "-"


def _circuit_open() -> bool:
    try:
        from backend.tools.pipeline_guard import is_circuit_open
    except Exception:
        return False
    try:
        return bool(is_circuit_open())
    except Exception:
        return False


def start_yunwei(version: str) -> None:
    """后台注册。注册中心地址为空时直接跳过。"""
    global _client, _atexit_registered
    if not cfg.yunwei_plat:
        logger.info("未配置运维平台地址，跳过注册")
        return
    with _client_lock:
        if _client is not None:
            return
        client = YunweiClient(version)
        client.start()
        _client = client
        if not _atexit_registered:
            atexit.register(stop_yunwei)
            _atexit_registered = True


def stop_yunwei() -> None:
    """标记下线并注销。重复调用无效果。"""
    global _client
    with _client_lock:
        client = _client
        _client = None
    if client is not None:
        client.stop()


def get_platform_config(key: str, default: Any = None) -> Any:
    """读取最近一份合并后的统一配置。日志里不要打印返回值。"""
    with _client_lock:
        client = _client
    if client is None:
        return default
    return client.get(key, default)
