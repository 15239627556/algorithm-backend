from __future__ import annotations

import gzip
import faulthandler
import logging
import logging.handlers
import os
import shutil
from pathlib import Path


# 手动修改区：服务日志全局配置。
# SERVICE_LOG_FILE：当前正在写入的日志文件路径；相对路径以启动目录为准。
# SERVICE_LOG_MAX_MB：单个日志文件最大大小，达到后自动轮转并 gzip 压缩旧文件。
# SERVICE_LOG_BACKUP_COUNT：最多保留多少个压缩备份；超过后删除最旧备份。
# SERVICE_LOG_LEVEL：日志级别，可选 DEBUG/INFO/WARNING/ERROR。
SERVICE_LOG_FILE = "logs/multi_pipeline_server.log"
SERVICE_LOG_MAX_MB = 50
SERVICE_LOG_BACKUP_COUNT = 10
SERVICE_LOG_LEVEL = "INFO"

DEFAULT_LOG_FILE = SERVICE_LOG_FILE
DEFAULT_LOG_MAX_MB = SERVICE_LOG_MAX_MB
DEFAULT_LOG_BACKUP_COUNT = SERVICE_LOG_BACKUP_COUNT
DEFAULT_LOG_LEVEL = SERVICE_LOG_LEVEL
_FAULTHANDLER_FP = None


class GzipRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler variant that gzips rotated log files."""

    def doRollover(self) -> None:
        if self.stream:
            self.stream.close()
            self.stream = None

        base = Path(self.baseFilename)
        if self.backupCount > 0:
            oldest = Path(f"{base}.{self.backupCount}.gz")
            if oldest.exists():
                oldest.unlink()
            for ix in range(self.backupCount - 1, 0, -1):
                src = Path(f"{base}.{ix}.gz")
                dst = Path(f"{base}.{ix + 1}.gz")
                if src.exists():
                    src.rename(dst)
            if base.exists():
                with base.open("rb") as src, gzip.open(f"{base}.1.gz", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                base.unlink()
        elif base.exists():
            base.unlink()

        if not self.delay:
            self.stream = self._open()


def parse_log_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    raw = (level or DEFAULT_LOG_LEVEL).strip().upper()
    value = logging.getLevelName(raw)
    if isinstance(value, int):
        return value
    raise ValueError(f"无效日志级别: {level!r}")


def configure_service_logging(
    *,
    log_file: str | None = None,
    log_max_mb: int = DEFAULT_LOG_MAX_MB,
    log_backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
    log_level: str | int = DEFAULT_LOG_LEVEL,
) -> str:
    path = Path((log_file or "").strip() or DEFAULT_LOG_FILE)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    max_bytes = max(1, int(log_max_mb)) * 1024 * 1024
    backup_count = max(0, int(log_backup_count))
    level = parse_log_level(log_level)

    handler = GzipRotatingFileHandler(
        str(path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    for old_handler in list(root.handlers):
        root.removeHandler(old_handler)
        old_handler.close()
    root.addHandler(handler)
    root.setLevel(level)

    logging.captureWarnings(True)
    logging.getLogger("py.warnings").setLevel(logging.WARNING)
    return os.path.abspath(path)


def configure_dedicated_file_logger(
    logger_name: str,
    *,
    log_file: str,
    log_max_mb: int = DEFAULT_LOG_MAX_MB,
    log_backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
    log_level: str | int = DEFAULT_LOG_LEVEL,
) -> str:
    path = Path(log_file.strip())
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    handler = GzipRotatingFileHandler(
        str(path),
        maxBytes=max(1, int(log_max_mb)) * 1024 * 1024,
        backupCount=max(0, int(log_backup_count)),
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)s %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    dedicated_logger = logging.getLogger(logger_name)
    for old_handler in list(dedicated_logger.handlers):
        dedicated_logger.removeHandler(old_handler)
        old_handler.close()
    dedicated_logger.addHandler(handler)
    dedicated_logger.setLevel(parse_log_level(log_level))
    dedicated_logger.propagate = False
    return os.path.abspath(path)


def enable_faulthandler_to_log(log_file: str) -> None:
    global _FAULTHANDLER_FP
    if _FAULTHANDLER_FP is not None:
        return
    _FAULTHANDLER_FP = open(log_file, "a", encoding="utf-8")
    faulthandler.enable(file=_FAULTHANDLER_FP, all_threads=True)
