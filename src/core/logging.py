from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import gzip
import json
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import re
from .config import LoggingConfig


def _gzip_file(path: Path) -> None:
    gz_path = path.with_suffix(path.suffix + ".gz")
    if gz_path.exists():
        return
    with path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
        dst.write(src.read())
    path.unlink()


@dataclass(frozen=True)
class LogPaths:
    log_dir: Path
    prefix: str

    def current_log_path(self, now: datetime | None = None) -> Path:
        date = (now or datetime.now()).date()
        return self.log_dir / f"{self.prefix}_{date:%Y-%m-%d}.log"


class DailyRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, log_paths: LogPaths) -> None:
        self._log_paths = log_paths
        filename = self._log_paths.current_log_path()
        super().__init__(
            filename=str(filename),
            when="midnight",
            interval=1,
            backupCount=0,
            encoding="utf-8",
            delay=True,
            utc=False,
        )

    def doRollover(self) -> None:
        if self.stream:
            self.stream.close()
            self.stream = None  # type: ignore[assignment]

        current_path = Path(self.baseFilename)
        if current_path.exists():
            _gzip_file(current_path)

        self.baseFilename = str(self._log_paths.current_log_path())
        if not self.delay:
            self.stream = self._open()

        current_time = int(datetime.now().timestamp())
        self.rolloverAt = self.computeRollover(current_time)


def cleanup_old_logs(log_dir: Path, prefix: str, retention_days: int) -> None:
    if retention_days <= 0:
        return
    date_re = re.compile(
        rf"^{re.escape(prefix)}_(\d{{4}}-\d{{2}}-\d{{2}})\.log(?:\.gz)?$"
    )
    cutoff = datetime.now().date() - timedelta(days=retention_days)
    for path in log_dir.iterdir():
        match = date_re.match(path.name)
        if not match:
            continue
        try:
            file_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except ValueError:
            continue
        if file_date < cutoff:
            path.unlink()


def setup_logging(config: LoggingConfig) -> None:
    log_dir = Path(config.dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    cleanup_old_logs(log_dir, config.prefix, config.retention_days)

    log_paths = LogPaths(log_dir=log_dir, prefix=config.prefix)
    file_handler = DailyRotatingFileHandler(log_paths)
    formatter = _StructuredFormatter()
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(config.level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


class _StructuredFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        created = self.formatTime(record, self.datefmt)
        message = record.getMessage()
        extra = _extract_extra(record)
        extra_json = json.dumps(extra, separators=(",", ":")) if extra else "{}"
        return f"{created} | {record.levelname} | {record.name} | {message} | extra={extra_json}"


def _extract_extra(record: logging.LogRecord) -> dict[str, object]:
    standard = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }
    extras: dict[str, object] = {}
    for key, value in record.__dict__.items():
        if key in standard:
            continue
        extras[key] = value
    return extras
