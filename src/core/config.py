from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import yaml
from pydantic import BaseModel, Field


class DbConfig(BaseModel):
    dsn: str
    schema_in: str = "public"
    schema_out: str = "cta"


class LoggingConfig(BaseModel):
    dir: str = "./logs"
    level: str = "INFO"
    retention_days: int = 365
    prefix: str = "app"


class AppConfig(BaseModel):
    env: str = "dev"
    db: DbConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_app_config(path: Path | None = None) -> AppConfig:
    config_path = path or Path(os.getenv("APP_CONFIG_PATH", "config/app.yaml"))
    if not config_path.exists():
        raise FileNotFoundError(f"app config not found: {config_path}")
    data: Any = yaml.safe_load(config_path.read_text())
    return AppConfig.model_validate(data)
