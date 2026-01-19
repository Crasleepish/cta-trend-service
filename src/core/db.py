from __future__ import annotations

import logging

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import AppConfig

logger = logging.getLogger(__name__)


def create_engine_from_config(config: AppConfig) -> Engine:
    return create_engine(config.db.dsn, future=True)


def check_connection(engine: Engine) -> None:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    logger.info("database connection ok")
