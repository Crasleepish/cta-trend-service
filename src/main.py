from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI

from .api.health import router as health_router
from .core.config import load_app_config
from .core.db import check_connection, create_engine_from_config
from .core.logging import setup_logging

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(health_router)

    @app.on_event("startup")
    def startup() -> None:
        config = load_app_config()
        setup_logging(config.logging)
        engine = create_engine_from_config(config)
        app.state.config = config
        app.state.engine = engine

        logger.info("service starting")
        check_connection(engine)
        logger.info("startup complete")

    return app


def run() -> None:
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000)


app = create_app()
