from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI

from .api.backtests import router as backtests_router
from .api.errors import ApiErrorException, error_response
from .api.health import router as health_router
from .api.jobs import router as jobs_router
from .api.runs import router as runs_router
from .api.signals import router as signals_router
from .api.weights import router as weights_router
from .core.config import load_app_config
from .core.db import check_connection, create_engine_from_config
from .core.logging import setup_logging

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(health_router)
    app.include_router(jobs_router)
    app.include_router(weights_router)
    app.include_router(runs_router)
    app.include_router(signals_router)
    app.include_router(backtests_router)

    app.add_exception_handler(ApiErrorException, error_response)

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
