from __future__ import annotations

import logging
from pathlib import Path

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
from .repo.inputs import BucketRepo, FactorRepo, MarketRepo, TradeCalendarRepo
from .services.auto_param_service import AutoParamService

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
        if config.auto_params.enabled:
            try:
                auto = AutoParamService(
                    bucket_repo=BucketRepo(engine, schema=config.db.schema_out),
                    market_repo=MarketRepo(engine, schema=config.db.schema_in),
                    factor_repo=FactorRepo(engine, schema=config.db.schema_in),
                    calendar_repo=TradeCalendarRepo(engine, schema=config.db.schema_in),
                    config=config,
                    output_path=Path(config.auto_params.path),
                )
                result = auto.compute_and_persist()
                auto.apply_overrides(config, result.params)
                logger.info(
                    "auto params applied (fallback=%s, warnings=%s)",
                    result.used_fallback,
                    len(result.warnings),
                )
            except Exception:
                logger.exception("auto params failed; using config defaults")
        app.state.config = config
        app.state.engine = engine

        logger.info("service starting")
        check_connection(engine)
        logger.info("startup complete")

    return app


def run() -> None:
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000)


app = create_app()
