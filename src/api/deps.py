from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

from fastapi import Request
from sqlalchemy.engine import Engine

from ..core.config import AppConfig
from ..repo.inputs import (
    AuxRepo,
    BetaRepo,
    BucketRepo,
    FactorRepo,
    MarketRepo,
    NavRepo,
    TradeCalendarRepo,
)
from ..repo.outputs import FeatureRepo, FeatureWeeklySampleRepo, RunRepo, SignalRepo, WeightRepo
from ..services.auto_param_service import AutoParamService
from ..services.backtest_service import BacktestService
from ..services.dynamic_param_service import DynamicParamService
from ..services.feature_service import FeatureService
from ..services.job_runner import JobRunner
from ..services.portfolio_service import PortfolioService
from ..services.run_audit_service import RunAuditService
from ..services.signal_service import SignalService


def get_config(request: Request) -> AppConfig:
    return cast(AppConfig, request.app.state.config)


def get_engine(request: Request) -> Engine:
    return cast(Engine, request.app.state.engine)


logger = logging.getLogger(__name__)


def _load_params_file(path: str) -> dict[str, object] | None:
    payload_path = Path(path)
    if not payload_path.exists():
        return None
    return cast(dict[str, object], json.loads(payload_path.read_text()))


def build_job_runner(config: AppConfig, engine: Engine) -> JobRunner:
    bucket_repo = BucketRepo(engine, schema=config.db.schema_out)
    market_repo = MarketRepo(engine, schema=config.db.schema_in)
    factor_repo = FactorRepo(engine, schema=config.db.schema_in)
    nav_repo = NavRepo(engine, schema=config.db.schema_in)
    beta_repo = BetaRepo(engine, schema=config.db.schema_in)
    aux_repo = AuxRepo(engine, schema=config.db.schema_in)
    calendar_repo = TradeCalendarRepo(engine, schema=config.db.schema_in)

    feature_repo = FeatureRepo(engine, schema=config.db.schema_out)
    feature_weekly_repo = FeatureWeeklySampleRepo(engine, schema=config.db.schema_out)
    signal_repo = SignalRepo(engine, schema=config.db.schema_out)
    weight_repo = WeightRepo(engine, schema=config.db.schema_out)
    run_repo = RunRepo(engine, schema=config.db.schema_out)

    auto_params = None
    if config.auto_params.enabled:
        auto_params = _load_params_file(config.auto_params.path)
        if auto_params is None:
            logger.warning(
                "auto_params missing at %s; using config defaults",
                config.auto_params.path,
            )
        else:
            AutoParamService(
                bucket_repo=bucket_repo,
                market_repo=market_repo,
                factor_repo=factor_repo,
                calendar_repo=calendar_repo,
                config=config,
                output_path=Path(config.auto_params.path),
            ).apply_overrides(config, cast(dict[str, object], auto_params.get("params", {})))

    dynamic_params = _load_params_file(config.dynamic_params.path)
    if dynamic_params is None:
        raise ValueError(f"dynamic params missing: {config.dynamic_params.path}")
    DynamicParamService(
        bucket_repo=bucket_repo,
        market_repo=market_repo,
        beta_repo=beta_repo,
        calendar_repo=calendar_repo,
        config=config,
        output_path=Path(config.dynamic_params.path),
    ).apply_overrides(config, cast(dict[str, object], dynamic_params.get("params", {})))

    feature_service = FeatureService(
        bucket_repo=bucket_repo,
        market_repo=market_repo,
        calendar_repo=calendar_repo,
        feature_repo=feature_repo,
        feature_weekly_repo=feature_weekly_repo,
        config=config.features,
    )
    signal_service = SignalService(
        bucket_repo=bucket_repo,
        feature_weekly_repo=feature_weekly_repo,
        factor_repo=factor_repo,
        beta_repo=beta_repo,
        signal_repo=signal_repo,
        config=config.signals,
    )
    portfolio_service = PortfolioService(
        bucket_repo=bucket_repo,
        signal_repo=signal_repo,
        weight_repo=weight_repo,
        config=config.portfolio,
    )

    audit = RunAuditService(repo=run_repo)
    return JobRunner(
        audit=audit,
        bucket_repo=bucket_repo,
        market_repo=market_repo,
        factor_repo=factor_repo,
        nav_repo=nav_repo,
        beta_repo=beta_repo,
        aux_repo=aux_repo,
        calendar_repo=calendar_repo,
        feature_repo=feature_repo,
        signal_repo=signal_repo,
        weight_repo=weight_repo,
        feature_service=feature_service,
        signal_service=signal_service,
        portfolio_service=portfolio_service,
    )


def build_backtest_service(config: AppConfig, engine: Engine) -> BacktestService:
    return BacktestService(
        weight_repo=WeightRepo(engine, schema=config.db.schema_out),
        nav_repo=NavRepo(engine, schema=config.db.schema_in),
        calendar_repo=TradeCalendarRepo(engine, schema=config.db.schema_in),
    )
