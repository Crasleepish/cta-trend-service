from __future__ import annotations

from typing import cast

from fastapi import Request
from sqlalchemy.engine import Engine

from ..core.config import AppConfig
from ..repo.inputs import AuxRepo, BetaRepo, BucketRepo, FactorRepo, MarketRepo, NavRepo
from ..repo.outputs import FeatureRepo, FeatureWeeklySampleRepo, RunRepo, SignalRepo, WeightRepo
from ..services.feature_service import FeatureService
from ..services.job_runner import JobRunner
from ..services.portfolio_service import PortfolioService
from ..services.run_audit_service import RunAuditService
from ..services.signal_service import SignalService


def get_config(request: Request) -> AppConfig:
    return cast(AppConfig, request.app.state.config)


def get_engine(request: Request) -> Engine:
    return cast(Engine, request.app.state.engine)


def build_job_runner(config: AppConfig, engine: Engine) -> JobRunner:
    bucket_repo = BucketRepo(engine, schema=config.db.schema_out)
    market_repo = MarketRepo(engine, schema=config.db.schema_in)
    factor_repo = FactorRepo(engine, schema=config.db.schema_in)
    nav_repo = NavRepo(engine, schema=config.db.schema_in)
    beta_repo = BetaRepo(engine, schema=config.db.schema_in)
    aux_repo = AuxRepo(engine, schema=config.db.schema_in)

    feature_repo = FeatureRepo(engine, schema=config.db.schema_out)
    feature_weekly_repo = FeatureWeeklySampleRepo(engine, schema=config.db.schema_out)
    signal_repo = SignalRepo(engine, schema=config.db.schema_out)
    weight_repo = WeightRepo(engine, schema=config.db.schema_out)
    run_repo = RunRepo(engine, schema=config.db.schema_out)

    feature_service = FeatureService(
        bucket_repo=bucket_repo,
        market_repo=market_repo,
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
        feature_repo=feature_repo,
        signal_repo=signal_repo,
        weight_repo=weight_repo,
        feature_service=feature_service,
        signal_service=signal_service,
        portfolio_service=portfolio_service,
    )
