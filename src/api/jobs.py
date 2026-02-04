from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, Request

from ..core.config import AppConfig
from ..repo.inputs import BetaRepo, BucketRepo, FactorRepo, MarketRepo, TradeCalendarRepo
from ..services.auto_param_service import AutoParamService
from ..services.contracts import JobType, RunContext, RunRequest, RunStatus
from ..services.dynamic_param_service import DynamicParamService
from ..services.feature_service import FeatureSetSpec
from .deps import build_job_runner, get_config, get_engine
from .errors import ApiErrorException, map_value_error
from .schemas import JobRequest, JobResponse, ParamPrepareRequest, ParamPrepareResponse, RunError

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _resolve_strategy(request: JobRequest, config: AppConfig) -> tuple[str, str, str, str | None]:
    strategy_id = request.strategy_id or config.strategy.default_strategy_id
    version = request.version or config.strategy.default_version
    snapshot_id = request.snapshot_id
    portfolio_id = request.portfolio_id or config.strategy.default_portfolio_id
    return strategy_id, version, portfolio_id, snapshot_id


def _build_universe(request: JobRequest, portfolio_id: str) -> dict[str, Any] | None:
    if request.universe is None:
        return {"portfolio_id": portfolio_id}
    universe = request.universe.model_dump(exclude_none=True)
    universe.setdefault("portfolio_id", portfolio_id)
    return universe


def _run_job(
    request: Request,
    body: JobRequest,
    job_type: JobType,
    idempotency_key: str | None,
) -> JobResponse:
    config = get_config(request)
    engine = get_engine(request)
    runner = build_job_runner(config, engine)
    run_repo = runner.audit.repo

    strategy_id, version, portfolio_id, snapshot_id = _resolve_strategy(body, config)
    if idempotency_key:
        existing = run_repo.get_by_idempotency_key(
            idempotency_key=idempotency_key,
            job_type=job_type.value,
            strategy_id=strategy_id,
            version=version,
        )
        if existing:
            error_stack = existing.get("error_stack")
            return JobResponse(
                run_id=existing["run_id"],
                status=existing["status"],
                job_type=existing["job_type"],
                time_start=existing["time_start"],
                time_end=existing["time_end"],
                outputs=existing.get("output_summary_json") or {},
                error=RunError(message=str(error_stack), stack=str(error_stack))
                if error_stack
                else None,
            )

    run_id = runner.audit.generate_run_id(datetime.now(timezone.utc))
    ctx = RunContext(
        strategy_id=strategy_id,
        version=version,
        snapshot_id=snapshot_id,
        job_type=job_type,
        run_id=run_id,
        trigger="api",
        created_at=datetime.now(timezone.utc),
    )
    req = RunRequest(
        rebalance_date=body.rebalance_date,
        calc_start=body.calc_start,
        calc_end=body.calc_end,
        lookback=body.lookback,
        universe=_build_universe(body, portfolio_id),
        dry_run=body.dry_run,
        force_recompute=body.force_recompute,
        feature_set=FeatureSetSpec(
            enabled_features=body.feature_set.enabled_features or [],
            feature_params=body.feature_set.feature_params or {},
        )
        if body.feature_set
        else None,
        tags={"idempotency_key": idempotency_key} if idempotency_key else None,
    )
    try:
        if job_type == JobType.FEATURE:
            result = runner.run_feature(ctx, req)
        elif job_type == JobType.SIGNAL:
            result = runner.run_signal(ctx, req)
        elif job_type == JobType.PORTFOLIO:
            result = runner.run_portfolio(ctx, req)
        else:
            result = runner.run_full(ctx, req)
    except ValueError as err:
        raise map_value_error(err) from err
    except ApiErrorException:
        raise
    except Exception as err:
        raise ApiErrorException("DB_ERROR", str(err), status_code=500) from err

    status = result.status.value if isinstance(result.status, RunStatus) else str(result.status)
    return JobResponse(
        run_id=result.run_id,
        status=status,
        job_type=result.job_type.value,
        time_start=result.time_start,
        time_end=result.time_end,
        outputs=result.outputs,
        error=RunError(**result.error) if result.error else None,
    )


@router.post("/full", response_model=JobResponse)
def run_full(
    request: Request,
    body: JobRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JobResponse:
    return _run_job(request, body, JobType.FULL, idempotency_key)


@router.post("/feature", response_model=JobResponse)
def run_feature(
    request: Request,
    body: JobRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JobResponse:
    return _run_job(request, body, JobType.FEATURE, idempotency_key)


@router.post("/signal", response_model=JobResponse)
def run_signal(
    request: Request,
    body: JobRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JobResponse:
    return _run_job(request, body, JobType.SIGNAL, idempotency_key)


@router.post("/portfolio", response_model=JobResponse)
def run_portfolio(
    request: Request,
    body: JobRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JobResponse:
    return _run_job(request, body, JobType.PORTFOLIO, idempotency_key)


@router.post("/param-prepare", response_model=ParamPrepareResponse)
def prepare_params(request: Request, body: ParamPrepareRequest) -> ParamPrepareResponse:
    config = get_config(request)
    engine = get_engine(request)

    bucket_repo = BucketRepo(engine, schema=config.db.schema_out)
    market_repo = MarketRepo(engine, schema=config.db.schema_in)
    factor_repo = FactorRepo(engine, schema=config.db.schema_in)
    beta_repo = BetaRepo(engine, schema=config.db.schema_in)
    calendar_repo = TradeCalendarRepo(engine, schema=config.db.schema_in)

    warnings: list[str] = []
    auto_fallback: bool | None = None
    dynamic_fallback: bool | None = None

    if config.auto_params.enabled:
        auto_service = AutoParamService(
            bucket_repo=bucket_repo,
            market_repo=market_repo,
            factor_repo=factor_repo,
            calendar_repo=calendar_repo,
            config=config,
            output_path=Path(config.auto_params.path),
        )
        auto_result = auto_service.compute_and_persist(as_of=body.as_of)
        auto_fallback = auto_result.used_fallback
        warnings.extend(auto_result.warnings)

    dynamic_service = DynamicParamService(
        bucket_repo=bucket_repo,
        market_repo=market_repo,
        beta_repo=beta_repo,
        calendar_repo=calendar_repo,
        config=config,
        output_path=Path(config.dynamic_params.path),
    )
    dynamic_result = dynamic_service.compute_and_persist(as_of=body.as_of)
    dynamic_fallback = dynamic_result.used_fallback
    warnings.extend(dynamic_result.warnings)

    return ParamPrepareResponse(
        auto_params_path=config.auto_params.path if config.auto_params.enabled else None,
        dynamic_params_path=config.dynamic_params.path,
        auto_enabled=config.auto_params.enabled,
        auto_fallback=auto_fallback,
        dynamic_fallback=dynamic_fallback,
        warnings=warnings,
    )
