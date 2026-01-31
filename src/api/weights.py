from __future__ import annotations

from datetime import date
from typing import Sequence, cast

from fastapi import APIRouter, Query, Request

from ..repo.outputs import WeightRepo, WeightRow
from .deps import get_config, get_engine
from .errors import ApiErrorException
from .schemas import WeightRowModel, WeightsHistoryResp, WeightsResp

router = APIRouter(prefix="/weights", tags=["weights"])


def _repo(request: Request) -> tuple[WeightRepo, str, str, str]:
    config = get_config(request)
    engine = get_engine(request)
    repo = WeightRepo(engine, schema=config.db.schema_out)
    return (
        repo,
        config.strategy.default_strategy_id,
        config.strategy.default_version,
        config.strategy.default_portfolio_id,
    )


def _weights_rows(rows: Sequence[WeightRow]) -> list[WeightRowModel]:
    return [
        WeightRowModel(
            rebalance_date=row["rebalance_date"],
            instrument_id=str(row["instrument_id"]),
            target_weight=float(row["target_weight"]),
            bucket=str(row["bucket"]),
            run_id=cast(
                str | None, cast(dict[str, object], row.get("meta_json") or {}).get("run_id")
            ),
        )
        for row in rows
    ]


@router.get("/latest", response_model=WeightsResp)
def weights_latest(
    request: Request,
    portfolio_id: str | None = None,
    strategy_id: str | None = None,
    version: str | None = None,
) -> WeightsResp:
    repo, default_strategy, default_version, default_portfolio = _repo(request)
    portfolio = portfolio_id or default_portfolio
    strategy = strategy_id or default_strategy
    ver = version or default_version
    latest = repo.get_latest(strategy_id=strategy, version=ver, portfolio_id=portfolio)
    if not latest:
        raise ApiErrorException("WEIGHTS_NOT_FOUND", "no weights found", status_code=404)
    rebalance_date, rows = latest
    weights = _weights_rows(rows)
    return WeightsResp(
        rebalance_date=rebalance_date,
        portfolio_id=portfolio,
        weights_sum=sum(w.target_weight for w in weights),
        weights=weights,
    )


@router.get("", response_model=WeightsResp)
def weights_by_date(
    request: Request,
    rebalance_date: date,
    portfolio_id: str | None = None,
    strategy_id: str | None = None,
    version: str | None = None,
) -> WeightsResp:
    repo, default_strategy, default_version, default_portfolio = _repo(request)
    portfolio = portfolio_id or default_portfolio
    strategy = strategy_id or default_strategy
    ver = version or default_version
    rows = repo.get_by_date(
        strategy_id=strategy, version=ver, portfolio_id=portfolio, rebalance_date=rebalance_date
    )
    if not rows:
        raise ApiErrorException("WEIGHTS_NOT_FOUND", "no weights found", status_code=404)
    weights = _weights_rows(rows)
    return WeightsResp(
        rebalance_date=rebalance_date,
        portfolio_id=portfolio,
        weights_sum=sum(w.target_weight for w in weights),
        weights=weights,
    )


@router.get("/history", response_model=WeightsHistoryResp)
def weights_history(
    request: Request,
    start_date: date = Query(..., alias="start_date"),
    end_date: date = Query(..., alias="end_date"),
    portfolio_id: str | None = None,
    strategy_id: str | None = None,
    version: str | None = None,
) -> WeightsHistoryResp:
    repo, default_strategy, default_version, default_portfolio = _repo(request)
    portfolio = portfolio_id or default_portfolio
    strategy = strategy_id or default_strategy
    ver = version or default_version
    rows = repo.get_history(
        strategy_id=strategy,
        version=ver,
        portfolio_id=portfolio,
        start_date=start_date,
        end_date=end_date,
    )
    weights = _weights_rows(rows)
    return WeightsHistoryResp(
        portfolio_id=portfolio,
        start_date=start_date,
        end_date=end_date,
        weights=weights,
    )
