from __future__ import annotations

from datetime import date
from typing import Sequence, cast

from fastapi import APIRouter, Request

from ..repo.outputs import SignalRepo, SignalRow
from .deps import get_config, get_engine
from .errors import ApiErrorException
from .schemas import SignalRowModel, SignalsResp

router = APIRouter(prefix="/signals", tags=["signals"])


def _repo(request: Request) -> tuple[SignalRepo, str, str]:
    config = get_config(request)
    engine = get_engine(request)
    repo = SignalRepo(engine, schema=config.db.schema_out)
    return repo, config.strategy.default_strategy_id, config.strategy.default_version


def _rows_to_models(rows: Sequence[SignalRow], include_meta: bool) -> list[SignalRowModel]:
    return [
        SignalRowModel(
            instrument_id=str(row["instrument_id"]),
            signal_name=str(row["signal_name"]),
            value=float(row["value"]),
            bucket_id=cast(str | None, row.get("bucket_id")),
            meta_json=cast(dict[str, object] | None, row.get("meta_json"))
            if include_meta
            else None,
        )
        for row in rows
    ]


@router.get("", response_model=SignalsResp)
def signals_by_date(
    request: Request,
    rebalance_date: date,
    strategy_id: str | None = None,
    version: str | None = None,
    include_meta: bool = False,
    instrument_id: str | None = None,
    signal_name_prefix: str | None = None,
) -> SignalsResp:
    repo, default_strategy, default_version = _repo(request)
    strategy = strategy_id or default_strategy
    ver = version or default_version
    rows = repo.get_range(
        strategy_id=strategy,
        version=ver,
        rebalance_date=rebalance_date,
        instrument_ids=[instrument_id] if instrument_id else None,
        signal_name_prefix=signal_name_prefix,
    )
    if not rows:
        raise ApiErrorException("SIGNAL_INCOMPLETE", "no signals found", status_code=404)
    return SignalsResp(
        rebalance_date=rebalance_date,
        signals=_rows_to_models(rows, include_meta),
    )


@router.get("/latest", response_model=SignalsResp)
def signals_latest(
    request: Request,
    strategy_id: str | None = None,
    version: str | None = None,
    include_meta: bool = False,
    instrument_id: str | None = None,
    signal_name_prefix: str | None = None,
) -> SignalsResp:
    repo, default_strategy, default_version = _repo(request)
    strategy = strategy_id or default_strategy
    ver = version or default_version
    latest = repo.get_latest(strategy_id=strategy, version=ver)
    if not latest:
        raise ApiErrorException("SIGNAL_INCOMPLETE", "no signals found", status_code=404)
    rebalance_date, rows = latest
    if instrument_id or signal_name_prefix:
        rows = repo.get_range(
            strategy_id=strategy,
            version=ver,
            rebalance_date=rebalance_date,
            instrument_ids=[instrument_id] if instrument_id else None,
            signal_name_prefix=signal_name_prefix,
        )
    return SignalsResp(
        rebalance_date=rebalance_date,
        signals=_rows_to_models(rows, include_meta),
    )
