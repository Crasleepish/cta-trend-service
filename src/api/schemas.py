from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel


class ApiError(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class HealthResp(BaseModel):
    status: str
    version: str
    time: datetime
    db_ok: bool


class UniverseModel(BaseModel):
    portfolio_id: str | None = None
    bucket_ids: list[int] | None = None
    instrument_ids: list[str] | None = None


class FeatureSetModel(BaseModel):
    enabled_features: list[str] | None = None
    feature_params: dict[str, float] | None = None


class JobRequest(BaseModel):
    rebalance_date: date
    calc_start: date | None = None
    calc_end: date | None = None
    lookback: dict[str, Any] | None = None
    universe: UniverseModel | None = None
    feature_set: FeatureSetModel | None = None
    dry_run: bool = False
    force_recompute: bool = False
    strategy_id: str | None = None
    version: str | None = None
    snapshot_id: str | None = None
    portfolio_id: str | None = None


class RunError(BaseModel):
    message: str
    stack: str | None = None


class JobResponse(BaseModel):
    run_id: str
    status: str
    job_type: str
    time_start: datetime
    time_end: datetime | None
    outputs: dict[str, Any]
    error: RunError | None = None


class WeightRowModel(BaseModel):
    rebalance_date: date
    instrument_id: str
    target_weight: float
    bucket: str
    run_id: str | None = None


class WeightsResp(BaseModel):
    rebalance_date: date
    portfolio_id: str
    weights_sum: float
    weights: list[WeightRowModel]


class WeightsHistoryResp(BaseModel):
    portfolio_id: str
    start_date: date
    end_date: date
    weights: list[WeightRowModel]


class SignalRowModel(BaseModel):
    instrument_id: str
    signal_name: str
    value: float
    bucket_id: str | None = None
    meta_json: dict[str, Any] | None = None


class SignalsResp(BaseModel):
    rebalance_date: date
    signals: list[SignalRowModel]


class RunSummary(BaseModel):
    run_id: str
    job_type: str
    status: str
    strategy_id: str
    version: str
    time_start: datetime
    time_end: datetime | None


class RunListResp(BaseModel):
    items: list[RunSummary]
    next_cursor: str | None = None


class RunDetailResp(BaseModel):
    run_id: str
    job_type: str
    status: str
    strategy_id: str
    version: str
    snapshot_id: str | None
    time_start: datetime
    time_end: datetime | None
    input_range: dict[str, Any] | None
    output_summary: dict[str, Any] | None
    error_stack: str | None = None
