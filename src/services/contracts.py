from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any


class JobType(str, Enum):
    FEATURE = "FEATURE"
    SIGNAL = "SIGNAL"
    PORTFOLIO = "PORTFOLIO"
    FULL = "FULL"


class RunStatus(str, Enum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


@dataclass(frozen=True)
class RunContext:
    strategy_id: str
    version: str
    snapshot_id: str | None
    job_type: JobType
    run_id: str
    trigger: str
    created_at: datetime


@dataclass(frozen=True)
class RunRequest:
    rebalance_date: date
    calc_start: date | None = None
    calc_end: date | None = None
    lookback: dict[str, Any] | None = None
    universe: dict[str, Any] | None = None
    dry_run: bool = False
    force_recompute: bool = False
    tags: dict[str, str] | None = None


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: RunStatus
    job_type: JobType
    time_start: datetime
    time_end: datetime
    input_range: dict[str, Any]
    outputs: dict[str, Any]
    error: dict[str, str] | None
