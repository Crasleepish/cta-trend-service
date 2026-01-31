from __future__ import annotations

import base64
from datetime import datetime

from fastapi import APIRouter, Query, Request

from ..repo.outputs import RunRepo
from .deps import get_config, get_engine
from .errors import ApiErrorException
from .schemas import RunDetailResp, RunListResp, RunSummary

router = APIRouter(prefix="/runs", tags=["runs"])


def _repo(request: Request) -> RunRepo:
    config = get_config(request)
    engine = get_engine(request)
    return RunRepo(engine, schema=config.db.schema_out)


def _encode_cursor(time_start: datetime, run_id: str) -> str:
    raw = f"{time_start.isoformat()}|{run_id}"
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("utf-8")


def _decode_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("utf-8")).decode("utf-8")
        time_part, run_id = raw.split("|", 1)
        return datetime.fromisoformat(time_part), run_id
    except Exception as err:
        raise ApiErrorException("BAD_REQUEST", "invalid cursor", status_code=400) from err


@router.get("", response_model=RunListResp)
def list_runs(
    request: Request,
    limit: int = Query(default=20, ge=1, le=200),
    cursor: str | None = None,
    status: str | None = None,
    job_type: str | None = None,
    strategy_id: str | None = None,
    version: str | None = None,
) -> RunListResp:
    repo = _repo(request)
    cursor_value = _decode_cursor(cursor) if cursor else None
    rows = repo.list_runs(
        limit=limit + 1,
        cursor=cursor_value,
        status=status,
        job_type=job_type,
        strategy_id=strategy_id,
        version=version,
    )
    next_cursor = None
    if len(rows) > limit:
        last = rows[limit - 1]
        next_cursor = _encode_cursor(last["time_start"], last["run_id"])
        rows = rows[:limit]

    items = [
        RunSummary(
            run_id=row["run_id"],
            job_type=row["job_type"],
            status=row["status"],
            strategy_id=row["strategy_id"],
            version=row["version"],
            time_start=row["time_start"],
            time_end=row["time_end"],
        )
        for row in rows
    ]
    return RunListResp(items=items, next_cursor=next_cursor)


@router.get("/{run_id}", response_model=RunDetailResp)
def run_detail(request: Request, run_id: str) -> RunDetailResp:
    repo = _repo(request)
    row = repo.get_by_id(run_id)
    if not row:
        raise ApiErrorException("RUN_NOT_FOUND", "run_id not found", status_code=404)
    return RunDetailResp(
        run_id=row["run_id"],
        job_type=row["job_type"],
        status=row["status"],
        strategy_id=row["strategy_id"],
        version=row["version"],
        snapshot_id=row["snapshot_id"],
        time_start=row["time_start"],
        time_end=row["time_end"],
        input_range=row["input_range_json"],
        output_summary=row["output_summary_json"],
        error_stack=row["error_stack"],
    )
