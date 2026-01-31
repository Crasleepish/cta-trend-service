from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request

from ..core.db import check_connection
from .schemas import HealthResp

router = APIRouter()


@router.get("/health", response_model=HealthResp)
def health(request: Request) -> HealthResp:
    config = request.app.state.config
    engine = request.app.state.engine
    db_ok = True
    try:
        check_connection(engine)
    except Exception:
        db_ok = False
    return HealthResp(
        status="ok",
        version=config.strategy.default_version,
        time=datetime.now(timezone.utc),
        db_ok=db_ok,
    )
