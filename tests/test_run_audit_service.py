from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock

from src.services.contracts import JobType, RunContext, RunRequest
from src.services.run_audit_service import RunAuditService


def test_run_audit_service_lifecycle_calls_repo() -> None:
    repo = MagicMock()
    audit = RunAuditService(repo=repo)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ctx = RunContext(
        strategy_id="cta_trend_v1",
        version="1.0.0",
        snapshot_id=None,
        job_type=JobType.FULL,
        run_id="RUN_TEST",
        trigger="cli",
        created_at=now,
    )
    req = RunRequest(rebalance_date=date(2026, 1, 1))

    audit.start_run(ctx, req)
    assert repo.upsert_many.called

    audit.update_inputs(ctx.run_id, {"calc_range": ["2026-01-01", "2026-01-01"]})
    repo.update_fields.assert_called_with(
        "RUN_TEST", {"input_range_json": {"calc_range": ["2026-01-01", "2026-01-01"]}}
    )

    audit.finish_success(ctx.run_id, {"rows": 1})
    assert repo.update_fields.called

    audit.finish_failed(ctx.run_id, RuntimeError("boom"))
    assert repo.update_fields.called
