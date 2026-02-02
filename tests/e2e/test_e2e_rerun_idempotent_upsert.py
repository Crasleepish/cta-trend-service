# Ref: WP §11.3 (Determinism/Auditability)

from __future__ import annotations

from datetime import date, datetime, timezone

from src.services.contracts import JobType, RunContext, RunRequest
from src.services.job_runner import JobRunner
from src.services.run_audit_service import RunAuditService
from tests.e2e.test_e2e_determinism_full import (
    MockFeatureService,
    MockPortfolioService,
    MockSignalService,
)
from tests.helpers.fakes import (
    FakeAuxRepo,
    FakeBucketRepo,
    FakeDateRepo,
    FakeRunRepo,
    FakeUpsertRepo,
)


def test_e2e_rerun_idempotent_upsert() -> None:
    today = date(2026, 1, 16)
    dates = [today]

    bucket_repo = FakeBucketRepo(buckets=[{"bucket_name": "RISK", "assets": "FND_A,FND_B"}])
    market_repo = FakeDateRepo(dates)
    factor_repo = FakeDateRepo(dates)
    nav_repo = FakeDateRepo(dates)
    beta_repo = FakeDateRepo(dates)
    aux_repo = FakeAuxRepo(dates)
    calendar_repo = FakeDateRepo(dates)

    feature_repo = FakeUpsertRepo(
        key_fields=["strategy_id", "version", "instrument_id", "calc_date", "feature_name"]
    )
    signal_repo = FakeUpsertRepo(
        key_fields=["strategy_id", "version", "instrument_id", "rebalance_date", "signal_name"]
    )
    weight_repo = FakeUpsertRepo(
        key_fields=[
            "strategy_id",
            "version",
            "portfolio_id",
            "rebalance_date",
            "instrument_id",
        ]
    )
    run_repo = FakeRunRepo(key_fields=["run_id"])

    audit = RunAuditService(repo=run_repo)
    runner = JobRunner(
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
        feature_service=MockFeatureService(),
        signal_service=MockSignalService(),
        portfolio_service=MockPortfolioService(),
        clock=lambda: datetime(2026, 1, 16, 12, 0, tzinfo=timezone.utc),
    )

    ctx = RunContext(
        strategy_id="s1",
        version="v1",
        snapshot_id="snap",
        job_type=JobType.FULL,
        run_id="RUN_TEST",
        trigger="cli",
        created_at=datetime(2026, 1, 16, 12, 0, tzinfo=timezone.utc),
    )
    req = RunRequest(
        rebalance_date=today,
        calc_start=date(2025, 12, 1),
        calc_end=today,
        universe={"bucket_ids": [1]},
        feature_set=None,
    )

    runner.run_full(ctx, req)
    first_counts = (len(feature_repo.rows), len(signal_repo.rows), len(weight_repo.rows))

    runner.run_full(ctx, req)
    second_counts = (len(feature_repo.rows), len(signal_repo.rows), len(weight_repo.rows))

    assert first_counts == second_counts
    assert len(run_repo.rows) == 1
    assert run_repo.rows[("RUN_TEST",)]["status"] == "SUCCESS"
