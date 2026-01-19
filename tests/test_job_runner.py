from __future__ import annotations

from datetime import date, datetime, timezone

from src.services.contracts import JobType, RunContext, RunRequest, RunStatus
from src.services.job_runner import JobRunner


class FakeAudit:
    def __init__(self) -> None:
        self.started = False
        self.inputs = None
        self.finished = None

    def start_run(self, ctx, req) -> str:
        self.started = True
        return ctx.run_id

    def update_inputs(self, run_id: str, input_range: dict) -> None:
        self.inputs = (run_id, input_range)

    def finish_success(self, run_id: str, output_summary: dict) -> None:
        self.finished = ("success", run_id, output_summary)

    def finish_failed(
        self, run_id: str, err: Exception, output_summary: dict | None = None
    ) -> None:
        self.finished = ("failed", run_id, str(err))


class FakeRepo:
    def __init__(self, rows):
        self._rows = rows

    def get_range(self, *args, **kwargs):
        return self._rows

    def get_gold_range(self, *args, **kwargs):
        return self._rows


class FakeUpsertRepo:
    def __init__(self) -> None:
        self.rows = []

    def upsert_many(self, rows):
        self.rows.extend(rows)
        return len(rows)


class FakeFeatureService:
    def compute(self, *args, **kwargs):
        return [
            {
                "strategy_id": "cta_trend_v1",
                "version": "1.0.0",
                "instrument_id": "F1",
                "calc_date": date(2026, 1, 1),
                "feature_name": "T",
                "value": 1.0,
                "meta_json": None,
            }
        ]


class FakeSignalService:
    def compute(self, *args, **kwargs):
        return [
            {
                "strategy_id": "cta_trend_v1",
                "version": "1.0.0",
                "instrument_id": "F1",
                "rebalance_date": date(2026, 1, 1),
                "signal_name": "S",
                "value": 1.0,
                "meta_json": None,
            }
        ]


class FakePortfolioService:
    def compute_weights(self, *args, **kwargs):
        return [
            {
                "strategy_id": "cta_trend_v1",
                "version": "1.0.0",
                "portfolio_id": "main",
                "rebalance_date": date(2026, 1, 1),
                "instrument_id": "F1",
                "target_weight": 1.0,
                "bucket": "GROWTH",
                "meta_json": None,
            }
        ]


def test_job_runner_run_full_success() -> None:
    audit = FakeAudit()
    bucket_repo = FakeRepo([{"id": 1, "bucket_name": "GROWTH", "assets": "F1"}])
    market_repo = FakeRepo([{"date": date(2026, 1, 1)}])
    factor_repo = FakeRepo([{"date": date(2026, 1, 1)}])
    nav_repo = FakeRepo([{"date": date(2026, 1, 1)}])
    beta_repo = FakeRepo([{"date": date(2026, 1, 1)}])
    aux_repo = FakeRepo([{"date": date(2026, 1, 1)}])
    feature_repo = FakeUpsertRepo()
    signal_repo = FakeUpsertRepo()
    weight_repo = FakeUpsertRepo()

    runner = JobRunner(
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
        feature_service=FakeFeatureService(),
        signal_service=FakeSignalService(),
        portfolio_service=FakePortfolioService(),
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    ctx = RunContext(
        strategy_id="cta_trend_v1",
        version="1.0.0",
        snapshot_id=None,
        job_type=JobType.FULL,
        run_id="RUN_1",
        trigger="cli",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    req = RunRequest(
        rebalance_date=date(2026, 1, 1),
        lookback={"market_days": 1},
    )

    result = runner.run_full(ctx, req)
    assert result.status == RunStatus.SUCCESS
    assert result.outputs["rows_upserted"]["feature_daily"] == 1
    assert audit.started is True
