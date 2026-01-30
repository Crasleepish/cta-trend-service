# Ref: WP §11.3 (Determinism/Auditability)

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from src.services.contracts import JobType, RunContext, RunRequest
from src.services.feature_service import FeatureRunSummary, FeatureSetSpec
from src.services.job_runner import JobRunner
from src.services.run_audit_service import RunAuditService
from tests.helpers.fakes import (
    FakeAuxRepo,
    FakeBucketRepo,
    FakeDateRepo,
    FakeRunRepo,
    FakeUpsertRepo,
)


@dataclass
class MockFeatureService:
    def compute_and_persist(
        self,
        *,
        run_id: str,
        strategy_id: str,
        version: str,
        snapshot_id: str | None,
        rebalance_date: date,
        calc_start: date,
        calc_end: date,
        universe: dict[str, Any],
        feature_set: FeatureSetSpec | None,
        dry_run: bool,
        force_recompute: bool,
    ) -> FeatureRunSummary:
        _ = (
            run_id,
            strategy_id,
            version,
            snapshot_id,
            rebalance_date,
            calc_start,
            calc_end,
            feature_set,
            dry_run,
            force_recompute,
        )
        return FeatureRunSummary(
            calc_date_range=(calc_start, calc_end),
            instruments_count=len(universe["instrument_ids"]),
            features_generated=["T"],
            rows_upserted_daily=len(universe["instrument_ids"]),
            rows_upserted_weekly=len(universe["instrument_ids"]),
            warnings=[],
            coverage={},
        )


@dataclass
class MockSignalService:
    def compute_and_persist_signals(
        self,
        *,
        run_id: str,
        strategy_id: str,
        version: str,
        snapshot_id: str | None,
        rebalance_date: date,
        universe: dict[str, Any],
        dry_run: bool,
        force_recompute: bool,
    ):
        _ = (
            run_id,
            strategy_id,
            version,
            snapshot_id,
            rebalance_date,
            universe,
            dry_run,
            force_recompute,
        )
        return type(
            "Summary",
            (),
            {"rows_upserted": len(universe["instrument_ids"])},
        )()


@dataclass
class MockPortfolioService:
    def compute_weights(
        self, rebalance_date: date, universe: dict[str, Any], snapshot_id: str | None, dry_run: bool
    ) -> list[dict[str, Any]]:
        instruments = list(universe["instrument_ids"])
        weight = 1.0 / len(instruments)
        return [
            {
                "strategy_id": "s1",
                "version": "v1",
                "portfolio_id": universe["portfolio_id"],
                "rebalance_date": rebalance_date,
                "instrument_id": inst,
                "target_weight": weight,
                "bucket": "RISK",
                "meta_json": None,
            }
            for inst in instruments
        ]


def _hash_rows(*repos: FakeUpsertRepo) -> str:
    payload = []
    for repo in repos:
        rows = [repo.rows[key] for key in sorted(repo.rows.keys())]
        payload.append(rows)
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_runner() -> tuple[JobRunner, FakeUpsertRepo, FakeUpsertRepo, FakeUpsertRepo]:
    today = date(2026, 1, 16)
    dates = [today]

    bucket_repo = FakeBucketRepo(buckets=[{"bucket_name": "RISK", "assets": "FND_A,FND_B"}])
    market_repo = FakeDateRepo(dates)
    factor_repo = FakeDateRepo(dates)
    nav_repo = FakeDateRepo(dates)
    beta_repo = FakeDateRepo(dates)
    aux_repo = FakeAuxRepo(dates)

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
        feature_repo=feature_repo,
        signal_repo=signal_repo,
        weight_repo=weight_repo,
        feature_service=MockFeatureService(),
        signal_service=MockSignalService(),
        portfolio_service=MockPortfolioService(),
        clock=lambda: datetime(2026, 1, 16, 12, 0, tzinfo=timezone.utc),
    )
    return runner, feature_repo, signal_repo, weight_repo


def test_e2e_determinism_full() -> None:
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
        rebalance_date=date(2026, 1, 16),
        calc_start=date(2025, 12, 1),
        calc_end=date(2026, 1, 16),
        universe={"bucket_ids": [1]},
        feature_set=None,
    )

    runner1, feature1, signal1, weight1 = _build_runner()
    runner1.run_full(ctx, req)
    hash1 = _hash_rows(feature1, signal1, weight1)

    runner2, feature2, signal2, weight2 = _build_runner()
    runner2.run_full(ctx, req)
    hash2 = _hash_rows(feature2, signal2, weight2)

    assert hash1 == hash2
