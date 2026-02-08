from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest

from src.core.config import PortfolioConfig
from src.services.portfolio_service import PortfolioService


@dataclass
class FakeBucketRepo:
    rows: list[dict[str, object]]

    def get_range(self, *_args, **_kwargs):
        return self.rows


@dataclass
class FakeSignalRepo:
    rows: list[dict[str, object]]

    def get_range(self, *, strategy_id: str, version: str, rebalance_date: date, **_kwargs):
        return [
            row
            for row in self.rows
            if row["strategy_id"] == strategy_id
            and row["version"] == version
            and row["rebalance_date"] == rebalance_date
        ]


@dataclass
class FakeWeightRepo:
    rows: list[dict[str, object]]

    def upsert_many(self, rows):
        self.rows.extend(rows)
        return len(rows)

    def get_latest_date_before(
        self,
        *,
        strategy_id: str,
        version: str,
        portfolio_id: str,
        rebalance_date: date,
    ):
        del strategy_id, version, portfolio_id
        prev_dates = [
            row["rebalance_date"] for row in self.rows if row["rebalance_date"] < rebalance_date
        ]
        return max(prev_dates) if prev_dates else None

    def get_by_date(
        self,
        *,
        strategy_id: str,
        version: str,
        portfolio_id: str,
        rebalance_date: date,
    ):
        del strategy_id, version, portfolio_id
        return [row for row in self.rows if row["rebalance_date"] == rebalance_date]


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def test_wp10_execution_2024_11_15(tmp_path: Path) -> None:
    rebalance = date(2024, 11, 15)
    strategy_id = "cta_trend_v1"
    version = "1.0.0"
    bucket_repo = FakeBucketRepo(
        rows=[
            {"bucket_name": "VALUE", "assets": "005562.OF,000312.OF,003194.OF"},
            {"bucket_name": "GROWTH", "assets": "004744.OF,002236.OF,004409.OF"},
            {"bucket_name": "CYCLE", "assets": "690008.OF,164304.OF,004195.OF"},
            {"bucket_name": "GOLD", "assets": "004253.OF"},
            {"bucket_name": "RATE", "assets": "003377.OF"},
            {"bucket_name": "CASH", "assets": "000602.OF"},
        ]
    )

    fixture_dir = Path("tests/fixtures/wp10_2024-11-15")
    signals = []
    for row in _load_csv(fixture_dir / "signals.csv"):
        signals.append(
            {
                "strategy_id": strategy_id,
                "version": version,
                "instrument_id": row["instrument_id"],
                "rebalance_date": rebalance,
                "signal_name": row["signal_name"],
                "bucket_id": row["instrument_id"],
                "value": float(row["value"]),
                "meta_json": {"bucket_id": row["instrument_id"]},
            }
        )
    for bucket in ["VALUE", "GROWTH", "CYCLE", "GOLD", "RATE", "CASH"]:
        signals.append(
            {
                "strategy_id": strategy_id,
                "version": version,
                "instrument_id": bucket,
                "rebalance_date": rebalance,
                "signal_name": "down_drift",
                "bucket_id": bucket,
                "value": 0.0,
                "meta_json": {"bucket_id": bucket},
            }
        )
    for row in _load_csv(fixture_dir / "tilt.csv"):
        signals.append(
            {
                "strategy_id": strategy_id,
                "version": version,
                "instrument_id": row["instrument_id"],
                "rebalance_date": rebalance,
                "signal_name": row["signal_name"],
                "bucket_id": row["bucket_id"],
                "value": float(row["value"]),
                "meta_json": {"bucket_id": row["bucket_id"]},
            }
        )

    prev_rows = []
    for row in _load_csv(fixture_dir / "prev_weights.csv"):
        prev_rows.append(
            {
                "strategy_id": strategy_id,
                "version": version,
                "portfolio_id": "main",
                "rebalance_date": date.fromisoformat(row["rebalance_date"]),
                "instrument_id": row["instrument_id"],
                "target_weight": float(row["target_weight"]),
                "run_id": "RUN_PREV",
            }
        )

    signal_repo = FakeSignalRepo(rows=signals)
    weight_repo = FakeWeightRepo(rows=prev_rows)
    config = PortfolioConfig(
        sigma_target=0.14824680700851597,
        alpha_on=0.35,
        alpha_off=0.75,
        risk_buckets=["VALUE", "GROWTH", "CYCLE", "GOLD"],
    )
    service = PortfolioService(
        bucket_repo=bucket_repo,
        signal_repo=signal_repo,
        weight_repo=weight_repo,
        config=config,
    )

    summary = service.compute_and_persist_weights_from_signals(
        run_id="RUN_1",
        strategy_id=strategy_id,
        version=version,
        snapshot_id=None,
        portfolio_id="main",
        rebalance_date=rebalance,
        universe={"bucket_ids": [1, 2, 3, 4, 5, 6]},
        dry_run=False,
        force_recompute=False,
    )

    assert summary.rows_upserted == 12
    weights = {row["instrument_id"]: row["target_weight"] for row in weight_repo.rows}
    assert weights["005562.OF"] == pytest.approx(0.025638354904825713, rel=1e-9)
    assert weights["000312.OF"] == pytest.approx(0.008905956978368464, rel=1e-9)
    assert weights["003194.OF"] == pytest.approx(0.2233246533210642, rel=1e-9)
    assert weights["003377.OF"] == pytest.approx(0.38351242658080326, rel=1e-9)
    assert weights["000602.OF"] == pytest.approx(0.3586186082149384, rel=1e-9)
