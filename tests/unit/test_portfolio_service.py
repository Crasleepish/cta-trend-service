from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

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


def _rows_to_csv(rows: list[dict[str, object]], path: Path) -> None:
    df = pd.DataFrame(rows)
    df = df[["rebalance_date", "instrument_id", "target_weight", "bucket"]]
    df = df.sort_values(["rebalance_date", "instrument_id"])
    df.to_csv(path, index=False)


def test_portfolio_service_from_signals(tmp_path: Path) -> None:
    rebalance = date(2026, 1, 16)
    strategy_id = "cta_trend_v1"
    version = "1.0.0"
    bucket_repo = FakeBucketRepo(
        rows=[
            {"bucket_name": "GROWTH", "assets": "A1,B1"},
            {"bucket_name": "RATE", "assets": "R1"},
            {"bucket_name": "CASH", "assets": "C1"},
        ]
    )

    signals = []
    for bucket in ["GROWTH", "RATE", "CASH"]:
        signals.extend(
            [
                {
                    "strategy_id": strategy_id,
                    "version": version,
                    "instrument_id": bucket,
                    "rebalance_date": rebalance,
                    "signal_name": "T",
                    "bucket_id": bucket,
                    "value": 1.0,
                    "meta_json": {"bucket_id": bucket},
                },
                {
                    "strategy_id": strategy_id,
                    "version": version,
                    "instrument_id": bucket,
                    "rebalance_date": rebalance,
                    "signal_name": "gate_state",
                    "bucket_id": bucket,
                    "value": 1.0,
                    "meta_json": {"bucket_id": bucket},
                },
                {
                    "strategy_id": strategy_id,
                    "version": version,
                    "instrument_id": bucket,
                    "rebalance_date": rebalance,
                    "signal_name": "sigma_eff",
                    "bucket_id": bucket,
                    "value": 0.2,
                    "meta_json": {"bucket_id": bucket},
                },
                {
                    "strategy_id": strategy_id,
                    "version": version,
                    "instrument_id": bucket,
                    "rebalance_date": rebalance,
                    "signal_name": "f_sigma",
                    "bucket_id": bucket,
                    "value": 1.0,
                    "meta_json": {"bucket_id": bucket},
                },
            ]
        )

    for name, value in [
        ("raw_weight_component_risk_budget", 1.0),
        ("raw_weight_component_gate", 1.0),
        ("raw_weight_component_trend", 1.0),
        ("raw_weight_component_inv_sigma_eff", 5.0),
        ("raw_weight_component_f_sigma", 1.0),
        ("raw_weight_component_path_quality", 1.0),
    ]:
        signals.append(
            {
                "strategy_id": strategy_id,
                "version": version,
                "instrument_id": "GROWTH",
                "rebalance_date": rebalance,
                "signal_name": name,
                "bucket_id": "GROWTH",
                "value": value,
                "meta_json": {"bucket_id": "GROWTH"},
            }
        )

    signals.append(
        {
            "strategy_id": strategy_id,
            "version": version,
            "instrument_id": "RATE",
            "rebalance_date": rebalance,
            "signal_name": "rate_pref",
            "bucket_id": "RATE",
            "value": 0.7,
            "meta_json": {"bucket_id": "RATE"},
        }
    )

    for asset, weight, bucket in [
        ("A1", 0.6, "GROWTH"),
        ("B1", 0.4, "GROWTH"),
        ("R1", 1.0, "RATE"),
        ("C1", 1.0, "CASH"),
    ]:
        signals.append(
            {
                "strategy_id": strategy_id,
                "version": version,
                "instrument_id": asset,
                "rebalance_date": rebalance,
                "signal_name": "tilt_weight",
                "bucket_id": bucket,
                "value": weight,
                "meta_json": {"bucket_id": bucket},
            }
        )

    signal_repo = FakeSignalRepo(rows=signals)
    weight_repo = FakeWeightRepo(rows=[])

    config = PortfolioConfig(sigma_target=0.1, risk_buckets=["GROWTH"])
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
        universe={"bucket_ids": [1, 2, 3]},
        dry_run=False,
        force_recompute=False,
    )

    assert summary.rows_upserted == len(weight_repo.rows)

    output_path = tmp_path / "weights.csv"
    _rows_to_csv(weight_repo.rows, output_path)

    golden_path = (
        Path("tests/fixtures/golden/portfolio_weight_weekly")
        / f"weights__{strategy_id}__{version}__{rebalance.isoformat()}.csv"
    )
    assert output_path.read_text() == golden_path.read_text()


def test_portfolio_service_rejects_defensive_in_risk_buckets() -> None:
    config = PortfolioConfig(risk_buckets=["RATE"])
    service = PortfolioService(
        bucket_repo=FakeBucketRepo(rows=[{"bucket_name": "RATE", "assets": "R1"}]),
        signal_repo=FakeSignalRepo(rows=[]),
        weight_repo=FakeWeightRepo(rows=[]),
        config=config,
    )
    try:
        service._risk_buckets(["RATE", "CASH"])
    except ValueError as exc:
        assert "risk_buckets cannot include defensive buckets" in str(exc)
    else:
        raise AssertionError("expected ValueError for defensive buckets in risk_buckets")
