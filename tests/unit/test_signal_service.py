from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from src.core.config import SignalConfig
from src.services.signal_service import SignalService


@dataclass
class FakeBucketRepo:
    rows: list[dict[str, object]]

    def get_range(self, *_args, **_kwargs):
        return self.rows


@dataclass
class FakeFeatureWeeklyRepo:
    rows: list[dict[str, object]]

    def get_range(
        self,
        *,
        strategy_id: str,
        version: str,
        rebalance_date: date,
        instrument_ids: list[str] | None = None,
        feature_names: list[str] | None = None,
    ):
        data = [row for row in self.rows if row["rebalance_date"] == rebalance_date]
        if instrument_ids:
            data = [row for row in data if row["instrument_id"] in instrument_ids]
        if feature_names:
            data = [row for row in data if row["feature_name"] in feature_names]
        return data


@dataclass
class FakeFactorRepo:
    rows: list[dict[str, object]]

    def get_range(self, *_args, **_kwargs):
        return self.rows


@dataclass
class FakeBetaRepo:
    rows: list[dict[str, object]]

    def get_range(self, codes, *_args, **_kwargs):
        return [row for row in self.rows if row["code"] in codes]


@dataclass
class FakeSignalRepo:
    rows: list[dict[str, object]]

    def upsert_many(self, rows):
        self.rows.extend(rows)
        return len(rows)


def _rows_to_csv(rows: list[dict[str, object]], path: Path) -> None:
    df = pd.DataFrame(rows)
    df = df[["rebalance_date", "instrument_id", "signal_name", "value"]]
    df = df.sort_values(["rebalance_date", "instrument_id", "signal_name"])
    df.to_csv(path, index=False)


def test_signal_service_tilt_and_golden(tmp_path: Path) -> None:
    rebalance = date(2026, 1, 16)
    strategy_id = "cta_trend_v1"
    version = "1.0.0"

    bucket_repo = FakeBucketRepo(
        rows=[
            {
                "bucket_name": "GROWTH",
                "assets": "A1,B1",
                "bucket_proxy": "IDX_G",
            },
            {"bucket_name": "RATE", "assets": "R1", "bucket_proxy": "IDX_R"},
        ]
    )

    feature_rows = []
    for bucket in ["GROWTH", "RATE"]:
        for name, value in [
            ("T", 1.0),
            ("gate_state", 1.0),
            ("sigma_ann", 0.2),
            ("sigma_eff", 0.2),
            ("f_sigma", 1.0),
        ]:
            feature_rows.append(
                {
                    "strategy_id": strategy_id,
                    "version": version,
                    "instrument_id": bucket,
                    "rebalance_date": rebalance,
                    "feature_name": name,
                    "value": value,
                }
            )
    feature_rows.extend(
        [
            {
                "strategy_id": strategy_id,
                "version": version,
                "instrument_id": "RATE",
                "rebalance_date": rebalance,
                "feature_name": "T_RATE",
                "value": 0.5,
            },
            {
                "strategy_id": strategy_id,
                "version": version,
                "instrument_id": "RATE",
                "rebalance_date": rebalance,
                "feature_name": "rate_pref",
                "value": 0.7,
            },
        ]
    )

    feature_repo = FakeFeatureWeeklyRepo(rows=feature_rows)

    factor_repo = FakeFactorRepo(rows=[{"date": rebalance, "SMB": 1.0, "QMJ": 0.0}])

    beta_repo = FakeBetaRepo(
        rows=[
            {"code": "A1", "date": rebalance, "SMB": 1.0, "QMJ": 0.0},
            {"code": "B1", "date": rebalance, "SMB": 0.0, "QMJ": 1.0},
            {"code": "R1", "date": rebalance, "SMB": 0.0, "QMJ": 0.0},
        ]
    )

    signal_repo = FakeSignalRepo(rows=[])

    config = SignalConfig(
        tilt_factors=["SMB", "QMJ"],
        tilt_lookback_days=60,
        tilt_scales={"SMB": 1.0, "QMJ": 1.0},
        tilt_eps=1e-12,
        tilt_temperature=1.0,
    )

    service = SignalService(
        bucket_repo=bucket_repo,
        feature_weekly_repo=feature_repo,
        factor_repo=factor_repo,
        beta_repo=beta_repo,
        signal_repo=signal_repo,
        config=config,
    )

    summary = service.compute_and_persist_signals(
        run_id="RUN_1",
        strategy_id=strategy_id,
        version=version,
        snapshot_id=None,
        rebalance_date=rebalance,
        universe={"bucket_ids": [1, 2]},
        dry_run=False,
        force_recompute=False,
    )

    assert summary.rows_upserted == len(signal_repo.rows)

    weights = [row for row in signal_repo.rows if row["signal_name"] == "tilt_weight"]
    assert len(weights) == 3
    growth_weights = [row["value"] for row in weights if row["instrument_id"] in {"A1", "B1"}]
    assert abs(sum(growth_weights) - 1.0) < 1e-8
    assert all(weight >= 0 for weight in growth_weights)

    output_path = tmp_path / "signals.csv"
    _rows_to_csv(signal_repo.rows, output_path)

    golden_path = (
        Path("tests/fixtures/golden/signal_weekly")
        / f"signals__{strategy_id}__{version}__{rebalance.isoformat()}.csv"
    )
    assert output_path.read_text() == golden_path.read_text()
