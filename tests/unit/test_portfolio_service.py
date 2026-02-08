from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
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
                    "signal_name": "down_drift",
                    "bucket_id": bucket,
                    "value": 0.0,
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


def test_portfolio_exec_ewma_applied() -> None:
    prev_rows = [
        {
            "strategy_id": "cta_trend_v1",
            "version": "1.0.0",
            "portfolio_id": "main",
            "rebalance_date": date(2024, 11, 8),
            "instrument_id": "000312.OF",
            "target_weight": 0.0,
            "run_id": "RUN_PREV",
        },
        {
            "strategy_id": "cta_trend_v1",
            "version": "1.0.0",
            "portfolio_id": "main",
            "rebalance_date": date(2024, 11, 8),
            "instrument_id": "003377.OF",
            "target_weight": 0.5152373022374269,
            "run_id": "RUN_PREV",
        },
        {
            "strategy_id": "cta_trend_v1",
            "version": "1.0.0",
            "portfolio_id": "main",
            "rebalance_date": date(2024, 11, 8),
            "instrument_id": "000602.OF",
            "target_weight": 0.4847626977625731,
            "run_id": "RUN_PREV",
        },
    ]
    target_rows = [
        {
            "strategy_id": "cta_trend_v1",
            "version": "1.0.0",
            "portfolio_id": "main",
            "rebalance_date": date(2024, 11, 15),
            "instrument_id": "000312.OF",
            "target_weight": 0.025445591366767037,
            "run_id": "RUN_NOW",
        },
        {
            "strategy_id": "cta_trend_v1",
            "version": "1.0.0",
            "portfolio_id": "main",
            "rebalance_date": date(2024, 11, 15),
            "instrument_id": "003377.OF",
            "target_weight": 0.13888051464707354,
            "run_id": "RUN_NOW",
        },
        {
            "strategy_id": "cta_trend_v1",
            "version": "1.0.0",
            "portfolio_id": "main",
            "rebalance_date": date(2024, 11, 15),
            "instrument_id": "000602.OF",
            "target_weight": 0.1243510133407597,
            "run_id": "RUN_NOW",
        },
    ]
    config = PortfolioConfig(alpha_on=0.35, alpha_off=0.75)
    repo = FakeWeightRepo(rows=prev_rows.copy())
    service = PortfolioService(
        bucket_repo=FakeBucketRepo(rows=[]),
        signal_repo=FakeSignalRepo(rows=[]),
        weight_repo=repo,
        config=config,
    )
    signal_df = pd.DataFrame(
        [
            {"instrument_id": "VALUE", "signal_name": "down_drift", "value": 0.0},
            {"instrument_id": "RATE", "signal_name": "down_drift", "value": 0.0},
            {"instrument_id": "CASH", "signal_name": "down_drift", "value": 0.0},
        ]
    )
    bucket_assets = {"VALUE": ["000312.OF"], "RATE": ["003377.OF"], "CASH": ["000602.OF"]}
    executed = service._apply_execution_controls(
        target_weights=target_rows,
        signal_df=signal_df,
        bucket_assets=bucket_assets,
        strategy_id="cta_trend_v1",
        version="1.0.0",
        portfolio_id="main",
        rebalance_date=date(2024, 11, 15),
    )
    weights = {row["instrument_id"]: row["target_weight"] for row in executed}
    assert weights["000312.OF"] == pytest.approx(0.011858213478048113, rel=1e-9)
    assert weights["003377.OF"] == pytest.approx(0.5106438574681452, rel=1e-9)
    assert weights["000602.OF"] == pytest.approx(0.47749792905380667, rel=1e-9)


def test_portfolio_dead_band_keeps_prev() -> None:
    prev_rows = [
        {
            "strategy_id": "cta_trend_v1",
            "version": "1.0.0",
            "portfolio_id": "main",
            "rebalance_date": date(2024, 11, 8),
            "instrument_id": "A1",
            "target_weight": 0.5,
            "run_id": "RUN_PREV",
        },
        {
            "strategy_id": "cta_trend_v1",
            "version": "1.0.0",
            "portfolio_id": "main",
            "rebalance_date": date(2024, 11, 8),
            "instrument_id": "C1",
            "target_weight": 0.5,
            "run_id": "RUN_PREV",
        },
    ]
    target_rows = [
        {
            "strategy_id": "cta_trend_v1",
            "version": "1.0.0",
            "portfolio_id": "main",
            "rebalance_date": date(2024, 11, 15),
            "instrument_id": "A1",
            "target_weight": 0.52,
            "run_id": "RUN_NOW",
        },
        {
            "strategy_id": "cta_trend_v1",
            "version": "1.0.0",
            "portfolio_id": "main",
            "rebalance_date": date(2024, 11, 15),
            "instrument_id": "C1",
            "target_weight": 0.48,
            "run_id": "RUN_NOW",
        },
    ]
    config = PortfolioConfig(alpha_on=1.0, dead_band=0.05)
    repo = FakeWeightRepo(rows=prev_rows.copy())
    service = PortfolioService(
        bucket_repo=FakeBucketRepo(rows=[]),
        signal_repo=FakeSignalRepo(rows=[]),
        weight_repo=repo,
        config=config,
    )
    signal_df = pd.DataFrame(
        [
            {"instrument_id": "A", "signal_name": "down_drift", "value": 0.0},
            {"instrument_id": "C", "signal_name": "down_drift", "value": 0.0},
        ]
    )
    executed = service._apply_execution_controls(
        target_weights=target_rows,
        signal_df=signal_df,
        bucket_assets={"A": ["A1"], "C": ["C1"]},
        strategy_id="cta_trend_v1",
        version="1.0.0",
        portfolio_id="main",
        rebalance_date=date(2024, 11, 15),
    )
    weights = {row["instrument_id"]: row["target_weight"] for row in executed}
    assert weights["A1"] == pytest.approx(0.5)
    assert weights["C1"] == pytest.approx(0.5)


def test_portfolio_caps_redistribute_to_defensive() -> None:
    config = PortfolioConfig(max_weight_asset=0.4, max_weight_bucket=0.6)
    service = PortfolioService(
        bucket_repo=FakeBucketRepo(rows=[]),
        signal_repo=FakeSignalRepo(rows=[]),
        weight_repo=FakeWeightRepo(rows=[]),
        config=config,
    )
    bucket_assets = {
        "VALUE": ["A1", "A2"],
        "RATE": ["R1"],
        "CASH": ["C1"],
    }
    target = [
        {
            "strategy_id": "cta_trend_v1",
            "version": "1.0.0",
            "portfolio_id": "main",
            "rebalance_date": date(2024, 11, 15),
            "instrument_id": "A1",
            "target_weight": 0.7,
            "bucket": "VALUE",
            "run_id": "RUN_NOW",
        },
        {
            "strategy_id": "cta_trend_v1",
            "version": "1.0.0",
            "portfolio_id": "main",
            "rebalance_date": date(2024, 11, 15),
            "instrument_id": "A2",
            "target_weight": 0.2,
            "bucket": "VALUE",
            "run_id": "RUN_NOW",
        },
        {
            "strategy_id": "cta_trend_v1",
            "version": "1.0.0",
            "portfolio_id": "main",
            "rebalance_date": date(2024, 11, 15),
            "instrument_id": "R1",
            "target_weight": 0.05,
            "bucket": "RATE",
            "run_id": "RUN_NOW",
        },
        {
            "strategy_id": "cta_trend_v1",
            "version": "1.0.0",
            "portfolio_id": "main",
            "rebalance_date": date(2024, 11, 15),
            "instrument_id": "C1",
            "target_weight": 0.05,
            "bucket": "CASH",
            "run_id": "RUN_NOW",
        },
    ]
    bucket_by_asset = {a: b for b, assets in bucket_assets.items() for a in assets}
    capped = service._apply_caps(weights=target, bucket_by_asset=bucket_by_asset)
    weights = {row["instrument_id"]: row["target_weight"] for row in capped}
    assert weights["A1"] == pytest.approx(0.4)
    assert weights["A2"] == pytest.approx(0.2)
    assert weights["R1"] + weights["C1"] == pytest.approx(0.4)


def test_portfolio_exec_uses_alpha_off_when_down_drift() -> None:
    prev_rows = [
        {
            "strategy_id": "cta_trend_v1",
            "version": "1.0.0",
            "portfolio_id": "main",
            "rebalance_date": date(2024, 11, 8),
            "instrument_id": "A1",
            "target_weight": 0.2,
            "run_id": "RUN_PREV",
        },
        {
            "strategy_id": "cta_trend_v1",
            "version": "1.0.0",
            "portfolio_id": "main",
            "rebalance_date": date(2024, 11, 8),
            "instrument_id": "C1",
            "target_weight": 0.8,
            "run_id": "RUN_PREV",
        },
    ]
    target_rows = [
        {
            "strategy_id": "cta_trend_v1",
            "version": "1.0.0",
            "portfolio_id": "main",
            "rebalance_date": date(2024, 11, 15),
            "instrument_id": "A1",
            "target_weight": 0.6,
            "run_id": "RUN_NOW",
        },
        {
            "strategy_id": "cta_trend_v1",
            "version": "1.0.0",
            "portfolio_id": "main",
            "rebalance_date": date(2024, 11, 15),
            "instrument_id": "C1",
            "target_weight": 0.4,
            "run_id": "RUN_NOW",
        },
    ]
    config = PortfolioConfig(
        alpha_on=0.35,
        alpha_off=0.75,
        dead_band=0.0,
        max_weight_asset=1.0,
        max_weight_bucket=1.0,
    )
    repo = FakeWeightRepo(rows=prev_rows.copy())
    service = PortfolioService(
        bucket_repo=FakeBucketRepo(rows=[]),
        signal_repo=FakeSignalRepo(rows=[]),
        weight_repo=repo,
        config=config,
    )
    signal_df = pd.DataFrame(
        [
            {"instrument_id": "RISK", "signal_name": "down_drift", "value": 1.0},
            {"instrument_id": "CASH", "signal_name": "down_drift", "value": 0.0},
        ]
    )
    executed = service._apply_execution_controls(
        target_weights=target_rows,
        signal_df=signal_df,
        bucket_assets={"RISK": ["A1"], "CASH": ["C1"]},
        strategy_id="cta_trend_v1",
        version="1.0.0",
        portfolio_id="main",
        rebalance_date=date(2024, 11, 15),
    )
    weights = {row["instrument_id"]: row["target_weight"] for row in executed}
    prev_a1 = 0.2
    prev_c1 = 0.8
    target_a1 = 0.6
    target_c1 = 0.4
    expected_a1_off = (1.0 - config.alpha_off) * prev_a1 + config.alpha_off * target_a1
    expected_a1_on = (1.0 - config.alpha_on) * prev_a1 + config.alpha_on * target_a1
    expected_c1_on = (1.0 - config.alpha_on) * prev_c1 + config.alpha_on * target_c1
    total = expected_a1_off + expected_c1_on
    assert weights["A1"] == pytest.approx(expected_a1_off / total, rel=1e-9)
    assert weights["C1"] == pytest.approx(expected_c1_on / total, rel=1e-9)
    assert weights["A1"] != pytest.approx(expected_a1_on / total, rel=1e-9)
