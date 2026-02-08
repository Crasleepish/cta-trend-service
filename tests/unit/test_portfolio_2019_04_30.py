from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from src.core.config import PortfolioConfig
from src.services.portfolio_service import PortfolioService

FIXTURE_DIR = Path("tests/fixtures/auto_params_2019-04-30")
REB_DATE = date(2019, 4, 30)

# Manual ground truth (computed via numerical-calc against DB fixtures, per WP rules)
RAW_COMPONENTS = {
    "VALUE": {
        "raw_weight_component_risk_budget": 1.0,
        "raw_weight_component_gate": 1.0,
        "raw_weight_component_trend": 0.3260451319810067,
        "raw_weight_component_inv_sigma_eff": 4.498869169929881,
        "raw_weight_component_f_sigma": 0.7004412180819205,
        "raw_weight_component_path_quality": 0.0,
    },
    "GROWTH": {
        "raw_weight_component_risk_budget": 1.0,
        "raw_weight_component_gate": 1.0,
        "raw_weight_component_trend": 0.2907995489476288,
        "raw_weight_component_inv_sigma_eff": 4.462342566219941,
        "raw_weight_component_f_sigma": 0.8370516305791624,
        "raw_weight_component_path_quality": 0.0,
    },
    "CYCLE": {
        "raw_weight_component_risk_budget": 1.0,
        "raw_weight_component_gate": 1.0,
        "raw_weight_component_trend": 0.3046726278421388,
        "raw_weight_component_inv_sigma_eff": 3.3970427081278194,
        "raw_weight_component_f_sigma": 0.6698331381409672,
        "raw_weight_component_path_quality": 0.0,
    },
    "GOLD": {
        "raw_weight_component_risk_budget": 1.0,
        "raw_weight_component_gate": 0.0,
        "raw_weight_component_trend": -0.1764591576317693,
        "raw_weight_component_inv_sigma_eff": 16.93733917610921,
        "raw_weight_component_f_sigma": 0.9774655971866564,
        "raw_weight_component_path_quality": 0.0,
    },
}

SIGMA_EFF = {
    "VALUE": 0.22227807971921665,
    "GROWTH": 0.22409754185391956,
    "CYCLE": 0.2943736908598128,
    "GOLD": 0.05904115100974892,
    "RATE": 0.021880387075949193,
    "CASH": 0.0,
}

TILT = {
    "VALUE": {
        "005562.OF": 0.250622618468586,
        "000312.OF": 0.5088234999109837,
        "003194.OF": 0.2405538816204302,
    },
    "GROWTH": {
        "004744.OF": 0.301398789451929,
        "002236.OF": 0.30065247198079775,
        "004409.OF": 0.39794873856727325,
    },
    "CYCLE": {
        "690008.OF": 0.4462122966773474,
        "164304.OF": 0.23893863245210858,
        "004195.OF": 0.31484907087054387,
    },
    "GOLD": {"004253.OF": 1.0},
    "RATE": {"003377.OF": 1.0},
    "CASH": {"000602.OF": 1.0},
}

EXPECTED_WEIGHTS = {
    "005562.OF": 0.0,
    "000312.OF": 0.0,
    "003194.OF": 0.0,
    "004744.OF": 0.0,
    "002236.OF": 0.0,
    "004409.OF": 0.0,
    "690008.OF": 0.0,
    "164304.OF": 0.0,
    "004195.OF": 0.0,
    "004253.OF": 0.0,
    "003377.OF": 0.2565822702665557,
    "000602.OF": 1.0 - 0.2565822702665557,
}

SIGMA_TARGET = 0.14824680700851597
RATE_PREF = 0.2565822702665557


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


def _load_buckets() -> list[dict[str, object]]:
    buckets = json.loads((FIXTURE_DIR / "buckets.json").read_text())
    return [
        {
            "bucket_name": b["bucket_name"],
            "assets": b.get("assets", ""),
        }
        for b in buckets
    ]


def _assert_close(actual: float, expected: float) -> None:
    assert np.isclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_portfolio_weights_2019_04_30_ground_truth() -> None:
    strategy_id = "cta_trend_v1"
    version = "1.0.0"

    bucket_repo = FakeBucketRepo(rows=_load_buckets())

    signals: list[dict[str, object]] = []
    for bucket in ["VALUE", "GROWTH", "CYCLE", "GOLD", "RATE", "CASH"]:
        signals.append(
            {
                "strategy_id": strategy_id,
                "version": version,
                "instrument_id": bucket,
                "rebalance_date": REB_DATE,
                "signal_name": "T",
                "bucket_id": bucket,
                "value": 0.0,
                "meta_json": {"bucket_id": bucket},
            }
        )
        signals.append(
            {
                "strategy_id": strategy_id,
                "version": version,
                "instrument_id": bucket,
                "rebalance_date": REB_DATE,
                "signal_name": "gate_state",
                "bucket_id": bucket,
                "value": 1.0 if bucket not in {"GOLD", "RATE", "CASH"} else 0.0,
                "meta_json": {"bucket_id": bucket},
            }
        )
        signals.append(
            {
                "strategy_id": strategy_id,
                "version": version,
                "instrument_id": bucket,
                "rebalance_date": REB_DATE,
                "signal_name": "sigma_eff",
                "bucket_id": bucket,
                "value": SIGMA_EFF.get(bucket, 0.0),
                "meta_json": {"bucket_id": bucket},
            }
        )
        signals.append(
            {
                "strategy_id": strategy_id,
                "version": version,
                "instrument_id": bucket,
                "rebalance_date": REB_DATE,
                "signal_name": "f_sigma",
                "bucket_id": bucket,
                "value": 1.0 if bucket not in {"GOLD", "RATE", "CASH"} else 0.0,
                "meta_json": {"bucket_id": bucket},
            }
        )

    for bucket, comps in RAW_COMPONENTS.items():
        for name, value in comps.items():
            signals.append(
                {
                    "strategy_id": strategy_id,
                    "version": version,
                    "instrument_id": bucket,
                    "rebalance_date": REB_DATE,
                    "signal_name": name,
                    "bucket_id": bucket,
                    "value": value,
                    "meta_json": {"bucket_id": bucket},
                }
            )

    signals.append(
        {
            "strategy_id": strategy_id,
            "version": version,
            "instrument_id": "RATE",
            "rebalance_date": REB_DATE,
            "signal_name": "rate_pref",
            "bucket_id": "RATE",
            "value": RATE_PREF,
            "meta_json": {"bucket_id": "RATE"},
        }
    )

    for bucket, assets in TILT.items():
        for asset, weight in assets.items():
            signals.append(
                {
                    "strategy_id": strategy_id,
                    "version": version,
                    "instrument_id": asset,
                    "rebalance_date": REB_DATE,
                    "signal_name": "tilt_weight",
                    "bucket_id": bucket,
                    "value": weight,
                    "meta_json": {"bucket_id": bucket},
                }
            )

    signal_repo = FakeSignalRepo(rows=signals)
    weight_repo = FakeWeightRepo(rows=[])

    config = PortfolioConfig(
        sigma_target=SIGMA_TARGET, risk_buckets=["VALUE", "GROWTH", "CYCLE", "GOLD"]
    )
    service = PortfolioService(
        bucket_repo=bucket_repo,
        signal_repo=signal_repo,
        weight_repo=weight_repo,
        config=config,
    )

    summary = service.compute_and_persist_weights_from_signals(
        run_id="RUN_TEST",
        strategy_id=strategy_id,
        version=version,
        snapshot_id=None,
        portfolio_id="main",
        rebalance_date=REB_DATE,
        universe={"bucket_ids": [1, 2, 3, 4, 5, 6]},
        dry_run=False,
        force_recompute=False,
    )

    assert summary.rows_upserted == len(weight_repo.rows)

    for row in weight_repo.rows:
        asset = row["instrument_id"]
        expected = EXPECTED_WEIGHTS[asset]
        _assert_close(float(row["target_weight"]), expected)
