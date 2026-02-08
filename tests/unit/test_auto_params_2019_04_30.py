from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.core.config import AppConfig, FeatureConfig, PortfolioConfig, SignalConfig
from src.services.auto_param_service import AutoParamService

FIXTURE_DIR = Path("tests/fixtures/auto_params_2019-04-30")


@dataclass
class FakeBucketRepo:
    rows: list[Mapping[str, Any]]

    def get_range(self) -> list[Mapping[str, Any]]:
        return list(self.rows)


@dataclass
class FakeMarketRepo:
    rows: list[Mapping[str, Any]]

    def get_range(
        self,
        index_codes: Iterable[str],
        start_date: date,
        end_date: date,
        order_by_date: bool = True,
    ) -> list[Mapping[str, Any]]:
        codes = set(index_codes)
        rows = [
            row
            for row in self.rows
            if row["index_code"] in codes and start_date <= row["date"] <= end_date
        ]
        if order_by_date:
            rows.sort(key=lambda r: (r["index_code"], r["date"]))
        return rows


@dataclass
class FakeFactorRepo:
    rows: list[Mapping[str, Any]]

    def get_range(
        self, start_date: date, end_date: date, order_by_date: bool = True
    ) -> list[Mapping[str, Any]]:
        rows = [row for row in self.rows if start_date <= row["date"] <= end_date]
        if order_by_date:
            rows.sort(key=lambda r: r["date"])
        return rows


@dataclass
class FakeCalendarRepo:
    dates: list[date]

    def get_range(self, start_date: date, end_date: date) -> list[Mapping[str, Any]]:
        return [{"date": d} for d in self.dates if start_date <= d <= end_date]


def _load_csv(path: Path) -> list[Mapping[str, Any]]:
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    return df.to_dict(orient="records")


def _load_buckets(path: Path) -> list[Mapping[str, Any]]:
    return json.loads(path.read_text())


def _build_config() -> AppConfig:
    # Minimal config for auto param computation.
    return AppConfig(
        db={"dsn": "", "schema_in": "public", "schema_out": "cta"},
        features=FeatureConfig(),
        signals=SignalConfig(),
        portfolio=PortfolioConfig(),
        auto_params={"enabled": True, "auto_param_window_size": 4, "min_points": 200},
    )


def test_auto_params_2019_04_30_ground_truth(tmp_path: Path) -> None:
    config = _build_config()

    buckets = _load_buckets(FIXTURE_DIR / "buckets.json")
    calendar_rows = _load_csv(FIXTURE_DIR / "trade_calendar.csv")
    index_rows = _load_csv(FIXTURE_DIR / "index_hist.csv")
    factor_rows = _load_csv(FIXTURE_DIR / "market_factors.csv")
    service = AutoParamService(
        bucket_repo=FakeBucketRepo(buckets),
        market_repo=FakeMarketRepo(index_rows),
        factor_repo=FakeFactorRepo(factor_rows),
        calendar_repo=FakeCalendarRepo([r["date"] for r in calendar_rows]),
        config=config,
        output_path=tmp_path / "auto_params.json",
    )

    result = service.compute_and_persist(as_of=date(2019, 4, 30))

    expected = {
        "features": {
            "theta_rate": 0.31861928041504267,
            "x0": 0.6179678437467533,
            "path_quality_gamma": 1.0253638828787857,
        },
        "portfolio": {"sigma_target": 0.14824680700851597},
        "signals": {
            "tilt_lookback_days": 20,
        },
    }

    actual = result.params

    assert np.isclose(
        actual["features"]["theta_rate"],
        expected["features"]["theta_rate"],
        rtol=1e-10,
        atol=1e-12,
    )
    assert np.isclose(
        actual["features"]["x0"],
        expected["features"]["x0"],
        rtol=1e-10,
        atol=1e-12,
    )
    assert np.isclose(
        actual["features"]["path_quality_gamma"],
        expected["features"]["path_quality_gamma"],
        rtol=1e-10,
        atol=1e-12,
    )
    assert np.isclose(
        actual["portfolio"]["sigma_target"],
        expected["portfolio"]["sigma_target"],
        rtol=1e-10,
        atol=1e-12,
    )
    assert actual["signals"]["tilt_lookback_days"] == expected["signals"]["tilt_lookback_days"]
