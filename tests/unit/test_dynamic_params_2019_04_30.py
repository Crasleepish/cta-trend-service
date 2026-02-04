from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.core.config import AppConfig, FeatureConfig, PortfolioConfig, SignalConfig
from src.services.dynamic_param_service import DynamicParamService

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
class FakeBetaRepo:
    rows: list[Mapping[str, Any]]

    def get_range(
        self,
        fund_codes: Iterable[str],
        start_date: date,
        end_date: date,
        order_by_date: bool = True,
    ) -> list[Mapping[str, Any]]:
        codes = set(fund_codes)
        rows = [
            row
            for row in self.rows
            if row["code"] in codes and start_date <= row["date"] <= end_date
        ]
        if order_by_date:
            rows.sort(key=lambda r: (r["code"], r["date"]))
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
    return AppConfig(
        db={"dsn": "", "schema_in": "public", "schema_out": "cta"},
        features=FeatureConfig(),
        signals=SignalConfig(),
        portfolio=PortfolioConfig(),
        auto_params={"enabled": True, "auto_param_window_size": 4, "min_points": 200},
    )


def test_dynamic_params_2019_04_30_ground_truth(tmp_path: Path) -> None:
    config = _build_config()

    buckets = _load_buckets(FIXTURE_DIR / "buckets.json")
    calendar_rows = _load_csv(FIXTURE_DIR / "trade_calendar.csv")
    index_rows = _load_csv(FIXTURE_DIR / "index_hist.csv")
    beta_rows = _load_csv(FIXTURE_DIR / "fund_beta.csv")

    service = DynamicParamService(
        bucket_repo=FakeBucketRepo(buckets),
        market_repo=FakeMarketRepo(index_rows),
        beta_repo=FakeBetaRepo(beta_rows),
        calendar_repo=FakeCalendarRepo([r["date"] for r in calendar_rows]),
        config=config,
        output_path=tmp_path / "dynamic_params.json",
    )

    result = service.compute_and_persist(as_of=date(2019, 4, 30))

    expected = json.loads((FIXTURE_DIR / "dynamic_params.json").read_text())["params"]
    actual = result.params

    def _assert_close_map(actual_map: dict[str, float], expected_map: dict[str, float]) -> None:
        assert set(actual_map.keys()) == set(expected_map.keys())
        for key, expected_value in expected_map.items():
            assert np.isclose(actual_map[key], expected_value, rtol=1e-10, atol=1e-12)

    for key in ["theta_on", "theta_off", "theta_minus", "sigma_min", "sigma_max", "kappa_sigma"]:
        _assert_close_map(actual["features"][key], expected["features"][key])

    _assert_close_map(actual["signals"]["tilt_scales"], expected["signals"]["tilt_scales"])
