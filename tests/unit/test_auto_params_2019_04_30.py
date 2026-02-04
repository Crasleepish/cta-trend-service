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
    # Minimal config for auto param computation.
    return AppConfig(
        db={"dsn": "", "schema_in": "public", "schema_out": "cta"},
        features=FeatureConfig(),
        signals=SignalConfig(),
        portfolio=PortfolioConfig(),
        auto_params={"enabled": True, "window_years": 4, "min_points": 200},
    )


def test_auto_params_2019_04_30_ground_truth(tmp_path: Path) -> None:
    config = _build_config()

    buckets = _load_buckets(FIXTURE_DIR / "buckets.json")
    calendar_rows = _load_csv(FIXTURE_DIR / "trade_calendar.csv")
    index_rows = _load_csv(FIXTURE_DIR / "index_hist.csv")
    factor_rows = _load_csv(FIXTURE_DIR / "market_factors.csv")
    beta_rows = _load_csv(FIXTURE_DIR / "fund_beta.csv")

    service = AutoParamService(
        bucket_repo=FakeBucketRepo(buckets),
        market_repo=FakeMarketRepo(index_rows),
        factor_repo=FakeFactorRepo(factor_rows),
        beta_repo=FakeBetaRepo(beta_rows),
        calendar_repo=FakeCalendarRepo([r["date"] for r in calendar_rows]),
        config=config,
        output_path=tmp_path / "auto_params.json",
    )

    result = service.compute_and_persist(as_of=date(2019, 4, 30))

    expected = {
        "features": {
            "theta_on": {
                "GROWTH": 430.61704391201624,
                "CYCLE": 945.1856827759191,
                "GOLD": 57.48896791718857,
                "RATE": 117.17354340819385,
                "VALUE": 3564.036467680621,
            },
            "theta_off": {
                "GROWTH": 104.05952138758168,
                "CYCLE": 345.73638452763515,
                "GOLD": 27.116638849039436,
                "RATE": 64.58427673056205,
                "VALUE": 1763.8045945109038,
            },
            "theta_minus": {
                "GROWTH": 384.9381371041566,
                "CYCLE": 879.8022441596108,
                "GOLD": 39.055242453537794,
                "RATE": 78.23722535348767,
                "VALUE": 2423.5316208757686,
            },
            "sigma_min": {
                "GROWTH": 0.1314713607995856,
                "CYCLE": 0.1606855582683171,
                "GOLD": 0.05289235090804694,
                "RATE": 0.009839687772979198,
                "VALUE": 0.09722297364904037,
            },
            "sigma_max": {
                "GROWTH": 0.41000721009032004,
                "CYCLE": 0.37757622096055643,
                "GOLD": 0.13070952598946672,
                "RATE": 0.02019030264056841,
                "VALUE": 0.32010346621631464,
            },
            "kappa_sigma": {
                "GROWTH": 0.11451924950711895,
                "CYCLE": 0.11280086642415488,
                "GOLD": 0.019064648396289765,
                "RATE": 0.004182330832910001,
                "VALUE": 0.12068702150468007,
            },
            "x0": {
                "GROWTH": 0.5809523771126269,
                "CYCLE": 0.5908500645527218,
                "GOLD": 0.7067768294384347,
                "RATE": 0.9615507370621633,
                "VALUE": 0.7586783273557413,
            },
            "path_quality_gamma": {
                "GROWTH": 2.0,
                "CYCLE": 2.0,
                "GOLD": 2.0,
                "RATE": 2.0,
                "VALUE": 2.0,
            },
            "theta_rate": 56.20589403674875,
        },
        "portfolio": {"sigma_target": 0.14824680700851597},
        "signals": {
            "tilt_lookback_days": 20,
            "tilt_scales": {
                "SMB": 0.2958124966247713,
                "QMJ": 0.7452523711657151,
            },
            "tilt_eps": 1e-12,
            "tilt_temperature": 1.0,
        },
    }

    actual = result.params

    def _assert_close_map(actual_map: dict[str, float], expected_map: dict[str, float]) -> None:
        assert set(actual_map.keys()) == set(expected_map.keys())
        for key, expected_value in expected_map.items():
            assert np.isclose(actual_map[key], expected_value, rtol=1e-10, atol=1e-12)

    _assert_close_map(actual["features"]["theta_on"], expected["features"]["theta_on"])
    _assert_close_map(actual["features"]["theta_off"], expected["features"]["theta_off"])
    _assert_close_map(actual["features"]["theta_minus"], expected["features"]["theta_minus"])
    _assert_close_map(actual["features"]["sigma_min"], expected["features"]["sigma_min"])
    _assert_close_map(actual["features"]["sigma_max"], expected["features"]["sigma_max"])
    _assert_close_map(actual["features"]["kappa_sigma"], expected["features"]["kappa_sigma"])
    _assert_close_map(actual["features"]["x0"], expected["features"]["x0"])

    assert np.isclose(
        actual["features"]["theta_rate"],
        expected["features"]["theta_rate"],
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
    _assert_close_map(actual["signals"]["tilt_scales"], expected["signals"]["tilt_scales"])
    assert np.isclose(
        actual["signals"]["tilt_eps"],
        expected["signals"]["tilt_eps"],
        rtol=1e-10,
        atol=1e-12,
    )
    assert np.isclose(
        actual["signals"]["tilt_temperature"],
        expected["signals"]["tilt_temperature"],
        rtol=1e-10,
        atol=1e-12,
    )
