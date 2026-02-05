from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from src.core.config import AppConfig, FeatureConfig, PortfolioConfig, SignalConfig
from src.services.auto_param_service import AutoParamService

FIXTURE_DIR = Path("tests/fixtures/auto_params_2023-04-28")


class FakeCalendarRepo:
    def __init__(self, dates: list[object]) -> None:
        self._dates = dates

    def get_range(self, *_args, **_kwargs) -> list[dict[str, str]]:
        return [{"date": d} for d in self._dates]


class FakeBucketRepo:
    def __init__(self, buckets: list[dict[str, str]]) -> None:
        self._buckets = buckets

    def get_range(self, *_args, **_kwargs) -> list[dict[str, str]]:
        return list(self._buckets)


class FakeMarketRepo:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self._rows = rows

    def get_range(self, *_args, **_kwargs) -> list[dict[str, str]]:
        return list(self._rows)


class FakeFactorRepo:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self._rows = rows

    def get_range(self, *_args, **_kwargs) -> list[dict[str, str]]:
        return list(self._rows)


def _load_csv(path: Path) -> list[dict[str, object]]:
    import pandas as pd

    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    return df.to_dict(orient="records")


def _load_buckets(path: Path) -> list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_config() -> AppConfig:
    return AppConfig(
        db={"dsn": "", "schema_in": "public", "schema_out": "cta"},
        features=FeatureConfig(),
        signals=SignalConfig(),
        portfolio=PortfolioConfig(),
        auto_params={"enabled": True, "auto_param_window_size": 4, "min_points": 200},
    )


def test_auto_params_2023_04_28_ground_truth(tmp_path: Path) -> None:
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

    result = service.compute_and_persist(as_of=date(2023, 4, 28))

    expected = json.loads((FIXTURE_DIR / "auto_params.json").read_text())["params"]
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
