from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.bucket_reco.trend.consistency import hit_ratio, trend_corr, window_slices


def test_trend_corr_identical() -> None:
    idx = pd.to_datetime([date(2026, 1, d) for d in range(1, 25)])
    series = pd.Series(np.linspace(0.1, 1.0, len(idx)), index=idx)
    rho = trend_corr(series, series, idx, min_points=5, min_coverage=0.6)
    assert rho == 1.0


def test_trend_corr_opposite() -> None:
    idx = pd.to_datetime([date(2026, 1, d) for d in range(1, 25)])
    series = pd.Series(np.linspace(0.1, 1.0, len(idx)), index=idx)
    rho = trend_corr(series, -series, idx, min_points=5, min_coverage=0.6)
    assert rho == -1.0


def test_trend_corr_noise_near_zero() -> None:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=100, freq="D")
    series_a = pd.Series(rng.normal(size=len(idx)), index=idx)
    series_b = pd.Series(rng.normal(size=len(idx)), index=idx)
    rho = trend_corr(series_a, series_b, idx, min_points=20, min_coverage=0.6)
    assert rho is not None
    assert abs(rho) < 0.2


def test_hit_ratio_all_match() -> None:
    idx = pd.to_datetime([date(2026, 1, d) for d in range(1, 31)])
    series = pd.Series(np.linspace(1.0, 2.0, len(idx)), index=idx)
    hr = hit_ratio(series, series, idx, min_points=10, min_coverage=0.6)
    assert hr == 1.0


def test_hit_ratio_all_opposite() -> None:
    idx = pd.to_datetime([date(2026, 1, d) for d in range(1, 31)])
    series = pd.Series(np.linspace(1.0, 2.0, len(idx)), index=idx)
    hr = hit_ratio(series, -series, idx, min_points=10, min_coverage=0.6)
    assert hr == 0.0


def test_hit_ratio_half_match() -> None:
    idx = pd.to_datetime([date(2026, 1, d) for d in range(1, 31)])
    series_a = pd.Series([1.0] * 15 + [-1.0] * 15, index=idx)
    series_b = pd.Series([1.0] * 15 + [1.0] * 15, index=idx)
    hr = hit_ratio(series_a, series_b, idx, min_points=10, min_coverage=0.6)
    assert hr == 0.5


def test_window_slices_months() -> None:
    dates = pd.to_datetime([date(2026, 1, 1), date(2026, 2, 1), date(2026, 3, 1), date(2026, 4, 1)])
    specs = [
        {"label": "3M", "months": 3},
        {"label": "12M", "months": 12},
    ]
    slices = window_slices(dates, specs)
    assert slices["3M"].equals(
        pd.to_datetime([date(2026, 2, 1), date(2026, 3, 1), date(2026, 4, 1)])
    )
    assert len(slices["12M"]) == 4
