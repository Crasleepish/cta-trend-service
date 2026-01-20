from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.bucket_reco.trend.trend_score import moving_average, rolling_vol, trend_score


def test_trend_score_constant_series_zero() -> None:
    idx = pd.to_datetime([date(2026, 1, d) for d in range(1, 8)])
    prices = pd.Series([1.0] * len(idx), index=idx)

    score = trend_score(prices, s=2, long_window=3, vol_window=3, eps=1e-8)
    tail = score.dropna()
    assert np.allclose(tail.values, 0.0)


def test_trend_score_increasing_positive() -> None:
    idx = pd.to_datetime([date(2026, 1, d) for d in range(1, 10)])
    prices = pd.Series(np.linspace(1.0, 1.9, len(idx)), index=idx)

    score = trend_score(prices, s=2, long_window=5, vol_window=3)
    tail = score.dropna()
    assert tail.iloc[-1] > 0


def test_trend_score_decreasing_negative() -> None:
    idx = pd.to_datetime([date(2026, 1, d) for d in range(1, 10)])
    prices = pd.Series(np.linspace(1.9, 1.0, len(idx)), index=idx)

    score = trend_score(prices, s=2, long_window=5, vol_window=3)
    tail = score.dropna()
    assert tail.iloc[-1] < 0


def test_trend_score_eps_no_inf_nan() -> None:
    idx = pd.to_datetime([date(2026, 1, d) for d in range(1, 8)])
    prices = pd.Series([1.0] * len(idx), index=idx)

    score = trend_score(prices, s=2, long_window=3, vol_window=3, eps=1e-8)
    assert not np.isinf(score.dropna()).any()
    assert not np.isnan(score.dropna()).any()


def test_trend_score_warmup_kept_nan() -> None:
    idx = pd.to_datetime([date(2026, 1, d) for d in range(1, 10)])
    prices = pd.Series(np.linspace(1.0, 1.9, len(idx)), index=idx)

    score = trend_score(prices, s=3, long_window=5, vol_window=4)
    warmup = max(5, 4) - 1
    assert score.isna().iloc[:warmup].all()
    assert score.notna().iloc[warmup:].any()


def test_moving_average_window_behavior() -> None:
    series = pd.Series([1.0, 2.0, 3.0])
    ma = moving_average(series, window=2)
    assert np.isnan(ma.iloc[0])
    assert np.isclose(ma.iloc[1], 1.5)


def test_rolling_vol_window_behavior() -> None:
    returns = pd.Series([0.0, 0.1, -0.1, 0.2])
    vol = rolling_vol(returns, window=3, annualize=None)
    assert np.isnan(vol.iloc[1])
    assert not np.isnan(vol.iloc[2])
