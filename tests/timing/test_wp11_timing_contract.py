# Ref: WP §11.1

from __future__ import annotations

import pandas as pd

from src.features.sampler import rolling_window_dates, sample_week_last_trading_day


def test_wp11_sample_week_last_trading_day_missing_friday() -> None:
    dates = pd.to_datetime(
        [
            "2025-12-01",
            "2025-12-02",
            "2025-12-03",
            "2025-12-04",
            "2025-12-08",
            "2025-12-09",
            "2025-12-10",
            "2025-12-11",
        ]
    )
    sampled = sample_week_last_trading_day(pd.DatetimeIndex(dates))
    assert sampled.to_list() == [pd.Timestamp("2025-12-04"), pd.Timestamp("2025-12-11")]


def test_wp11_sample_week_last_trading_day_regular_week() -> None:
    dates = pd.to_datetime(
        [
            "2025-12-15",
            "2025-12-16",
            "2025-12-17",
            "2025-12-18",
            "2025-12-19",
        ]
    )
    sampled = sample_week_last_trading_day(pd.DatetimeIndex(dates))
    assert sampled.to_list() == [pd.Timestamp("2025-12-19")]


def test_wp11_rolling_window_excludes_decision_date() -> None:
    dates = pd.to_datetime(
        [
            "2025-12-01",
            "2025-12-02",
            "2025-12-03",
            "2025-12-04",
            "2025-12-05",
        ]
    )
    window = rolling_window_dates(pd.DatetimeIndex(dates), pd.Timestamp("2025-12-05"), 3)
    assert window.to_list() == [
        pd.Timestamp("2025-12-02"),
        pd.Timestamp("2025-12-03"),
        pd.Timestamp("2025-12-04"),
    ]
