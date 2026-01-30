# Ref: WP §11.1

from __future__ import annotations

import pandas as pd


def sample_week_last_trading_day(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if dates.empty:
        return dates
    iso = dates.isocalendar()
    df = pd.DataFrame({"date": dates, "year": iso.year, "week": iso.week})
    last = df.groupby(["year", "week"])["date"].max().sort_values()
    return pd.DatetimeIndex(last)


def rolling_window_dates(
    dates: pd.DatetimeIndex, decision_date: pd.Timestamp, window: int
) -> pd.DatetimeIndex:
    if window <= 0:
        raise ValueError("window must be positive")
    if decision_date not in dates:
        raise ValueError("decision_date must be in dates")
    prior = dates[dates < decision_date]
    if len(prior) < window:
        raise ValueError("not enough history before decision_date")
    return pd.DatetimeIndex(prior[-window:])


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
