from __future__ import annotations

from datetime import date

import pandas as pd

from ..utils.trading_calendar import sample_week_last_trading_day


def weekly_history(
    series: pd.DataFrame, *, calendar: pd.DatetimeIndex, rebalance_date: date
) -> pd.DataFrame:
    if series.empty:
        raise ValueError("series is empty")
    weekly_dates = sample_week_last_trading_day(calendar)
    target = pd.Timestamp(rebalance_date)
    if target not in weekly_dates:
        raise ValueError("rebalance_date must be a weekly sampling date")
    return series.loc[weekly_dates[weekly_dates <= target]]


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
