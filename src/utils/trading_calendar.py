from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, cast

import pandas as pd


@dataclass(frozen=True)
class TradingCalendar:
    dates: pd.DatetimeIndex

    @classmethod
    def from_dates(cls, dates: Iterable[date]) -> "TradingCalendar":
        idx = pd.DatetimeIndex(pd.to_datetime(list(dates))).sort_values().unique()
        return cls(idx)

    def between(self, start: date, end: date) -> pd.DatetimeIndex:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        return self.dates[(self.dates >= start_ts) & (self.dates <= end_ts)]

    def shift(self, anchor: date, offset: int) -> date:
        if offset == 0:
            if pd.Timestamp(anchor) not in self.dates:
                raise ValueError("anchor is not a trading day")
            return anchor
        if offset > 0:
            subset = self.dates[self.dates > pd.Timestamp(anchor)]
            if len(subset) < offset:
                raise ValueError("not enough future trading days")
            return cast(date, pd.Timestamp(subset[offset - 1]).date())
        subset = self.dates[self.dates < pd.Timestamp(anchor)]
        if len(subset) < abs(offset):
            raise ValueError("not enough past trading days")
        return cast(date, pd.Timestamp(subset[offset]).date())


def sample_week_last_trading_day(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if dates.empty:
        return dates
    iso = dates.isocalendar()
    df = pd.DataFrame({"date": dates, "year": iso.year, "week": iso.week})
    last = df.groupby(["year", "week"])["date"].max().sort_values()
    return pd.DatetimeIndex(last)
