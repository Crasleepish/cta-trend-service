from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.series import log_return


def moving_average(series: pd.Series, window: int) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be positive")
    return series.rolling(window=window, min_periods=window).mean()


def rolling_vol(
    returns: pd.Series,
    window: int,
    *,
    annualize: int | None = None,
) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be positive")
    vol = returns.rolling(window=window, min_periods=window).std(ddof=0)
    if annualize is None:
        return vol
    return vol * np.sqrt(annualize)


def trend_score(
    price_or_nav: pd.Series,
    s: int,
    long_window: int,
    vol_window: int,
    *,
    eps: float = 1e-12,
    annualize: int | None = 252,
) -> pd.Series:
    if s <= 0 or long_window <= 0 or vol_window <= 0:
        raise ValueError("windows must be positive")
    if s >= long_window:
        raise ValueError("short window must be smaller than long window")

    ma_s = moving_average(price_or_nav, s)
    ma_l = moving_average(price_or_nav, long_window)
    returns = log_return(price_or_nav, dropna=False)
    vol = rolling_vol(returns, vol_window, annualize=annualize)

    denom = vol.replace(0.0, np.nan) + eps
    score = (ma_s - ma_l) / denom
    return score
