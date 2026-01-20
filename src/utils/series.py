from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import pandas as pd


def align_series(
    series_list: Sequence[pd.Series],
    *,
    join: Literal["inner", "outer"] = "inner",
    sort: bool = True,
) -> pd.DataFrame:
    if not series_list:
        return pd.DataFrame()
    return pd.concat(series_list, axis=1, join=join, sort=sort)


def log_return(prices: pd.Series, *, dropna: bool = True) -> pd.Series:
    returns = np.log(prices / prices.shift(1))
    return returns.dropna() if dropna else returns


def build_index_from_returns(returns: pd.Series, *, base: float = 1.0) -> pd.Series:
    cumulative = returns.fillna(0).cumsum()
    return base * np.exp(cumulative)
