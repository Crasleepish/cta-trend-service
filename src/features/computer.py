from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def sigma_annualized(
    returns: pd.DataFrame,
    *,
    window: int,
    annualize: int = 252,
) -> pd.DataFrame:
    if window <= 1:
        raise ValueError("window must be > 1")
    sigma = returns.rolling(window=window, min_periods=window).std(ddof=1)
    return sigma * math.sqrt(annualize)


def trend_strength(
    prices: pd.DataFrame,
    sigma_ann: pd.DataFrame,
    *,
    short_window: int,
    long_window: int,
) -> pd.DataFrame:
    if short_window <= 0 or long_window <= 0:
        raise ValueError("windows must be positive")
    if short_window >= long_window:
        raise ValueError("short window must be smaller than long window")
    ma_short = prices.rolling(window=short_window, min_periods=short_window).mean()
    ma_long = prices.rolling(window=long_window, min_periods=long_window).mean()
    denom = sigma_ann.replace(0.0, np.nan)
    ma_long_safe = ma_long.replace(0.0, np.nan)
    return (ma_short / ma_long_safe - 1.0) / denom


def path_quality_z(prices: pd.DataFrame, *, window_days: int) -> pd.DataFrame:
    if window_days <= 1:
        raise ValueError("window_days must be > 1")
    roll_min = prices.rolling(window=window_days, min_periods=window_days).min()
    roll_max = prices.rolling(window=window_days, min_periods=window_days).max()
    runup = np.log(prices / roll_min)
    drawdown = np.log(roll_max / prices)
    denom = runup + drawdown
    z = runup / denom
    return z.where(denom != 0.0, 0.5)


def path_quality_g(
    z: pd.DataFrame,
    *,
    x0: float | Mapping[str, float],
    gamma: float | Mapping[str, float],
) -> pd.DataFrame:
    if isinstance(x0, Mapping):
        x0_series = pd.Series(x0)
    else:
        x0_series = pd.Series({col: float(x0) for col in z.columns})
    if isinstance(gamma, Mapping):
        gamma_series = pd.Series(gamma)
    else:
        gamma_series = pd.Series({col: float(gamma) for col in z.columns})

    out = pd.DataFrame(index=z.index, columns=z.columns, dtype=float)
    for col in z.columns:
        x0_val = float(x0_series.get(col, 0.0))
        gamma_val = float(gamma_series.get(col, 1.0))
        if gamma_val <= 0:
            raise ValueError("gamma must be positive")
        if x0_val >= 1.0:
            out[col] = 0.0
            continue
        ratio = (z[col] - x0_val) / (1.0 - x0_val)
        out[col] = np.where(z[col] <= x0_val, 0.0, ratio.clip(lower=0.0) ** gamma_val)
    return out


def hysteresis_gate(
    trend_weekly: pd.DataFrame,
    *,
    theta_on: float,
    theta_off: float,
) -> pd.DataFrame:
    states = pd.DataFrame(index=trend_weekly.index, columns=trend_weekly.columns)
    for col in trend_weekly.columns:
        state = 0
        values = []
        for t in trend_weekly[col].to_list():
            if pd.isna(t):
                values.append(np.nan)
                continue
            if t > theta_on:
                state = 1
            elif t < theta_off:
                state = 0
            values.append(state)
        states[col] = values
    return states.astype(float)


def down_drift(trend_weekly: pd.DataFrame, *, theta_minus: float) -> pd.DataFrame:
    return (trend_weekly < -theta_minus).astype(float)


def sigma_eff(sigma_weekly: pd.DataFrame, *, sigma_min: float) -> pd.DataFrame:
    return sigma_weekly.clip(lower=sigma_min)


def tradability_filter(
    sigma_weekly: pd.DataFrame, *, sigma_max: float, kappa: float
) -> pd.DataFrame:
    return 1.0 / (1.0 + np.exp((sigma_weekly - sigma_max) / kappa))


def rate_preference(trend_rate: pd.Series, *, k: float, theta_rate: float) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-k * (trend_rate - theta_rate)))


def build_feature_frame(
    feature_name: str,
    data: pd.DataFrame,
    *,
    instrument_id_label: str = "instrument_id",
    value_label: str = "value",
) -> pd.DataFrame:
    stacked = data.stack(dropna=True).rename(value_label).reset_index()
    if "level_0" in stacked.columns:
        stacked = stacked.rename(columns={"level_0": "calc_date"})
    if "date" in stacked.columns and "calc_date" not in stacked.columns:
        stacked = stacked.rename(columns={"date": "calc_date"})
    if "level_1" in stacked.columns:
        stacked = stacked.rename(columns={"level_1": instrument_id_label})
    if instrument_id_label not in stacked.columns and "instrument_id" in stacked.columns:
        stacked = stacked.rename(columns={"instrument_id": instrument_id_label})
    stacked["feature_name"] = feature_name
    return stacked


def feature_params_hash(params: Mapping[str, object]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
