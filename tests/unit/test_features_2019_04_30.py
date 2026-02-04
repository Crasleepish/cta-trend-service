from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import computer, sampler

FIXTURE_DIR = Path("tests/fixtures/auto_params_2019-04-30")

REB_DATE = pd.Timestamp("2019-04-30")

# Manual ground truth (computed via numerical-calc against DB fixtures, per WP rules)
GROUND_TRUTH = {
    "VALUE": {
        "r_log_daily": 0.009073427906944577,
        "sigma_ann": 0.22227807971921665,
        "T": 4454.848484913654,
        "gate_state": 1.0,
        "down_drift": 0.0,
        "sigma_eff": 0.22227807971921665,
        "f_sigma": 0.6922311442703354,
        "path_quality_z": 0.22185474641564593,
        "path_quality_g": 0.0,
    },
    "GROWTH": {
        "r_log_daily": 0.0050413790080600755,
        "sigma_ann": 0.22409754185391956,
        "T": 464.980304876604,
        "gate_state": 1.0,
        "down_drift": 0.0,
        "sigma_eff": 0.22409754185391956,
        "f_sigma": 0.835262438716919,
        "path_quality_z": 0.05286798969775208,
        "path_quality_g": 0.0,
    },
    "CYCLE": {
        "r_log_daily": 0.0039284966825046536,
        "sigma_ann": 0.2943736908598128,
        "T": 1147.3530034114967,
        "gate_state": 1.0,
        "down_drift": 0.0,
        "sigma_eff": 0.2943736908598128,
        "f_sigma": 0.6764720050615407,
        "path_quality_z": 0.18018536066838436,
        "path_quality_g": 0.0,
    },
    "GOLD": {
        "r_log_daily": 0.0013894581568102506,
        "sigma_ann": 0.05904115100974892,
        "T": -50.1768673092241,
        "gate_state": 0.0,
        "down_drift": 1.0,
        "sigma_eff": 0.05904115100974892,
        "f_sigma": 0.9772289012762934,
        "path_quality_z": 0.2632359760187164,
        "path_quality_g": 0.0,
    },
    "RATE": {
        "r_log_daily": 0.001108267258593012,
        "sigma_ann": 0.021880387075949193,
        "T": -40.68331440284041,
        "gate_state": 0.0,
        "down_drift": 0.0,
        "sigma_eff": 0.021880387075949193,
        "f_sigma": 0.4003274096970538,
        "T_RATE": -40.68331440284041,
        "rate_pref": 6.967914891732686e-85,
        "path_quality_z": 0.13415473993985108,
        "path_quality_g": 0.0,
    },
}

PARAMS = {
    "theta_on": {
        "VALUE": 3564.036467680621,
        "GROWTH": 430.61704391201624,
        "CYCLE": 945.1856827759191,
        "GOLD": 57.48896791718857,
        "RATE": 117.17354340819385,
    },
    "theta_off": {
        "VALUE": 1763.8045945109038,
        "GROWTH": 104.05952138758168,
        "CYCLE": 345.73638452763515,
        "GOLD": 27.116638849039436,
        "RATE": 64.58427673056205,
    },
    "theta_minus": {
        "VALUE": 2423.5316208757686,
        "GROWTH": 384.9381371041566,
        "CYCLE": 879.8022441596108,
        "GOLD": 39.055242453537794,
        "RATE": 78.23722535348767,
    },
    "sigma_min": {
        "VALUE": 0.09722297364904037,
        "GROWTH": 0.1314713607995856,
        "CYCLE": 0.1606855582683171,
        "GOLD": 0.05289235090804694,
        "RATE": 0.009839687772979198,
    },
    "sigma_max": {
        "VALUE": 0.32010346621631464,
        "GROWTH": 0.41000721009032004,
        "CYCLE": 0.37757622096055643,
        "GOLD": 0.13070952598946672,
        "RATE": 0.02019030264056841,
    },
    "kappa_sigma": {
        "VALUE": 0.12068702150468007,
        "GROWTH": 0.11451924950711895,
        "CYCLE": 0.11280086642415488,
        "GOLD": 0.019064648396289765,
        "RATE": 0.004182330832910001,
    },
    "theta_rate": 56.20589403674875,
    "x0": {
        "VALUE": 0.7586783273557413,
        "GROWTH": 0.5809523771126269,
        "CYCLE": 0.5908500645527218,
        "GOLD": 0.7067768294384347,
        "RATE": 0.9615507370621633,
    },
    "path_quality_gamma": {
        "VALUE": 2.0,
        "GROWTH": 2.0,
        "CYCLE": 2.0,
        "GOLD": 2.0,
        "RATE": 2.0,
    },
}

SHORT_WINDOW = 20
LONG_WINDOW = 60
VOL_WINDOW = 20
ANNUALIZE = 252
RATE_K = 2.0


def _load_buckets() -> dict[str, str]:
    buckets = json.loads((FIXTURE_DIR / "buckets.json").read_text())
    return {b["bucket_name"]: b.get("bucket_proxy") for b in buckets}


def _load_series() -> pd.DataFrame:
    dates = pd.read_csv(FIXTURE_DIR / "trade_calendar.csv", parse_dates=["date"])
    dates["date"] = dates["date"].dt.normalize()
    idx = pd.DatetimeIndex(dates["date"]).sort_values().unique()

    index_hist = pd.read_csv(FIXTURE_DIR / "index_hist.csv", parse_dates=["date"])
    index_hist["date"] = index_hist["date"].dt.normalize()

    proxies = _load_buckets()
    frames = []
    for bucket, proxy in proxies.items():
        if not proxy:
            continue
        data = index_hist[index_hist["index_code"] == proxy].copy()
        series = data.set_index("date")["close"].sort_index().reindex(idx)
        frames.append(series.rename(bucket))
    return pd.concat(frames, axis=1)


def _assert_close(actual: float, expected: float) -> None:
    assert np.isclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_features_2019_04_30_ground_truth() -> None:
    prices = _load_series()
    returns = computer.log_returns(prices)
    sigma_ann = computer.sigma_annualized(returns, window=VOL_WINDOW, annualize=ANNUALIZE)
    trend = computer.trend_strength(
        prices, sigma_ann, short_window=SHORT_WINDOW, long_window=LONG_WINDOW
    )
    path_z = computer.path_quality_z(prices, window_days=40)
    path_g = computer.path_quality_g(path_z, x0=PARAMS["x0"], gamma=PARAMS["path_quality_gamma"])

    weekly_trend = sampler.weekly_history(
        trend, calendar=prices.index, rebalance_date=REB_DATE.date()
    )
    weekly_sigma = sampler.weekly_history(
        sigma_ann, calendar=prices.index, rebalance_date=REB_DATE.date()
    )

    gate = pd.DataFrame(index=weekly_trend.index, columns=weekly_trend.columns)
    for bucket in weekly_trend.columns:
        gate[bucket] = computer.hysteresis_gate(
            weekly_trend[[bucket]],
            theta_on=PARAMS["theta_on"][bucket],
            theta_off=PARAMS["theta_off"][bucket],
        )[bucket]

    down_drift = pd.DataFrame(index=weekly_trend.index, columns=weekly_trend.columns)
    for bucket in weekly_trend.columns:
        down_drift[bucket] = computer.down_drift(
            weekly_trend[[bucket]],
            theta_minus=PARAMS["theta_minus"][bucket],
        )[bucket]

    sigma_eff = pd.DataFrame(index=weekly_sigma.index, columns=weekly_sigma.columns)
    f_sigma = pd.DataFrame(index=weekly_sigma.index, columns=weekly_sigma.columns)
    for bucket in weekly_sigma.columns:
        sigma_eff[bucket] = computer.sigma_eff(
            weekly_sigma[[bucket]], sigma_min=PARAMS["sigma_min"][bucket]
        )[bucket]
        f_sigma[bucket] = computer.tradability_filter(
            weekly_sigma[[bucket]],
            sigma_max=PARAMS["sigma_max"][bucket],
            kappa=PARAMS["kappa_sigma"][bucket],
        )[bucket]

    for bucket, expected in GROUND_TRUTH.items():
        _assert_close(returns.loc[REB_DATE, bucket], expected["r_log_daily"])
        _assert_close(sigma_ann.loc[REB_DATE, bucket], expected["sigma_ann"])
        _assert_close(trend.loc[REB_DATE, bucket], expected["T"])
        _assert_close(path_z.loc[REB_DATE, bucket], expected["path_quality_z"])
        _assert_close(path_g.loc[REB_DATE, bucket], expected["path_quality_g"])
        _assert_close(gate.loc[REB_DATE, bucket], expected["gate_state"])
        _assert_close(down_drift.loc[REB_DATE, bucket], expected["down_drift"])
        _assert_close(sigma_eff.loc[REB_DATE, bucket], expected["sigma_eff"])
        _assert_close(f_sigma.loc[REB_DATE, bucket], expected["f_sigma"])

    # RATE-specific
    rate_t = weekly_trend.loc[REB_DATE, "RATE"]
    rate_series = pd.Series([rate_t], index=[REB_DATE], name="RATE")
    rate_pref = computer.rate_preference(rate_series, k=RATE_K, theta_rate=PARAMS["theta_rate"])
    _assert_close(rate_t, GROUND_TRUTH["RATE"]["T_RATE"])
    _assert_close(rate_pref.loc[REB_DATE], GROUND_TRUTH["RATE"]["rate_pref"])
