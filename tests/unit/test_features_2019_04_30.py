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
        "T": 0.3260451319810067,
        "gate_state": 1.0,
        "down_drift": 0.0,
        "sigma_eff": 0.22227807971921665,
        "f_sigma": 0.7004412180819205,
        "path_quality_z": 0.22185474641564593,
        "path_quality_g": 0.0,
    },
    "GROWTH": {
        "r_log_daily": 0.0050413790080600755,
        "sigma_ann": 0.22409754185391956,
        "T": 0.2907995489476288,
        "gate_state": 1.0,
        "down_drift": 0.0,
        "sigma_eff": 0.22409754185391956,
        "f_sigma": 0.8370516305791624,
        "path_quality_z": 0.05286798969775208,
        "path_quality_g": 0.0,
    },
    "CYCLE": {
        "r_log_daily": 0.0039284966825046536,
        "sigma_ann": 0.2943736908598128,
        "T": 0.3046726278421388,
        "gate_state": 1.0,
        "down_drift": 0.0,
        "sigma_eff": 0.2943736908598128,
        "f_sigma": 0.6698331381409672,
        "path_quality_z": 0.18018536066838436,
        "path_quality_g": 0.0,
    },
    "GOLD": {
        "r_log_daily": 0.0013894581568102506,
        "sigma_ann": 0.05904115100974892,
        "T": -0.1764591576317693,
        "gate_state": 0.0,
        "down_drift": 1.0,
        "sigma_eff": 0.05904115100974892,
        "f_sigma": 0.9774655971866564,
        "path_quality_z": 0.2632359760187164,
        "path_quality_g": 0.0,
    },
    "RATE": {
        "r_log_daily": 0.001108267258593012,
        "sigma_ann": 0.021880387075949193,
        "T": -0.2132850956524614,
        "gate_state": 0.0,
        "down_drift": 0.0,
        "sigma_eff": 0.021880387075949193,
        "f_sigma": 0.3969241958820217,
        "T_RATE": -0.2132850956524614,
        "rate_pref": 0.2565822702665557,
        "path_quality_z": 0.13415473993985108,
        "path_quality_g": 0.0,
    },
}

PARAMS = {
    "theta_on": {
        "VALUE": 0.26528227367646884,
        "GROWTH": 0.19379705299744035,
        "CYCLE": 0.22583391258875235,
        "GOLD": 0.2110476459202071,
        "RATE": 0.6587625573047501,
    },
    "theta_off": {
        "VALUE": 0.2082104216955758,
        "GROWTH": 0.10634305255327521,
        "CYCLE": 0.16479813230402673,
        "GOLD": 0.1544205793347547,
        "RATE": 0.5356284079404983,
    },
    "theta_minus": {
        "VALUE": 0.18685025514405096,
        "GROWTH": 0.20427491641021842,
        "CYCLE": 0.21009598816040526,
        "GOLD": 0.15647832412679033,
        "RATE": 0.43312866611889606,
    },
    "sigma_min": {
        "VALUE": 0.09757766177543571,
        "GROWTH": 0.13298117764570963,
        "CYCLE": 0.160762533143436,
        "GOLD": 0.05234831252446782,
        "RATE": 0.010265043129805444,
    },
    "sigma_max": {
        "VALUE": 0.3210343447277779,
        "GROWTH": 0.4110292497365814,
        "CYCLE": 0.3768634968865508,
        "GOLD": 0.1319654870873155,
        "RATE": 0.020037113082964714,
    },
    "kappa_sigma": {
        "VALUE": 0.1162659409595575,
        "GROWTH": 0.11422985049474679,
        "CYCLE": 0.11660482561901143,
        "GOLD": 0.019343735998570637,
        "RATE": 0.004406609264499705,
    },
    "theta_rate": 0.31861928041504267,
    "x0": 0.6179678437467533,
    "path_quality_gamma": 1.0253638828787857,
}

SHORT_WINDOW = 20
LONG_WINDOW = 60
VOL_WINDOW = 20
ANNUALIZE = 252
RATE_K = 2.0
GOLD_DATE = pd.Timestamp("2019-06-28")

GOLD_2019_06_28 = {
    "r_log_daily": 0.001718869789130015,
    "sigma_ann": 0.15942602629183647,
    "T": 0.2951171580118067,
    "path_quality_z": 0.867205121516706,
    "path_quality_g": 0.6453694613613951,
    "gate_state": 1.0,
    "sigma_eff": 0.15942602629183647,
    "f_sigma": 0.1947229077426719,
    "raw_weight": 0.2326273083269963,
}


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


def test_gold_2019_06_28_ground_truth() -> None:
    prices = _load_series()
    returns = computer.log_returns(prices)
    sigma_ann = computer.sigma_annualized(returns, window=VOL_WINDOW, annualize=ANNUALIZE)
    trend = computer.trend_strength(
        prices, sigma_ann, short_window=SHORT_WINDOW, long_window=LONG_WINDOW
    )
    path_z = computer.path_quality_z(prices, window_days=40)
    path_g = computer.path_quality_g(path_z, x0=PARAMS["x0"], gamma=PARAMS["path_quality_gamma"])

    weekly_trend = sampler.weekly_history(
        trend, calendar=prices.index, rebalance_date=GOLD_DATE.date()
    )
    weekly_sigma = sampler.weekly_history(
        sigma_ann, calendar=prices.index, rebalance_date=GOLD_DATE.date()
    )

    gate = computer.hysteresis_gate(
        weekly_trend[["GOLD"]],
        theta_on=PARAMS["theta_on"]["GOLD"],
        theta_off=PARAMS["theta_off"]["GOLD"],
    )["GOLD"]

    sigma_eff = computer.sigma_eff(
        weekly_sigma[["GOLD"]], sigma_min=PARAMS["sigma_min"]["GOLD"]
    )["GOLD"]
    f_sigma = computer.tradability_filter(
        weekly_sigma[["GOLD"]],
        sigma_max=PARAMS["sigma_max"]["GOLD"],
        kappa=PARAMS["kappa_sigma"]["GOLD"],
    )["GOLD"]

    raw_weight = (
        gate.loc[GOLD_DATE]
        * trend.loc[GOLD_DATE, "GOLD"]
        * path_g.loc[GOLD_DATE, "GOLD"]
        * f_sigma.loc[GOLD_DATE]
        * (1.0 / sigma_eff.loc[GOLD_DATE])
    )

    _assert_close(returns.loc[GOLD_DATE, "GOLD"], GOLD_2019_06_28["r_log_daily"])
    _assert_close(sigma_ann.loc[GOLD_DATE, "GOLD"], GOLD_2019_06_28["sigma_ann"])
    _assert_close(trend.loc[GOLD_DATE, "GOLD"], GOLD_2019_06_28["T"])
    _assert_close(path_z.loc[GOLD_DATE, "GOLD"], GOLD_2019_06_28["path_quality_z"])
    _assert_close(path_g.loc[GOLD_DATE, "GOLD"], GOLD_2019_06_28["path_quality_g"])
    _assert_close(gate.loc[GOLD_DATE], GOLD_2019_06_28["gate_state"])
    _assert_close(sigma_eff.loc[GOLD_DATE], GOLD_2019_06_28["sigma_eff"])
    _assert_close(f_sigma.loc[GOLD_DATE], GOLD_2019_06_28["f_sigma"])
    _assert_close(raw_weight, GOLD_2019_06_28["raw_weight"])
