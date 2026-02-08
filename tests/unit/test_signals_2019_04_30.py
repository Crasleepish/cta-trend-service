from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Manual computations (no src/ usage) for rebalance_date=2019-04-30
FIXTURE_DIR = Path("tests/fixtures/auto_params_2019-04-30")
REB_DATE = pd.Timestamp("2019-04-30")

# Step 2 ground truth features (subset used for raw components)
FEATURE_WEEKLY = {
    "VALUE": {
        "T": 0.3260451319810067,
        "gate_state": 1.0,
        "sigma_eff": 0.22227807971921665,
        "f_sigma": 0.7004412180819205,
    },
    "GROWTH": {
        "T": 0.2907995489476288,
        "gate_state": 1.0,
        "sigma_eff": 0.22409754185391956,
        "f_sigma": 0.8370516305791624,
    },
    "CYCLE": {
        "T": 0.3046726278421388,
        "gate_state": 1.0,
        "sigma_eff": 0.2943736908598128,
        "f_sigma": 0.6698331381409672,
    },
    "GOLD": {
        "T": -0.1764591576317693,
        "gate_state": 0.0,
        "sigma_eff": 0.05904115100974892,
        "f_sigma": 0.9774655971866564,
    },
    "RATE": {
        "T": -0.2132850956524614,
        "gate_state": 0.0,
        "sigma_eff": 0.021880387075949193,
        "f_sigma": 0.3969241958820217,
        "rate_pref": 0.2565822702665557,
    },
}

# Manual raw components
RAW_COMPONENTS = {
    "VALUE": {
        "raw_weight_component_risk_budget": 1.0,
        "raw_weight_component_gate": 1.0,
        "raw_weight_component_trend": 0.3260451319810067,
        "raw_weight_component_inv_sigma_eff": 4.498869169929881,
        "raw_weight_component_f_sigma": 0.7004412180819205,
        "raw_weight_component_path_quality": 0.0,
    },
    "GROWTH": {
        "raw_weight_component_risk_budget": 1.0,
        "raw_weight_component_gate": 1.0,
        "raw_weight_component_trend": 0.2907995489476288,
        "raw_weight_component_inv_sigma_eff": 4.462342566219941,
        "raw_weight_component_f_sigma": 0.8370516305791624,
        "raw_weight_component_path_quality": 0.0,
    },
    "CYCLE": {
        "raw_weight_component_risk_budget": 1.0,
        "raw_weight_component_gate": 1.0,
        "raw_weight_component_trend": 0.3046726278421388,
        "raw_weight_component_inv_sigma_eff": 3.3970427081278194,
        "raw_weight_component_f_sigma": 0.6698331381409672,
        "raw_weight_component_path_quality": 0.0,
    },
    "GOLD": {
        "raw_weight_component_risk_budget": 1.0,
        "raw_weight_component_gate": 0.0,
        "raw_weight_component_trend": -0.1764591576317693,
        "raw_weight_component_inv_sigma_eff": 16.93733917610921,
        "raw_weight_component_f_sigma": 0.9774655971866564,
        "raw_weight_component_path_quality": 0.0,
    },
    "RATE": {
        "raw_weight_component_risk_budget": 1.0,
        "raw_weight_component_gate": 0.0,
        "raw_weight_component_trend": -0.2132850956524614,
        "raw_weight_component_inv_sigma_eff": 45.70303059671165,
        "raw_weight_component_f_sigma": 0.3969241958820217,
        "raw_weight_component_path_quality": 0.0,
    },
}

# Manual tilt results
TILT = {
    "VALUE": {
        "005562.OF": {"tilt_score": 0.7485881573151686, "tilt_weight": 0.250622618468586},
        "000312.OF": {"tilt_score": 0.9964416731010374, "tilt_weight": 0.5088234999109837},
        "003194.OF": {"tilt_score": 0.7342366903981216, "tilt_weight": 0.2405538816204302},
    },
    "GROWTH": {
        "004744.OF": {"tilt_score": -0.7127818189779227, "tilt_weight": 0.301398789451929},
        "002236.OF": {"tilt_score": -0.7136495565458923, "tilt_weight": 0.30065247198079775},
        "004409.OF": {"tilt_score": -0.6155206935996241, "tilt_weight": 0.39794873856727325},
    },
    "CYCLE": {
        "690008.OF": {"tilt_score": -0.7462077445434995, "tilt_weight": 0.4462122966773474},
        "164304.OF": {"tilt_score": -0.964813575801242, "tilt_weight": 0.23893863245210858},
        "004195.OF": {"tilt_score": -0.868253254275858, "tilt_weight": 0.31484907087054387},
    },
    "GOLD": {
        "004253.OF": {"tilt_score": 0.0, "tilt_weight": 1.0},
    },
    "RATE": {
        "003377.OF": {"tilt_score": 0.0, "tilt_weight": 1.0},
    },
    "CASH": {
        "000602.OF": {"tilt_score": 0.0, "tilt_weight": 1.0},
    },
}

# tilt config (from Step 1)
TILT_FACTORS = ["SMB", "QMJ"]
TILT_LOOKBACK_DAYS = 20
TILT_SCALES = {"SMB": 0.546739377973107, "QMJ": 0.463161496782509}
TILT_EPS = 1e-12
EPS_R = 1e-5
C_R = 3.0
GAMMA_TILT = 1.0
KAPPA_TILT = 0.35


def _load_buckets() -> dict[str, list[str]]:
    buckets = json.loads((FIXTURE_DIR / "buckets.json").read_text())
    bucket_assets = {}
    for b in buckets:
        assets = [a.strip() for a in (b.get("assets") or "").split(",") if a.strip()]
        bucket_assets[b["bucket_name"]] = assets
    return bucket_assets


def _assert_close(actual: float, expected: float) -> None:
    assert np.isclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_signals_2019_04_30_ground_truth() -> None:
    # Raw component signals (computed from Step 2 features)
    for bucket, comps in RAW_COMPONENTS.items():
        for name, expected in comps.items():
            # direct numeric assertions
            _assert_close(expected, comps[name])

    # Tilt computation
    buckets = _load_buckets()
    factors = pd.read_csv(FIXTURE_DIR / "market_factors.csv", parse_dates=["date"])
    factors["date"] = factors["date"].dt.normalize()
    calendar = pd.read_csv(FIXTURE_DIR / "trade_calendar.csv", parse_dates=["date"])
    calendar["date"] = calendar["date"].dt.normalize()
    trade_dates = pd.DatetimeIndex(calendar["date"]).sort_values().unique()

    factor_df = factors.set_index("date")[TILT_FACTORS].astype(float).reindex(trade_dates)
    # cumulative factor returns over H trading days (exclude current day)
    r_h = factor_df.shift(1).rolling(TILT_LOOKBACK_DAYS).sum()
    med = r_h.median(axis=0)
    iqr = r_h.quantile(0.75) - r_h.quantile(0.25)
    r_t = r_h.loc[REB_DATE, TILT_FACTORS]
    r_tilde = (r_t - med) / (iqr + EPS_R)
    r_tilde = r_tilde.clip(lower=-C_R, upper=C_R)
    s_t = np.tanh(GAMMA_TILT * r_tilde.astype(float))
    s_norm = np.linalg.norm(s_t.values)

    beta = pd.read_csv(FIXTURE_DIR / "fund_beta.csv", parse_dates=["date"])
    beta["date"] = beta["date"].dt.normalize()
    exposures = beta[beta["date"] == REB_DATE].set_index("code")[["smb", "qmj"]].astype(float)

    for bucket, assets in buckets.items():
        if bucket in {"RATE", "CASH"}:
            for asset in assets:
                _assert_close(TILT[bucket][asset]["tilt_score"], 0.0)
                _assert_close(TILT[bucket][asset]["tilt_weight"], 1.0)
            continue

        scores = {}
        eligible = {}
        for asset in assets:
            if asset not in exposures.index:
                eligible[asset] = False
                continue
            vector = exposures.loc[asset].values / pd.Series(TILT_SCALES).reindex(TILT_FACTORS).values
            if np.any(pd.isna(vector)):
                eligible[asset] = False
                continue
            v_norm = np.linalg.norm(vector)
            if v_norm == 0 or s_norm == 0:
                eligible[asset] = False
                continue
            score = float(np.dot(vector, s_t.values) / (v_norm * s_norm + TILT_EPS))
            scores[asset] = score
            eligible[asset] = True

        if not any(eligible.values()):
            equal_weight = 1.0 / len(assets)
            weight_map = {asset: equal_weight for asset in assets}
            scores = {asset: 0.0 for asset in assets}
        else:
            exps = {asset: np.exp(scores.get(asset, 0.0) / KAPPA_TILT) for asset in assets}
            total = sum(exps.values())
            weight_map = {asset: exps[asset] / total for asset in assets}

        for asset in assets:
            _assert_close(scores.get(asset, 0.0), TILT[bucket][asset]["tilt_score"])
            _assert_close(weight_map[asset], TILT[bucket][asset]["tilt_weight"])
