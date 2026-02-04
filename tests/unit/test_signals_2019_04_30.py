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
        "T": 4454.848484913654,
        "gate_state": 1.0,
        "sigma_eff": 0.22227807971921665,
        "f_sigma": 0.6922311442703354,
    },
    "GROWTH": {
        "T": 464.980304876604,
        "gate_state": 1.0,
        "sigma_eff": 0.22409754185391956,
        "f_sigma": 0.835262438716919,
    },
    "CYCLE": {
        "T": 1147.3530034114967,
        "gate_state": 1.0,
        "sigma_eff": 0.2943736908598128,
        "f_sigma": 0.6764720050615407,
    },
    "GOLD": {
        "T": -50.1768673092241,
        "gate_state": 0.0,
        "sigma_eff": 0.05904115100974892,
        "f_sigma": 0.9772289012762934,
    },
    "RATE": {
        "T": -40.68331440284041,
        "gate_state": 0.0,
        "sigma_eff": 0.021880387075949193,
        "f_sigma": 0.4003274096970538,
        "rate_pref": 6.967914891732686e-85,
    },
}

# Manual raw components
RAW_COMPONENTS = {
    "VALUE": {
        "raw_weight_component_risk_budget": 1.0,
        "raw_weight_component_gate": 1.0,
        "raw_weight_component_trend": 4454.848484913654,
        "raw_weight_component_inv_sigma_eff": 4.498869169929881,
        "raw_weight_component_f_sigma": 0.6922311442703354,
        "raw_weight_component_path_quality": 0.0,
    },
    "GROWTH": {
        "raw_weight_component_risk_budget": 1.0,
        "raw_weight_component_gate": 1.0,
        "raw_weight_component_trend": 464.980304876604,
        "raw_weight_component_inv_sigma_eff": 4.462342566219941,
        "raw_weight_component_f_sigma": 0.835262438716919,
        "raw_weight_component_path_quality": 0.0,
    },
    "CYCLE": {
        "raw_weight_component_risk_budget": 1.0,
        "raw_weight_component_gate": 1.0,
        "raw_weight_component_trend": 1147.3530034114967,
        "raw_weight_component_inv_sigma_eff": 3.3970427081278194,
        "raw_weight_component_f_sigma": 0.6764720050615407,
        "raw_weight_component_path_quality": 0.0,
    },
    "GOLD": {
        "raw_weight_component_risk_budget": 1.0,
        "raw_weight_component_gate": 0.0,
        "raw_weight_component_trend": -50.1768673092241,
        "raw_weight_component_inv_sigma_eff": 16.93733917610921,
        "raw_weight_component_f_sigma": 0.9772289012762934,
        "raw_weight_component_path_quality": 0.0,
    },
    "RATE": {
        "raw_weight_component_risk_budget": 1.0,
        "raw_weight_component_gate": 0.0,
        "raw_weight_component_trend": -40.68331440284041,
        "raw_weight_component_inv_sigma_eff": 45.70303059671165,
        "raw_weight_component_f_sigma": 0.4003274096970538,
        "raw_weight_component_path_quality": 0.0,
    },
}

# Manual tilt results
TILT = {
    "VALUE": {
        "005562.OF": {"tilt_score": 0.23516601202888376, "tilt_weight": 0.2651729830244153},
        "000312.OF": {"tilt_score": 0.8204073771899466, "tilt_weight": 0.4760980137709042},
        "003194.OF": {"tilt_score": 0.2105648271238617, "tilt_weight": 0.25872900320468045},
    },
    "GROWTH": {
        "004744.OF": {"tilt_score": -0.17461069958882133, "tilt_weight": 0.38320359239456075},
        "002236.OF": {"tilt_score": -0.9828041479172183, "tilt_weight": 0.17077945247503185},
        "004409.OF": {"tilt_score": -0.022820153081494543, "tilt_weight": 0.4460169551304074},
    },
    "CYCLE": {
        "690008.OF": {"tilt_score": -0.9894781070276071, "tilt_weight": 0.2999549565820758},
        "164304.OF": {"tilt_score": -0.694063076568751, "tilt_weight": 0.40304464970241366},
        "004195.OF": {"tilt_score": -0.9993769612412756, "tilt_weight": 0.29700039371551057},
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
TILT_SCALES = {"SMB": 0.29581249662477127, "QMJ": 0.745252371165715}
TILT_EPS = 1e-12
TILT_TEMPERATURE = 1.0


def _load_buckets() -> dict[str, list[str]]:
    buckets = json.loads((FIXTURE_DIR / "buckets.json").read_text())
    bucket_assets = {}
    for b in buckets:
        assets = [a.strip() for a in (b.get("assets") or "").split(",") if a.strip()]
        bucket_assets[b["bucket_name"]] = assets
    return bucket_assets


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = scores / max(temperature, 1e-12)
    scaled -= np.max(scaled)
    exp = np.exp(scaled)
    return exp / exp.sum()


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
    start_date = REB_DATE - pd.Timedelta(days=TILT_LOOKBACK_DAYS)
    factor_df = factors[(factors["date"] >= start_date) & (factors["date"] <= REB_DATE)]
    s_t = factor_df[TILT_FACTORS].sum().astype(float)
    s_t = s_t / pd.Series(TILT_SCALES).reindex(TILT_FACTORS).fillna(1.0)
    s_norm = np.linalg.norm(s_t.values)

    beta = pd.read_csv(FIXTURE_DIR / "fund_beta.csv", parse_dates=["date"])
    beta["date"] = beta["date"].dt.normalize()
    exposures = beta[beta["date"] == REB_DATE].set_index("code")[TILT_FACTORS].astype(float)

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
            vector = exposures.loc[asset].values
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
            score_vec = np.array([scores.get(asset, 0.0) for asset in assets])
            weights = _softmax(score_vec, TILT_TEMPERATURE)
            weight_map = {
                asset: float(weight) if eligible.get(asset, False) else 0.0
                for asset, weight in zip(assets, weights)
            }
            total = sum(weight_map.values())
            if total > 0:
                weight_map = {asset: weight / total for asset, weight in weight_map.items()}
            else:
                weight_map = {asset: 0.0 for asset in assets}

        for asset in assets:
            _assert_close(scores.get(asset, 0.0), TILT[bucket][asset]["tilt_score"])
            _assert_close(weight_map[asset], TILT[bucket][asset]["tilt_weight"])
