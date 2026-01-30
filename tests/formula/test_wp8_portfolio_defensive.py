# Ref: WP §8.1, §8.4, §8.5

from __future__ import annotations

import math
from typing import Mapping

import pytest


def normalize_risk_weights(raw: Mapping[str, float]) -> dict[str, float]:
    total = sum(raw.values())
    if total <= 0:
        return {k: 0.0 for k in raw}
    return {k: v / total for k, v in raw.items()}


def defensive_budget(risk_weights: Mapping[str, float]) -> float:
    return 1.0 - sum(risk_weights.values())


def rate_preference(trend_rate: float, k: float, theta_rate: float) -> float:
    return 1.0 / (1.0 + math.exp(-k * (trend_rate - theta_rate)))


def split_defensive(def_budget: float, pref_rate: float) -> dict[str, float]:
    w_rate = def_budget * pref_rate
    w_cash = def_budget - w_rate
    return {"RATE": w_rate, "CASH": w_cash}


def implied_risk_vol(weights: Mapping[str, float], cov: dict[tuple[str, str], float]) -> float:
    keys = list(weights.keys())
    total = 0.0
    for i in keys:
        for j in keys:
            total += weights[i] * weights[j] * cov[(i, j)]
    return math.sqrt(total)


def test_wp8_weights_sum_to_one_nonnegativity() -> None:
    raw = {"EQUITY": 0.2, "GOLD": 0.3}
    risk = normalize_risk_weights(raw)
    w_def = defensive_budget(risk)
    defensive = split_defensive(w_def, pref_rate=0.4)
    total = sum(risk.values()) + sum(defensive.values())
    assert total == pytest.approx(1.0, rel=1e-12)
    assert all(value >= 0 for value in list(risk.values()) + list(defensive.values()))


def test_wp8_defensive_allocation_trigger_all_zero_risk() -> None:
    raw = {"EQUITY": 0.0, "GOLD": 0.0}
    risk = normalize_risk_weights(raw)
    assert sum(risk.values()) == 0.0
    w_def = defensive_budget(risk)
    assert w_def == pytest.approx(1.0, rel=1e-12)


def test_wp8_rate_cash_split_monotone() -> None:
    w_def = 0.4
    pref_high = rate_preference(trend_rate=2.0, k=2.0, theta_rate=0.0)
    pref_low = rate_preference(trend_rate=-2.0, k=2.0, theta_rate=0.0)
    high = split_defensive(w_def, pref_high)
    low = split_defensive(w_def, pref_low)
    assert high["RATE"] > high["CASH"]
    assert low["RATE"] < low["CASH"]


def test_wp8_implied_risk_volatility_basic() -> None:
    weights = {"EQUITY": 0.6, "GOLD": 0.4}
    cov = {
        ("EQUITY", "EQUITY"): 0.09,
        ("GOLD", "GOLD"): 0.04,
        ("EQUITY", "GOLD"): 0.01,
        ("GOLD", "EQUITY"): 0.01,
    }
    vol = implied_risk_vol(weights, cov)
    # sqrt(0.6^2*0.09 + 2*0.6*0.4*0.01 + 0.4^2*0.04) = sqrt(0.0436)
    assert vol == pytest.approx(math.sqrt(0.0436), rel=1e-12)
