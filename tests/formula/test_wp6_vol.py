# Ref: WP §6.1, §6.2, §6.3

from __future__ import annotations

import math
from typing import Iterable

import pytest


def sigma_annualized(returns: Iterable[float], window: int, annualize: int = 252) -> float:
    values = list(returns)
    if window <= 1:
        raise ValueError("window must be > 1")
    if len(values) < window:
        raise ValueError("not enough data")
    window_vals = values[-window:]
    mean = sum(window_vals) / window
    var = sum((x - mean) ** 2 for x in window_vals) / (window - 1)
    return math.sqrt(var) * math.sqrt(annualize)


def sigma_eff(sigma: float, sigma_min: float) -> float:
    return max(sigma, sigma_min)


def tradability_filter(sigma: float, sigma_max: float, kappa: float) -> float:
    return 1.0 / (1.0 + math.exp((sigma - sigma_max) / kappa))


def test_wp6_sigma_annualization_known_series() -> None:
    returns = [0.01, -0.01, 0.02, -0.02]
    sigma = sigma_annualized(returns, window=4, annualize=252)
    assert sigma == pytest.approx(0.2898275349, rel=1e-6)


def test_wp6_sigma_eff_floor_applied() -> None:
    assert sigma_eff(0.1, 0.2) == 0.2
    assert sigma_eff(0.3, 0.2) == 0.3


def test_wp6_tradability_filter_center() -> None:
    value = tradability_filter(sigma=0.2, sigma_max=0.2, kappa=0.05)
    assert value == pytest.approx(0.5, rel=1e-6)
