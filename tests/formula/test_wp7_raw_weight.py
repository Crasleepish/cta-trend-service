# Ref: WP §7

from __future__ import annotations


def raw_weight(
    *,
    gate_state: float,
    f_sigma: float,
    trend_score: float,
    sigma_eff: float,
    path_quality: float = 1.0,
) -> float:
    return gate_state * f_sigma * trend_score * path_quality * (1.0 / sigma_eff)


def test_wp7_raw_weight_zero_trend_returns_zero_or_min() -> None:
    value = raw_weight(
        gate_state=1.0,
        f_sigma=1.0,
        trend_score=0.0,
        sigma_eff=0.2,
        path_quality=1.0,
    )
    assert value == 0.0


def test_wp7_raw_weight_scaling_consistency() -> None:
    base = raw_weight(
        gate_state=1.0,
        f_sigma=0.8,
        trend_score=1.0,
        sigma_eff=0.5,
        path_quality=1.0,
    )
    doubled = raw_weight(
        gate_state=1.0,
        f_sigma=0.8,
        trend_score=2.0,
        sigma_eff=0.5,
        path_quality=1.0,
    )
    assert doubled == base * 2.0
