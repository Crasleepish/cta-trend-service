# Ref: WP §4.1, §4.2

from __future__ import annotations

import math
from typing import Iterable

import pytest


def _rolling_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def trend_strength(prices: Iterable[float], short: int, long: int, vol_window: int) -> float:
    prices = list(prices)
    if short <= 0 or long <= 0 or vol_window <= 0:
        raise ValueError("windows must be positive")
    if short >= long:
        raise ValueError("short window must be smaller than long window")
    if len(prices) < long + 1:
        raise ValueError("not enough data")

    ma_short = _rolling_mean(prices[-short:])
    ma_long = _rolling_mean(prices[-long:])

    returns = [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
    window = returns[-vol_window:]
    mean = _rolling_mean(window)
    var = sum((x - mean) ** 2 for x in window) / (vol_window - 1)
    sigma = math.sqrt(var) * math.sqrt(252)
    return (ma_short - ma_long) / sigma


def hysteresis_gate(trends: Iterable[float], theta_on: float, theta_off: float) -> list[int]:
    state = 0
    states: list[int] = []
    for t in trends:
        if t > theta_on:
            state = 1
        elif t < theta_off:
            state = 0
        states.append(state)
    return states


def test_wp4_trend_strength_known_value() -> None:
    prices = [1.0, 1.02, 1.01, 1.03, 1.06, 1.08]
    value = trend_strength(prices, short=2, long=4, vol_window=4)
    assert value == pytest.approx(0.0942020875, rel=1e-6)


def test_wp4_trend_strength_signs() -> None:
    up = [1.0, 1.02, 1.01, 1.03, 1.06, 1.08]
    down = [1.08, 1.06, 1.03, 1.01, 1.0, 0.98]
    assert trend_strength(up, short=2, long=4, vol_window=4) > 0
    assert trend_strength(down, short=2, long=4, vol_window=4) < 0


def test_wp4_gate_hysteresis_state_hold() -> None:
    trends = [0.2, 1.2, 0.8, 0.4, 0.9, 0.5]
    states = hysteresis_gate(trends, theta_on=1.0, theta_off=0.6)
    assert states == [0, 1, 1, 0, 0, 0]
