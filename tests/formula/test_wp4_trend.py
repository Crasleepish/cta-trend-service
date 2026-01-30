# Ref: WP §4.1, §4.2

from __future__ import annotations

import pandas as pd
import pytest

from src.features import computer


def test_wp4_trend_strength_known_value() -> None:
    prices = [1.0, 1.02, 1.01, 1.03, 1.06, 1.08]
    frame = pd.DataFrame({"B1": prices}, index=pd.to_datetime(range(len(prices))))
    returns = computer.log_returns(frame)
    sigma = computer.sigma_annualized(returns, window=4, annualize=252)
    trend = computer.trend_strength(frame, sigma, short_window=2, long_window=4)
    value = trend.iloc[-1]["B1"]
    assert value == pytest.approx(0.0942020875, rel=1e-6)


def test_wp4_trend_strength_signs() -> None:
    up = [1.0, 1.02, 1.01, 1.03, 1.06, 1.08]
    down = [1.08, 1.06, 1.03, 1.01, 1.0, 0.98]
    for series, sign in [(up, 1), (down, -1)]:
        frame = pd.DataFrame({"B1": series}, index=pd.to_datetime(range(len(series))))
        returns = computer.log_returns(frame)
        sigma = computer.sigma_annualized(returns, window=4, annualize=252)
        trend = computer.trend_strength(frame, sigma, short_window=2, long_window=4)
        assert trend.iloc[-1]["B1"] * sign > 0


def test_wp4_gate_hysteresis_state_hold() -> None:
    trends = pd.DataFrame(
        {"B1": [0.2, 1.2, 0.8, 0.4, 0.9, 0.5]},
        index=pd.to_datetime(range(6)),
    )
    states = computer.hysteresis_gate(trends, theta_on=1.0, theta_off=0.6)
    assert states["B1"].to_list() == [0, 1, 1, 0, 0, 0]
