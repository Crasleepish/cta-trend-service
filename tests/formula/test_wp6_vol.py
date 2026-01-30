# Ref: WP §6.1, §6.2, §6.3

from __future__ import annotations

import pandas as pd
import pytest

from src.features import computer


def test_wp6_sigma_annualization_known_series() -> None:
    returns = pd.DataFrame({"B1": [0.01, -0.01, 0.02, -0.02]})
    sigma = computer.sigma_annualized(returns, window=4, annualize=252)
    assert sigma.iloc[-1]["B1"] == pytest.approx(0.2898275349, rel=1e-6)


def test_wp6_sigma_eff_floor_applied() -> None:
    data = pd.DataFrame({"B1": [0.1, 0.3]})
    clipped = computer.sigma_eff(data, sigma_min=0.2)
    assert clipped.iloc[0]["B1"] == 0.2
    assert clipped.iloc[1]["B1"] == 0.3


def test_wp6_tradability_filter_center() -> None:
    data = pd.DataFrame({"B1": [0.2]})
    value = computer.tradability_filter(data, sigma_max=0.2, kappa=0.05)
    assert value.iloc[0]["B1"] == pytest.approx(0.5, rel=1e-6)
