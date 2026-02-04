# Ref: WP §5.1, §5.3
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import computer


def test_wp5_path_quality_monotone_up() -> None:
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)
    z = computer.path_quality_z(prices, window_days=5)
    assert np.isclose(float(z.iloc[-1, 0]), 1.0)


def test_wp5_path_quality_monotone_down() -> None:
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"A": [5.0, 4.0, 3.0, 2.0, 1.0]}, index=dates)
    z = computer.path_quality_z(prices, window_days=5)
    assert np.isclose(float(z.iloc[-1, 0]), 0.0)


def test_wp5_path_quality_g_mapping() -> None:
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    z = pd.DataFrame({"A": [0.65, 1.0]}, index=dates)
    g = computer.path_quality_g(z, x0=0.65, gamma=2.0)
    assert np.isclose(float(g.iloc[0, 0]), 0.0)
    assert np.isclose(float(g.iloc[1, 0]), 1.0)
