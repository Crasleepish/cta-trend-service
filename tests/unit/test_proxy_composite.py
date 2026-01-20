from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.bucket_reco.proxy.composite import (
    build_composite_proxy,
    compute_vol_annualized,
    compute_weights_equal,
    compute_weights_inv_vol,
)


def test_compute_weights_equal() -> None:
    weights = compute_weights_equal(3)
    assert np.allclose(weights, [1 / 3, 1 / 3, 1 / 3])


def test_compute_weights_inv_vol() -> None:
    sigmas = np.array([1.0, 2.0, 4.0])
    weights = compute_weights_inv_vol(sigmas)
    expected = np.array([1.0, 0.5, 0.25])
    expected = expected / expected.sum()
    assert np.allclose(weights, expected)
    assert np.isclose(weights.sum(), 1.0)


def test_compute_weights_inv_vol_clip() -> None:
    sigmas = np.array([0.1, 10.0, 10.0])
    weights = compute_weights_inv_vol(sigmas, clip=(0.1, 0.6))
    assert np.all(weights <= 0.6 + 1e-12)
    assert np.all(weights >= 0.1 - 1e-12)
    assert np.isclose(weights.sum(), 1.0)


def test_build_composite_proxy_same_prices() -> None:
    idx = pd.to_datetime([date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)])
    prices = pd.Series([1.0, 1.1, 1.21], index=idx)
    price_dict = {"IDX1": prices, "IDX2": prices.copy(), "IDX3": prices.copy()}
    weights = {"IDX1": 0.2, "IDX2": 0.3, "IDX3": 0.5}

    returns, _ = build_composite_proxy(price_dict, weights)
    single_returns = np.log(prices / prices.shift(1)).dropna()

    assert np.allclose(returns.values, single_returns.values)


def test_build_composite_proxy_handles_missing_values() -> None:
    idx = pd.to_datetime([date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)])
    series_a = pd.Series([1.0, np.nan, 1.2], index=idx)
    series_b = pd.Series([2.0, 2.1, 2.2], index=idx)
    price_dict = {"A": series_a, "B": series_b}
    weights = {"A": 0.5, "B": 0.5}

    returns, index = build_composite_proxy(price_dict, weights)
    assert not returns.isna().any()
    assert not index.isna().any()
    assert len(returns) == 1


def test_compute_vol_annualized() -> None:
    returns = pd.Series([0.0, 0.1, -0.1])
    vol = compute_vol_annualized(returns, annualize=252)
    expected = float(returns.std(ddof=0) * np.sqrt(252))
    assert np.isclose(vol, expected)
