from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from utils.series import align_series, build_index_from_returns, log_return


def test_align_series_intersection() -> None:
    idx_a = pd.to_datetime([date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)])
    idx_b = pd.to_datetime([date(2026, 1, 2), date(2026, 1, 3), date(2026, 1, 4)])
    series_a = pd.Series([1.0, 1.1, 1.2], index=idx_a, name="A")
    series_b = pd.Series([2.0, 2.1, 2.2], index=idx_b, name="B")

    aligned = align_series([series_a, series_b], join="inner")
    expected_index = pd.to_datetime([date(2026, 1, 2), date(2026, 1, 3)])

    assert aligned.index.equals(expected_index)
    assert len(aligned) == 2


def test_align_series_outer_join() -> None:
    idx_a = pd.to_datetime([date(2026, 1, 1), date(2026, 1, 3)])
    idx_b = pd.to_datetime([date(2026, 1, 2), date(2026, 1, 3)])
    series_a = pd.Series([1.0, 1.2], index=idx_a, name="A")
    series_b = pd.Series([2.0, 2.2], index=idx_b, name="B")

    aligned = align_series([series_a, series_b], join="outer")
    expected_index = pd.to_datetime([date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)])

    assert aligned.index.equals(expected_index)
    assert aligned.isna().sum().sum() == 2


def test_log_return_e_powers() -> None:
    prices = pd.Series([1.0, np.e, np.e**2])
    returns = log_return(prices)
    assert np.allclose(returns.values, [1.0, 1.0])


def test_log_return_missing_values() -> None:
    prices = pd.Series([1.0, None, 2.0, 4.0])
    returns = log_return(prices, dropna=False)

    assert np.isnan(returns.iloc[1])
    assert np.isnan(returns.iloc[2])
    assert np.isclose(returns.iloc[3], np.log(2.0))


def test_build_index_zero_returns() -> None:
    returns = pd.Series([0.0, 0.0, 0.0])
    index = build_index_from_returns(returns)
    assert np.allclose(index.values, [1.0, 1.0, 1.0])


def test_build_index_missing_returns() -> None:
    returns = pd.Series([0.1, None, 0.2])
    index = build_index_from_returns(returns)
    expected = np.exp(np.array([0.1, 0.1, 0.3]))
    assert np.allclose(index.values, expected)


def test_build_index_constant_returns() -> None:
    c = 0.5
    returns = pd.Series([c, c, c])
    index = build_index_from_returns(returns)
    expected = np.exp(np.array([c, 2 * c, 3 * c]))
    assert np.allclose(index.values, expected)
