from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.bucket_reco.beta.ols import beta_positive_gate, ols_beta


def test_ols_beta_positive() -> None:
    idx = pd.to_datetime([date(2026, 1, d) for d in range(1, 6)])
    x = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0], index=idx)
    y = 2.0 * x + 0.1

    alpha, beta = ols_beta(y, x)
    assert np.isclose(beta, 2.0, atol=1e-8)
    assert np.isclose(alpha, 0.1, atol=1e-8)
    assert beta_positive_gate(beta) is True


def test_ols_beta_negative() -> None:
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    y = -1.0 * x
    _, beta = ols_beta(y, x)
    assert np.isclose(beta, -1.0, atol=1e-8)
    assert beta_positive_gate(beta) is False


def test_ols_beta_zero_variance_raises() -> None:
    x = pd.Series([1.0, 1.0, 1.0])
    y = pd.Series([2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="variance is zero"):
        ols_beta(y, x)
