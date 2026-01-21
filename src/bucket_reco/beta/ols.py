from __future__ import annotations

import numpy as np
import pandas as pd


def ols_beta(y: pd.Series | np.ndarray, x: pd.Series | np.ndarray) -> tuple[float, float]:
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if y_arr.shape != x_arr.shape:
        raise ValueError("x and y must have the same shape")
    if y_arr.size == 0:
        raise ValueError("x and y must be non-empty")

    x_mean = float(np.mean(x_arr))
    y_mean = float(np.mean(y_arr))
    var_x = float(np.mean((x_arr - x_mean) ** 2))
    if np.isclose(var_x, 0.0):
        raise ValueError("x variance is zero")

    cov_xy = float(np.mean((x_arr - x_mean) * (y_arr - y_mean)))
    beta = cov_xy / var_x
    alpha = y_mean - beta * x_mean
    return alpha, beta


def beta_positive_gate(beta: float) -> bool:
    return bool(np.isfinite(beta) and beta > 0.0)
