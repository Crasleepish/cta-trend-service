from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
import pandas as pd

from src.utils.series import align_series, build_index_from_returns, log_return


def compute_vol_annualized(returns: pd.Series, *, annualize: int = 252) -> float:
    if returns.empty:
        raise ValueError("returns must be non-empty")
    return float(returns.std(ddof=0) * np.sqrt(annualize))


def compute_weights_equal(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    return np.full(n, 1.0 / n)


def compute_weights_inv_vol(
    sigmas: np.ndarray,
    *,
    clip: tuple[float | None, float | None] | None = None,
) -> np.ndarray:
    if sigmas.size == 0:
        raise ValueError("sigmas must be non-empty")
    if np.any(sigmas <= 0):
        raise ValueError("sigmas must be positive")

    inv = 1.0 / sigmas
    weights = inv / inv.sum()

    if clip:
        lower, upper = clip
        if lower is not None:
            weights = np.maximum(weights, lower)
        if upper is not None:
            weights = np.minimum(weights, upper)
        total = weights.sum()
        if total <= 0:
            raise ValueError("clipped weights sum to zero")
        weights = weights / total

    return weights


def build_composite_proxy(
    price_dict: Mapping[str, pd.Series],
    weights: Mapping[str, float],
    *,
    join: Literal["inner", "outer"] = "inner",
) -> tuple[pd.Series, pd.Series]:
    if not price_dict:
        raise ValueError("price_dict must be non-empty")

    missing = [key for key in price_dict.keys() if key not in weights]
    if missing:
        raise ValueError(f"missing weights for keys: {', '.join(missing)}")

    series_list = [price_dict[key].rename(key) for key in price_dict.keys()]
    aligned = align_series(series_list, join=join)
    aligned = aligned.dropna()
    if aligned.empty:
        raise ValueError("aligned prices are empty after dropping NaN")

    returns = aligned.apply(log_return)
    returns = returns.dropna()
    if returns.empty:
        raise ValueError("aligned returns are empty")

    weight_vec = np.array([weights[key] for key in aligned.columns], dtype=float)
    if weight_vec.sum() == 0:
        raise ValueError("weights sum to zero")
    weight_vec = weight_vec / weight_vec.sum()

    composite_returns = returns.mul(weight_vec, axis=1).sum(axis=1)
    composite_index = build_index_from_returns(composite_returns)
    return composite_returns, composite_index
