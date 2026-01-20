from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowSpec:
    label: str
    months: int


def window_slices(
    dates: pd.DatetimeIndex,
    specs: Sequence[Mapping[str, int | str]],
) -> dict[str, pd.DatetimeIndex]:
    if dates.empty:
        raise ValueError("dates must be non-empty")
    t_end = dates.max()
    slices: dict[str, pd.DatetimeIndex] = {}
    for spec in specs:
        label = str(spec["label"])
        months = int(spec["months"])
        if months <= 0:
            raise ValueError("months must be positive")
        t_start = t_end - pd.DateOffset(months=months)
        mask = (dates > t_start) & (dates <= t_end)
        slices[label] = dates[mask]
    return slices


def trend_corr(
    T_i: pd.Series,
    T_G: pd.Series,
    window_idx: pd.DatetimeIndex,
    *,
    min_points: int = 20,
    min_coverage: float = 0.6,
) -> float | None:
    aligned = pd.concat(
        [T_i.reindex(window_idx), T_G.reindex(window_idx)], axis=1, keys=["i", "g"]
    ).dropna()
    expected = len(window_idx)
    if not _is_valid(expected, len(aligned), min_points, min_coverage):
        return None
    return float(aligned["i"].corr(aligned["g"]))


def hit_ratio(
    T_i: pd.Series,
    T_G: pd.Series,
    window_idx: pd.DatetimeIndex,
    *,
    min_points: int = 20,
    min_coverage: float = 0.6,
) -> float | None:
    aligned = pd.concat(
        [T_i.reindex(window_idx), T_G.reindex(window_idx)], axis=1, keys=["i", "g"]
    ).dropna()
    expected = len(window_idx)
    if aligned.empty:
        return None
    sign_i = np.sign(aligned["i"].to_numpy())
    sign_g = np.sign(aligned["g"].to_numpy())
    mask = (sign_i != 0) & (sign_g != 0)
    n_dir = int(mask.sum())
    if not _is_valid(expected, n_dir, min_points, min_coverage):
        return None
    return float((sign_i[mask] == sign_g[mask]).mean())


def _is_valid(expected: int, actual: int, min_points: int, min_coverage: float) -> bool:
    if expected <= 0:
        return False
    if actual < min_points:
        return False
    coverage = actual / expected
    return coverage >= min_coverage
