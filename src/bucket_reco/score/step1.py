from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class WindowScore:
    score: float
    valid: bool


@dataclass(frozen=True)
class CandidateConstraints:
    top_k: int
    score_threshold: float
    min_count: int = 1


def score_window(
    rho_T: float,
    HR: float,
    beta: float,
    w_rho: float,
    w_h: float,
) -> WindowScore:
    if not np.isfinite(beta) or beta <= 0:
        return WindowScore(score=float("nan"), valid=False)
    if not (np.isfinite(rho_T) and np.isfinite(HR)):
        return WindowScore(score=float("nan"), valid=False)

    score = w_rho * rho_T + w_h * (2 * HR - 1)
    return WindowScore(score=float(score), valid=True)


def aggregate_scores(
    scores_by_window: Mapping[str, WindowScore],
    lambdas: Mapping[str, float],
) -> WindowScore:
    valid_items = [
        (label, ws) for label, ws in scores_by_window.items() if ws.valid and np.isfinite(ws.score)
    ]
    if not valid_items:
        return WindowScore(score=float("nan"), valid=False)

    lambda_sum = sum(lambdas.get(label, 0.0) for label, _ in valid_items)
    if lambda_sum <= 0:
        return WindowScore(score=float("nan"), valid=False)

    weighted = 0.0
    for label, ws in valid_items:
        weight = lambdas.get(label, 0.0) / lambda_sum
        weighted += weight * ws.score
    return WindowScore(score=float(weighted), valid=True)


def select_candidates(
    funds: Sequence[str],
    scores: Mapping[str, WindowScore],
    constraints: CandidateConstraints,
) -> list[str]:
    if constraints.top_k <= 0:
        raise ValueError("top_k must be positive")
    if constraints.min_count <= 0:
        raise ValueError("min_count must be positive")

    eligible = []
    for fund in funds:
        ws = scores.get(fund)
        if ws is None or not ws.valid or not np.isfinite(ws.score):
            continue
        if ws.score < constraints.score_threshold:
            continue
        eligible.append((fund, ws.score))

    eligible.sort(key=lambda x: x[1], reverse=True)
    selected = [fund for fund, _ in eligible[: constraints.top_k]]
    if len(selected) < constraints.min_count:
        return []
    return selected
