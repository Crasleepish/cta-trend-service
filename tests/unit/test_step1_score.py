from __future__ import annotations

import numpy as np

from src.bucket_reco.score.step1 import (
    CandidateConstraints,
    WindowScore,
    aggregate_scores,
    score_window,
    select_candidates,
)


def test_score_window_beta_gate() -> None:
    result = score_window(rho_T=0.9, HR=1.0, beta=0.0, w_rho=0.5, w_h=0.5)
    assert result.valid is False
    assert np.isnan(result.score)


def test_score_window_weight_consistency() -> None:
    result_rho = score_window(rho_T=0.7, HR=0.2, beta=1.0, w_rho=1.0, w_h=0.0)
    assert result_rho.valid is True
    assert np.isclose(result_rho.score, 0.7)

    result_hr = score_window(rho_T=0.7, HR=0.8, beta=1.0, w_rho=0.0, w_h=1.0)
    assert np.isclose(result_hr.score, 2 * 0.8 - 1)


def test_aggregate_scores_renormalize_lambda() -> None:
    scores = {"3M": WindowScore(0.4, True), "12M": WindowScore(float("nan"), False)}
    lambdas = {"3M": 0.3, "12M": 0.7}
    agg = aggregate_scores(scores, lambdas)
    assert agg.valid is True
    assert np.isclose(agg.score, 0.4)


def test_aggregate_scores_no_valid_windows() -> None:
    scores = {"3M": WindowScore(float("nan"), False)}
    lambdas = {"3M": 1.0}
    agg = aggregate_scores(scores, lambdas)
    assert agg.valid is False
    assert np.isnan(agg.score)


def test_select_candidates_threshold_and_topk() -> None:
    funds = ["F1", "F2", "F3"]
    scores = {
        "F1": WindowScore(0.9, True),
        "F2": WindowScore(0.4, True),
        "F3": WindowScore(float("nan"), False),
    }
    constraints = CandidateConstraints(top_k=1, score_threshold=0.5, min_count=1)
    selected = select_candidates(funds, scores, constraints)
    assert selected == ["F1"]


def test_select_candidates_min_count() -> None:
    funds = ["F1", "F2"]
    scores = {"F1": WindowScore(0.4, True), "F2": WindowScore(0.3, True)}
    constraints = CandidateConstraints(top_k=2, score_threshold=0.5, min_count=1)
    selected = select_candidates(funds, scores, constraints)
    assert selected == []
