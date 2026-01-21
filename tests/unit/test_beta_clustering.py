from __future__ import annotations

import numpy as np
import pandas as pd

from src.bucket_reco.beta.clustering import (
    choose_n_eff,
    select_representative,
    topk_candidates,
    ward_cluster,
)


def _real_beta_samples() -> dict[str, tuple[float, float]]:
    return {
        "F161604.OF": (9.406573218280596e-05, 0.0002932633433694149),
        "960022.OF": (-0.0037390522525421192, 0.017392191567361187),
        "952303.OF": (0.0005308297887081913, -0.019915032222256965),
        "952003.OF": (-0.00035182141618152814, -0.024844572351335463),
        "740101.OF": (-0.29337140454448674, 0.07106066586664145),
        "700002.OF": (-0.3061744684219334, 0.23030956463618502),
    }


def test_ward_cluster_separates_two_groups() -> None:
    samples = _real_beta_samples()
    codes = list(samples.keys())
    beta = np.array([samples[c] for c in codes], dtype=float)
    offset = np.array([5.0, 5.0])
    beta[3:] = beta[3:] + offset
    beta_hat = pd.DataFrame(beta, index=codes, columns=["SMB", "QMJ"])

    labels = ward_cluster(beta_hat, n_clusters=2)
    first_group_label = set(labels.iloc[:3])
    second_group_label = set(labels.iloc[3:])

    assert len(first_group_label) == 1
    assert len(second_group_label) == 1
    assert first_group_label != second_group_label


def test_choose_n_eff_min_cluster_size() -> None:
    labels = pd.Series(
        [1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
        index=[f"F{i}" for i in range(10)],
    )
    assert choose_n_eff(labels, m_min=3) == 2


def test_select_representative_eta_effect() -> None:
    samples = _real_beta_samples()
    base = np.array(samples["F161604.OF"])
    outlier = base + np.array([10.0, 10.0])
    cluster = pd.DataFrame(
        [base, outlier],
        index=["F161604.OF", "OUTLIER.OF"],
        columns=["SMB", "QMJ"],
    )
    scores = pd.Series(
        {
            "F161604.OF": float(base.sum()),
            "OUTLIER.OF": float(outlier.sum()),
        }
    )
    centroid = base

    assert select_representative(cluster, scores, centroid, eta=0.0) == "OUTLIER.OF"
    assert select_representative(cluster, scores, centroid, eta=3.0) == "F161604.OF"


def test_topk_candidates_stable_ordering() -> None:
    samples = _real_beta_samples()
    codes = ["952003.OF", "960022.OF", "952303.OF"]
    scores = pd.Series({code: sum(samples[code]) for code in codes})
    tie_value = float((scores.iloc[0] + scores.iloc[1]) / 2.0)
    scores.iloc[0] = tie_value
    scores.iloc[1] = tie_value

    ordered = topk_candidates(codes, scores, k=2)
    expected = sorted(codes[:2])
    assert ordered == expected
