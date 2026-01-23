from __future__ import annotations

import numpy as np

from src.bucket_reco.beta.convex_hull import select_representatives


def test_select_representatives_respects_max_iters() -> None:
    X = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=float,
    )

    selected_init = select_representatives(
        X,
        epsilon=0.0,
        M=128,
        rng_seed=0,
        topk_per_iter=16,
        violation_tol=1e-9,
        max_iters=0,
    )
    assert len(selected_init) == 2

    selected_full = select_representatives(
        X,
        epsilon=0.0,
        M=128,
        rng_seed=0,
        topk_per_iter=16,
        violation_tol=1e-9,
        max_iters=1,
    )
    assert len(selected_full) == 3


def test_select_representatives_skips_zero_vector() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    selected = select_representatives(
        X,
        epsilon=0.0,
        M=64,
        rng_seed=1,
        topk_per_iter=8,
        violation_tol=1e-9,
        max_iters=0,
    )
    assert set(selected.tolist()) == {1, 2}
