# Ref: WP §9

import numpy as np

from src.services.signal_service import SignalService


def test_wp9_tilt_cosine_zero_vector_stable() -> None:
    weights = SignalService._softmax(scores=np.array([0.0, 0.0, 0.0]), kappa_tilt=1.0)
    assert weights.shape == (3,)
    assert not (weights != weights).any()
    assert abs(weights.sum() - 1.0) < 1e-12
    assert all(abs(weight - 1 / 3) < 1e-12 for weight in weights)


def test_wp9_tilt_softmax_temperature_effect() -> None:
    low_kappa = SignalService._softmax(scores=np.array([0.0, 1.0]), kappa_tilt=0.1)
    high_kappa = SignalService._softmax(scores=np.array([0.0, 1.0]), kappa_tilt=2.0)
    assert low_kappa[1] > high_kappa[1]
