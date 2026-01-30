# Ref: WP §7

import pytest


@pytest.mark.xfail(reason="WP §7 raw weight not implemented yet", strict=False)
def test_wp7_raw_weight_zero_trend_returns_zero_or_min() -> None:
    assert False


@pytest.mark.xfail(reason="WP §7 raw weight scaling not implemented yet", strict=False)
def test_wp7_raw_weight_scaling_consistency() -> None:
    assert False
