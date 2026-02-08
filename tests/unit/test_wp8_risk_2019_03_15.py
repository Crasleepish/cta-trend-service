from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.config import PortfolioConfig
from src.services.portfolio_service import PortfolioService


@dataclass
class _NoopRepo:
    pass


def _signal_rows() -> list[dict[str, object]]:
    # Source: cta.signal_weekly @ 2019-03-15 (DB snapshot)
    rows: list[dict[str, object]] = []
    buckets = {
        "VALUE": {
            "raw_weight_component_risk_budget": 1.0,
            "raw_weight_component_gate": 1.0,
            "raw_weight_component_trend": 0.2845461149885471,
            "raw_weight_component_inv_sigma_eff": 2.9363023331121263,
            "raw_weight_component_f_sigma": 0.45810413681525813,
            "raw_weight_component_path_quality": 0.6577706902977881,
            "sigma_eff": 0.34056438559585267,
        },
        "GROWTH": {
            "raw_weight_component_risk_budget": 1.0,
            "raw_weight_component_gate": 1.0,
            "raw_weight_component_trend": 0.3646148592395068,
            "raw_weight_component_inv_sigma_eff": 2.4389814993246355,
            "raw_weight_component_f_sigma": 0.5022367901730559,
            "raw_weight_component_path_quality": 0.5321224701878327,
            "sigma_eff": 0.4100072100903202,
        },
        "CYCLE": {
            "raw_weight_component_risk_budget": 1.0,
            "raw_weight_component_gate": 1.0,
            "raw_weight_component_trend": 0.28008784552218324,
            "raw_weight_component_inv_sigma_eff": 2.799125124888262,
            "raw_weight_component_f_sigma": 0.5419428013699246,
            "raw_weight_component_path_quality": 0.5504916885402767,
            "sigma_eff": 0.35725448323426373,
        },
        "GOLD": {
            "raw_weight_component_risk_budget": 1.0,
            "raw_weight_component_gate": 0.0,
            "raw_weight_component_trend": 0.04963267904928012,
            "raw_weight_component_inv_sigma_eff": 11.042785509163494,
            "raw_weight_component_f_sigma": 0.8947940405296421,
            "raw_weight_component_path_quality": 0.0,
            "sigma_eff": 0.09055686168767679,
        },
    }
    for bucket, values in buckets.items():
        for name, value in values.items():
            rows.append(
                {
                    "instrument_id": bucket,
                    "signal_name": name,
                    "value": value,
                }
            )
    return rows


def test_wp8_risk_cap_2019_03_15_ground_truth() -> None:
    # WP §8 risk-sleeve cap, computed from DB-derived signals
    risk_buckets = ["VALUE", "GROWTH", "CYCLE", "GOLD"]
    config = PortfolioConfig(
        portfolio_id="main",
        risk_buckets=risk_buckets,
        sigma_target=0.14824680700851597,
        bucket_signal_names=[],
    )
    svc = PortfolioService(
        bucket_repo=_NoopRepo(),
        signal_repo=_NoopRepo(),
        weight_repo=_NoopRepo(),
        config=config,
    )

    signal_df = pd.DataFrame(_signal_rows())
    raw_weights = svc._build_raw_bucket_weights(signal_df, risk_buckets)
    w0 = svc._normalize(raw_weights)
    sigma_eff = svc._bucket_signal(signal_df, risk_buckets, "sigma_eff")
    implied = svc._implied_risk_vol(w0, sigma_eff)
    alpha = 0.0 if implied <= 0 else min(1.0, config.sigma_target / implied)
    w_risk = {b: alpha * w0[b] for b in risk_buckets}

    assert np.isclose(implied, 0.21341410513607925, rtol=1e-10, atol=1e-12)
    assert np.isclose(alpha, 0.694643903288348, rtol=1e-10, atol=1e-12)
    assert np.isclose(w_risk["VALUE"], 0.24178135563912317, rtol=1e-10, atol=1e-12)
    assert np.isclose(w_risk["GROWTH"], 0.2282409872333789, rtol=1e-10, atol=1e-12)
    assert np.isclose(w_risk["CYCLE"], 0.224621560415846, rtol=1e-10, atol=1e-12)
    assert np.isclose(w_risk["GOLD"], 0.0, rtol=1e-10, atol=1e-12)
