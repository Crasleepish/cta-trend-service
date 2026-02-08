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
    # Source: cta.signal_weekly @ 2019-06-28 (DB snapshot)
    rows: list[dict[str, object]] = []
    buckets = {
        "VALUE": {
            "raw_weight_component_risk_budget": 1.0,
            "raw_weight_component_gate": 0.0,
            "raw_weight_component_trend": -0.24570359200004507,
            "raw_weight_component_inv_sigma_eff": 5.7570275293718,
            "raw_weight_component_f_sigma": 0.9997421803767416,
            "raw_weight_component_path_quality": 0.0,
            "sigma_eff": 0.17370075006556707,
        },
        "GROWTH": {
            "raw_weight_component_risk_budget": 1.0,
            "raw_weight_component_gate": 0.0,
            "raw_weight_component_trend": -0.23250225391159246,
            "raw_weight_component_inv_sigma_eff": 4.288597438690738,
            "raw_weight_component_f_sigma": 0.9995327764218317,
            "raw_weight_component_path_quality": 0.0,
            "sigma_eff": 0.23317646720072868,
        },
        "CYCLE": {
            "raw_weight_component_risk_budget": 1.0,
            "raw_weight_component_gate": 0.0,
            "raw_weight_component_trend": -0.2580571313508658,
            "raw_weight_component_inv_sigma_eff": 5.546586327581266,
            "raw_weight_component_f_sigma": 0.9997246216773829,
            "raw_weight_component_path_quality": 0.0,
            "sigma_eff": 0.18029107291224225,
        },
        "GOLD": {
            "raw_weight_component_risk_budget": 1.0,
            "raw_weight_component_gate": 0.0,
            "raw_weight_component_trend": 0.29511715801180527,
            "raw_weight_component_inv_sigma_eff": 6.27250156865514,
            "raw_weight_component_f_sigma": 0.9997764696351323,
            "raw_weight_component_path_quality": 0.31063947401573827,
            "sigma_eff": 0.1594260262918365,
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


def test_wp8_risk_cap_gold_2019_06_28_ground_truth() -> None:
    # WP §8 risk-sleeve cap, computed from DB-derived signals (see docs/WP8_risk_2019-06-28.csv)
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

    assert np.isclose(alpha, 0.0, atol=1e-12)
    assert np.isclose(w_risk["GOLD"], 0.0, atol=1e-12)
