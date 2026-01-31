from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class DbConfig(BaseModel):
    dsn: str
    schema_in: str = "public"
    schema_out: str = "cta"


class LoggingConfig(BaseModel):
    dir: str = "./logs"
    level: str = "INFO"
    retention_days: int = 365
    prefix: str = "app"


class FeatureConfig(BaseModel):
    enabled_features: list[str] = Field(
        default_factory=lambda: [
            "r_log_daily",
            "sigma_ann",
            "sigma_eff",
            "f_sigma",
            "T",
            "gate_state",
            "down_drift",
            "T_RATE",
            "rate_pref",
        ]
    )
    short_window: int = 20
    long_window: int = 60
    vol_window: int = 20
    annualize: int = 252
    theta_on: float = 0.5
    theta_off: float = 0.2
    theta_minus: float = 0.5
    sigma_min: float = 0.0
    sigma_max: float = 1.0
    kappa_sigma: float = 0.1
    rate_k: float = 2.0
    theta_rate: float = 0.0


class SignalConfig(BaseModel):
    tilt_factors: list[str] = Field(default_factory=lambda: ["SMB", "QMJ"])
    tilt_lookback_days: int = 60
    tilt_scales: dict[str, float] = Field(default_factory=lambda: {"SMB": 1.0, "QMJ": 1.0})
    tilt_eps: float = 1e-12
    tilt_temperature: float = 1.0


class PortfolioConfig(BaseModel):
    sigma_target: float = 0.1
    risk_buckets: list[str] = Field(default_factory=lambda: ["VALUE", "GROWTH", "CYCLE", "GOLD"])
    bucket_signal_names: list[str] = Field(
        default_factory=lambda: [
            "T",
            "gate_state",
            "sigma_eff",
            "f_sigma",
            "raw_weight_component_risk_budget",
            "raw_weight_component_gate",
            "raw_weight_component_trend",
            "raw_weight_component_inv_sigma_eff",
            "raw_weight_component_f_sigma",
            "rate_pref",
        ]
    )


class StrategyConfig(BaseModel):
    default_strategy_id: str = "cta_trend_v1"
    default_version: str = "1.0.0"
    default_portfolio_id: str = "main"


class AppConfig(BaseModel):
    env: str = "dev"
    db: DbConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    signals: SignalConfig = Field(default_factory=SignalConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    bucket_reco: "BucketRecoConfig" = Field(default_factory=lambda: BucketRecoConfig())


class BucketRecoProxyConfig(BaseModel):
    weight_mode: Literal["equal", "inv_vol"] = "inv_vol"
    annualize: int = 252
    clip_min: float | None = None
    clip_max: float | None = None
    join: Literal["inner", "outer"] = "inner"


class BucketRecoTrendConfig(BaseModel):
    short_window: int = 20
    long_window: int = 60
    vol_window: int = 20
    eps: float = 1e-12


class BucketRecoWindowSpec(BaseModel):
    label: str
    months: int


class BucketRecoConsistencyConfig(BaseModel):
    min_points: int = 20
    min_coverage: float = 0.6
    windows: list[BucketRecoWindowSpec] = Field(
        default_factory=lambda: [
            BucketRecoWindowSpec(label="3M", months=3),
            BucketRecoWindowSpec(label="12M", months=12),
            BucketRecoWindowSpec(label="36M", months=36),
        ]
    )


class BucketRecoScoreConfig(BaseModel):
    w_rho: float = 0.5
    w_h: float = 0.5
    lambdas: dict[str, float] = Field(default_factory=lambda: {"3M": 0.2, "12M": 0.3, "36M": 0.5})
    top_k: int = 50
    score_threshold: float = 0.0
    min_count: int = 1
    beta_min: float = 0.3


class BucketRecoBetaConfig(BaseModel):
    u_mode: Literal["absolute", "quantile"] = "quantile"
    u_value: float = 0.9
    mad_eps: float = 1e-12
    normalize_eps: float = 1e-12
    strict_decode: bool = True
    vector_fields: list[str] = Field(default_factory=lambda: ["SMB", "QMJ"])


class BucketRecoConvexHullConfig(BaseModel):
    n: int = 3
    epsilon: float = 0.0
    M: int = 1024
    rng_seed: int = 42
    topk_per_iter: int = 64
    violation_tol: float = 1e-9
    max_iters: int | None = None
    clip_rhopow: float | None = None
    clip_viol: float | None = None
    diversity_beta: float = 1.5
    nms_cos_thresh: float | None = 0.98


class BucketRecoConfig(BaseModel):
    proxy: BucketRecoProxyConfig = Field(default_factory=BucketRecoProxyConfig)
    trend: BucketRecoTrendConfig = Field(default_factory=BucketRecoTrendConfig)
    consistency: BucketRecoConsistencyConfig = Field(default_factory=BucketRecoConsistencyConfig)
    score: BucketRecoScoreConfig = Field(default_factory=BucketRecoScoreConfig)
    beta: BucketRecoBetaConfig = Field(default_factory=BucketRecoBetaConfig)
    convex_hull: BucketRecoConvexHullConfig = Field(default_factory=BucketRecoConvexHullConfig)


def load_app_config(path: Path | None = None) -> AppConfig:
    config_path = path or Path(os.getenv("APP_CONFIG_PATH", "config/app.yaml"))
    if not config_path.exists():
        raise FileNotFoundError(f"app config not found: {config_path}")
    data: Any = yaml.safe_load(config_path.read_text())
    return AppConfig.model_validate(data)
