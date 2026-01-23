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


class AppConfig(BaseModel):
    env: str = "dev"
    db: DbConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
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


class BucketRecoBetaConfig(BaseModel):
    u_mode: Literal["absolute", "quantile"] = "quantile"
    u_value: float = 0.9
    mad_eps: float = 1e-12
    normalize_eps: float = 1e-12
    strict_decode: bool = True


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
