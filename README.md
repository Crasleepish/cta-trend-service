# CTA Trend Service

CTA trend-following allocation microservice (FastAPI + PostgreSQL).

## Quickstart

1) Prepare config:

```bash
cp config/app.yaml.template config/app.yaml
```

2) Edit `config/app.yaml` with your real DSN and settings.

3) Run the service:

```bash
./run.sh --reload --host 0.0.0.0 --port 8081
```

4) Health check:

```bash
curl http://localhost:8081/health
```

## Config

- Default config path: `config/app.yaml`
- Override: `APP_CONFIG_PATH=/path/to/app.yaml`

## Logs

Logs are written to `./logs/` with daily rotation and gzip archive,
retained for 365 days.

## bucket_reco

Bucket Tradable Assets recommender tool for a single bucket. It builds a proxy
index, runs Step1 trend consistency filters, then Step2 beta convex-hull
selection to produce representative candidates. Configuration lives under
`bucket_reco.*` in `config/app.yaml`, and can be overridden via CLI flags.

Example:

```bash
uv run python -m src.bucket_reco.runner \
  --bucket-name GROWTH \
  --proxy-index 399372.SZ,399374.SZ,399376.SZ \
  --as-of-date 2025-12-31 \
  --beta-vector-fields SMB,QMJ \
  --score-beta-min 0.3
```

## app.yaml Reference

| Key | Meaning |
| --- | --- |
| `env` | Environment name for the service. |
| `db.dsn` | Database DSN for Postgres (sync SQLAlchemy). |
| `db.schema_in` | Schema for input/source tables. |
| `db.schema_out` | Schema for output tables. |
| `logging.dir` | Log directory path. |
| `logging.level` | Log level (`INFO`, `DEBUG`, etc.). |
| `logging.retention_days` | Days to retain rotated logs. |
| `logging.prefix` | Log file prefix. |
| `bucket_reco.proxy.weight_mode` | Proxy weight mode (`equal` or `inv_vol`). |
| `bucket_reco.proxy.annualize` | Annualization factor for volatility (e.g., 252). |
| `bucket_reco.proxy.clip_min` | Minimum weight clip for proxy weights (nullable). |
| `bucket_reco.proxy.clip_max` | Maximum weight clip for proxy weights (nullable). |
| `bucket_reco.proxy.join` | Align mode for proxy series (`inner`/`outer`). |
| `bucket_reco.trend.short_window` | Short moving-average window length. |
| `bucket_reco.trend.long_window` | Long moving-average window length. |
| `bucket_reco.trend.vol_window` | Volatility window length for trend score. |
| `bucket_reco.trend.eps` | Epsilon guard for divide-by-zero. |
| `bucket_reco.consistency.min_points` | Minimum valid points per window. |
| `bucket_reco.consistency.min_coverage` | Minimum coverage ratio per window. |
| `bucket_reco.consistency.windows` | Window specs list (`label`, `months`). |
| `bucket_reco.score.w_rho` | Weight for correlation term in Step1 score. |
| `bucket_reco.score.w_h` | Weight for hit-ratio term in Step1 score. |
| `bucket_reco.score.lambdas` | Window aggregation weights (by label). |
| `bucket_reco.score.top_k` | Max candidate count after Step1 scoring. |
| `bucket_reco.score.score_threshold` | Minimum Step1 score threshold. |
| `bucket_reco.score.min_count` | Minimum candidates required after Step1. |
| `bucket_reco.score.beta_min` | Minimum beta to accept a window score. |
| `bucket_reco.beta.u_mode` | Uncertainty filter mode (`absolute`/`quantile`). |
| `bucket_reco.beta.u_value` | Uncertainty threshold value. |
| `bucket_reco.beta.mad_eps` | MAD epsilon guard. |
| `bucket_reco.beta.normalize_eps` | Legacy normalize epsilon (unused in Step2). |
| `bucket_reco.beta.strict_decode` | Fail on invalid covariance decode when true. |
| `bucket_reco.beta.vector_fields` | Exposure vector fields used for Step2 selection. |
| `bucket_reco.convex_hull.n` | Target number of selections. |
| `bucket_reco.convex_hull.epsilon` | Relative gain stop threshold. |
| `bucket_reco.convex_hull.M` | Number of sampled sphere directions. |
| `bucket_reco.convex_hull.rng_seed` | RNG seed for sphere sampling. |
| `bucket_reco.convex_hull.topk_per_iter` | Candidate shortlist size per iteration. |
| `bucket_reco.convex_hull.violation_tol` | Constraint violation tolerance. |
| `bucket_reco.convex_hull.max_iters` | Max greedy iterations (nullable). |
| `bucket_reco.convex_hull.clip_rhopow` | Clip for rho power (nullable). |
| `bucket_reco.convex_hull.clip_viol` | Clip for violation values (nullable). |
| `bucket_reco.convex_hull.diversity_beta` | Diversity weighting exponent. |
| `bucket_reco.convex_hull.nms_cos_thresh` | Cosine threshold for NMS (nullable). |
