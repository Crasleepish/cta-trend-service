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
index, runs Step1 trend consistency filters, then Step2 beta clustering to
produce representatives and Top-K candidates. Configuration lives under
`bucket_reco.*` in `config/app.yaml`, and can be overridden via CLI flags.

Example:

```bash
uv run python src/bucket_reco/runner.py \
  --bucket-name GROWTH \
  --proxy-index 000300.SH,000905.SH,000852.SH \
  --as-of-date 2025-12-31
```
