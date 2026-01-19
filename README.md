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
