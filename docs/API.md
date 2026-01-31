# API 文档（V1）

> 说明：V1 无鉴权，同步执行。返回结构稳定（Pydantic）。  
> 默认参数来自 `app.yaml` 的 `strategy.*`。

## 1. 基础信息

- Base URL：`/`
- OpenAPI：`/docs`, `/openapi.json`
- 时间格式：ISO8601（UTC）

## 2. Health

### GET `/health`

返回服务状态、版本、时间与 DB 连通性。

**Response 200**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "time": "2026-01-31T22:10:00+00:00",
  "db_ok": true
}
```

---

## 3. Jobs（同步执行）

### POST `/jobs/full`
### POST `/jobs/feature`
### POST `/jobs/signal`
### POST `/jobs/portfolio`

触发一次同步 job。成功或失败都会返回 `run_id`。

**Header**
- `Idempotency-Key`（可选）：幂等键

**Request**
```json
{
  "rebalance_date": "2026-01-16",
  "calc_start": "2025-09-01",
  "calc_end": "2026-01-16",
  "lookback": {"market_days": 252},
  "universe": {
    "portfolio_id": "main",
    "bucket_ids": [1,2],
    "instrument_ids": ["A1","B1"]
  },
  "feature_set": {
    "enabled_features": ["T","sigma_eff"],
    "feature_params": {"short_window": 20}
  },
  "dry_run": false,
  "force_recompute": false,
  "strategy_id": "cta_trend_v1",
  "version": "1.0.0",
  "snapshot_id": "snap_20260116",
  "portfolio_id": "main"
}
```

**Response 200**
```json
{
  "run_id": "RUN_20260116_203015_8f3c",
  "status": "SUCCESS",
  "job_type": "FULL",
  "time_start": "2026-01-16T20:30:15+00:00",
  "time_end": "2026-01-16T20:30:42+00:00",
  "outputs": {
    "rows_upserted": {
      "feature_daily": 100,
      "signal_weekly": 20,
      "portfolio_weight_weekly": 10
    },
    "checks": {}
  },
  "error": null
}
```

**错误**
- 400：业务错误（输入缺失、倾斜无效、权重校验失败）
- 500：系统异常（DB、未捕获异常）

---

## 4. Weights

### GET `/weights/latest`

**Query**
- `portfolio_id`（可选，默认 `strategy.default_portfolio_id`）
- `strategy_id`（可选）
- `version`（可选）

**Response 200**
```json
{
  "rebalance_date": "2026-01-16",
  "portfolio_id": "main",
  "weights_sum": 1.0,
  "weights": [
    {
      "rebalance_date": "2026-01-16",
      "instrument_id": "A1",
      "target_weight": 0.12,
      "bucket": "GROWTH",
      "run_id": "RUN_20260116_203015_8f3c"
    }
  ]
}
```

### GET `/weights`

**Query**
- `rebalance_date`（必填）
- `portfolio_id`（可选）
- `strategy_id`（可选）
- `version`（可选）

### GET `/weights/history`

**Query**
- `start_date`（必填）
- `end_date`（必填）
- `portfolio_id`（可选）
- `strategy_id`（可选）
- `version`（可选）

---

## 5. Runs

### GET `/runs`

**Query**
- `limit`（默认 20，上限 200）
- `cursor`（可选，base64 游标）
- `status`（可选）
- `job_type`（可选）
- `strategy_id`（可选）
- `version`（可选）

**Response 200**
```json
{
  "items": [
    {
      "run_id": "RUN_20260116_203015_8f3c",
      "job_type": "FULL",
      "status": "SUCCESS",
      "strategy_id": "cta_trend_v1",
      "version": "1.0.0",
      "time_start": "2026-01-16T20:30:15+00:00",
      "time_end": "2026-01-16T20:30:42+00:00"
    }
  ],
  "next_cursor": "MjAyNi0wMS0xNlQyMDozMDoxNS4wMDAwMDBafFJVTl8yMDI2MDExNl8yMDMwMTVfOGYzYw=="
}
```

### GET `/runs/{run_id}`

**Response 200**
```json
{
  "run_id": "RUN_20260116_203015_8f3c",
  "job_type": "FULL",
  "status": "SUCCESS",
  "strategy_id": "cta_trend_v1",
  "version": "1.0.0",
  "snapshot_id": "snap_20260116",
  "time_start": "2026-01-16T20:30:15+00:00",
  "time_end": "2026-01-16T20:30:42+00:00",
  "input_range": {},
  "output_summary": {},
  "error_stack": null
}
```

---

## 6. Signals

### GET `/signals`

**Query**
- `rebalance_date`（必填）
- `strategy_id`（可选）
- `version`（可选）
- `include_meta`（默认 false）
- `instrument_id`（可选）
- `signal_name_prefix`（可选，例：`raw_weight_component_`）

**Response 200**
```json
{
  "rebalance_date": "2026-01-16",
  "signals": [
    {
      "instrument_id": "GROWTH",
      "signal_name": "T",
      "value": 0.8,
      "bucket_id": "GROWTH",
      "meta_json": null
    }
  ]
}
```

### GET `/signals/latest`

同 `/signals`，但自动选择最新 `rebalance_date`。

---

## 7. 错误码

```json
{
  "code": "SIGNAL_INCOMPLETE",
  "message": "missing tilt_weight signals",
  "details": null
}
```

| code | 含义 |
|---|---|
| `INPUT_COVERAGE_MISSING` | 输入覆盖不足 |
| `SIGNAL_INCOMPLETE` | 信号不完备 |
| `TILT_INVALID` | 倾斜层失败 |
| `WEIGHTS_SANITY_FAIL` | 权重校验失败 |
| `WEIGHTS_NOT_FOUND` | 权重不存在 |
| `RUN_NOT_FOUND` | run_id 不存在 |
| `DB_ERROR` | 数据库错误 |
