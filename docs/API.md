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
### POST `/jobs/param-prepare`

触发一次同步 job。成功或失败都会返回 `run_id`。

`/jobs/param-prepare` 用于参数准备（AutoParam + DynamicParam）。支持传入 `as_of`（默认今天），用于回测场景按历史时点计算参数。所有参数计算窗口以 `as_of` 为结束日期。

**Header**
- `Idempotency-Key`（可选）：幂等键

**Request**
```json
{
  "as_of": "2026-01-16",
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

**Request (param-prepare only)**
```json
{
  "as_of": "2026-01-16"
}
```

**Response 200 (param-prepare)**
```json
{
  "auto_params_path": "config/auto_params.json",
  "dynamic_params_path": "config/dynamic_params.json",
  "auto_enabled": true,
  "auto_fallback": false,
  "dynamic_fallback": false,
  "warnings": []
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

---

## 7. Backtests

### POST `/backtests/run`

Run a backtest over a natural date range and write outputs to disk.

**Request**
```json
{
  "start_date": "2019-01-01",
  "end_date": "2019-06-30",
  "strategy_id": "cta_trend_v1",
  "version": "1.0.0",
  "portfolio_id": "main",
  "output_dir": "docs/backtests",
  "buy_fee": 0.0005,
  "sell_fee": 0.0005,
  "slippage": 0.0002,
  "init_cash": 1000000.0,
  "cash_sharing": true,
  "freq": "D"
}
```

**Response**
```json
{
  "weights_csv": "docs/backtests/weights_2019-01-01_2019-06-30.csv",
  "nav_returns_csv": "docs/backtests/nav_returns_2019-01-01_2019-06-30.csv",
  "equity_curve_html": "docs/backtests/equity_curve_2019-01-01_2019-06-30.html",
  "report_md": "docs/backtests/report_2019-01-01_2019-06-30.md",
  "warnings": [],
  "stats": {
    "max_drawdown": -0.21,
    "sharpe_ratio": 0.9,
    "annual_return": 0.12,
    "calmar_ratio": 0.6,
    "max_recovery_time": 120,
    "annual_volatility": 0.15,
    "ulcer_index": 0.08
  }
}
```

---

## 8. 错误码约定

- `INPUT_COVERAGE_MISSING`：输入表日期覆盖不足
- `SIGNAL_INCOMPLETE`：signal_weekly 缺必备 signal_name
- `TILT_INVALID`：倾斜层 tilt_weight 不完备或归一化失败
- `WEIGHTS_SANITY_FAIL`：权重和不为 1、出现 NaN/负权重等
- `RUN_NOT_FOUND` / `WEIGHTS_NOT_FOUND`
- `DB_ERROR`：数据库错误（尽量只在 500 返回）

## CLI

```bash
uv run python -m src.backtest.runner \
  --start-date 2019-01-01 \
  --end-date 2019-06-30 \
  --strategy-id cta_trend_v1 \
  --version 1.0.0 \
  --portfolio-id main \
  --output-dir docs/backtests
```
