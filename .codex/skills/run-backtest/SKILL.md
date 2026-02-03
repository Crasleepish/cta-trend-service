---
name: run-backtest
description: 执行回测任务：读取历史 weights 与 NAV 数据，生成 weights-by-date CSV、NAV/returns CSV、equity curve HTML 和 summary report。支持 config/app.yaml 的 backtest.* 配置与 CLI 覆盖。
---

# 执行回测（Run Backtest）

## 目标
运行回测 runner，基于历史权重与 NAV 数据生成标准化输出：
- weights-by-date CSV
- NAV/returns CSV
- equity curve HTML
- summary report

## 输入（必需）
- start-date：回测开始日期（YYYY-MM-DD）
- end-date：回测结束日期（YYYY-MM-DD）
- strategy-id：策略标识（如 cta_trend_v1）
- version：版本号（如 1.0.0）
- portfolio-id：组合标识（如 main）
- output-dir：输出目录（如 docs/backtests）

## 配置来源
- 默认配置：`config/app.yaml` 下的 `backtest.*`
- CLI flags 可覆盖同名配置（以 CLI 为准）

## 执行步骤（标准）
1. 运行 FULL（生成 features/signals/weights）建议按年度分批跑：

  要点：
  - /jobs/full 会依次写入 feature_daily / feature_weekly_sample /
    signal_weekly / portfolio_weight_weekly
  - 回测只依赖 portfolio_weight_weekly，所以必须先跑 FULL（或至少
    Feature+Signal+Portfolio）

2. 确认权重已覆盖目标区间

  SELECT MIN(rebalance_date), MAX(rebalance_date), COUNT(*)
  FROM cta.portfolio_weight_weekly
  WHERE strategy_id='cta_trend_v1' AND version='1.0.0';
  
3. 跑回测

```bash
uv run python -m src.backtest.runner \
  --start-date <YYYY-MM-DD> \
  --end-date <YYYY-MM-DD> \
  --strategy-id <strategy_id> \
  --version <version> \
  --portfolio-id <portfolio_id> \
  --output-dir <output_dir>
```

4. 查看输出
