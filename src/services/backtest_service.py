from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestConfig, run_backtest
from ..repo.inputs import NavRepo, TradeCalendarRepo
from ..repo.outputs import WeightRepo


@dataclass(frozen=True)
class BacktestRequest:
    start_date: date
    end_date: date
    strategy_id: str
    version: str
    portfolio_id: str
    output_dir: Path
    buy_fee: float | None = None
    sell_fee: float | None = None
    slippage: float | None = None
    init_cash: float | None = None
    cash_sharing: bool | None = None
    freq: str | None = None


@dataclass(frozen=True)
class BacktestArtifacts:
    weights_csv: Path
    nav_returns_csv: Path
    equity_curve_html: Path
    report_md: Path


@dataclass(frozen=True)
class BacktestResult:
    artifacts: BacktestArtifacts
    stats: dict[str, float | None]
    warnings: list[str]


class BacktestService:
    def __init__(
        self,
        *,
        weight_repo: WeightRepo,
        nav_repo: NavRepo,
        calendar_repo: TradeCalendarRepo,
    ) -> None:
        self.weight_repo = weight_repo
        self.nav_repo = nav_repo
        self.calendar_repo = calendar_repo

    def run_backtest_task(self, req: BacktestRequest) -> BacktestResult:
        weights = self._load_weights(
            strategy_id=req.strategy_id,
            version=req.version,
            portfolio_id=req.portfolio_id,
            start_date=req.start_date,
            end_date=req.end_date,
        )
        if weights.empty:
            raise ValueError("no weights available for date range")

        close = self._load_close(
            assets=list(weights.columns),
            start_date=req.start_date,
            end_date=req.end_date,
        )

        warnings: list[str] = []
        weights, close, dropped = self._drop_missing_assets(weights, close)
        if dropped:
            warnings.append(f"dropped assets with missing NAV: {', '.join(dropped)}")

        cfg = BacktestConfig(
            buy_fee=req.buy_fee or BacktestConfig.buy_fee,
            sell_fee=req.sell_fee or BacktestConfig.sell_fee,
            slippage=req.slippage or BacktestConfig.slippage,
            init_cash=req.init_cash or BacktestConfig.init_cash,
            cash_sharing=(
                req.cash_sharing if req.cash_sharing is not None else BacktestConfig.cash_sharing
            ),
            freq=req.freq or BacktestConfig.freq,
        )

        result = run_backtest(weights=weights, close=close, cfg=cfg)
        stats = self._extract_stats(result["stats"], result["nav"], result["returns"])

        artifacts = self._write_outputs(
            req=req,
            weights=weights,
            nav=result["nav"],
            returns=result["returns"],
            stats=stats,
        )

        return BacktestResult(artifacts=artifacts, stats=stats, warnings=warnings)

    def _load_weights(
        self,
        *,
        strategy_id: str,
        version: str,
        portfolio_id: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        rows = self.weight_repo.get_history(
            strategy_id=strategy_id,
            version=version,
            portfolio_id=portfolio_id,
            start_date=start_date,
            end_date=end_date,
        )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["rebalance_date"] = pd.to_datetime(df["rebalance_date"])
        pivot = df.pivot_table(
            index="rebalance_date",
            columns="instrument_id",
            values="target_weight",
            aggfunc="first",
        ).sort_index()
        return pivot.astype(float)

    def _load_close(
        self,
        *,
        assets: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        calendar_rows = self.calendar_repo.get_range(start_date, end_date)
        if not calendar_rows:
            raise ValueError("trade_calendar is empty for date range")
        calendar = pd.DatetimeIndex([row["date"] for row in calendar_rows])

        nav_rows = self.nav_repo.get_range(assets, start_date, end_date)
        if not nav_rows:
            raise ValueError("fund_hist is empty for date range")
        nav_df = pd.DataFrame(nav_rows)
        nav_df["date"] = pd.to_datetime(nav_df["date"])
        pivot = nav_df.pivot_table(
            index="date", columns="fund_code", values="net_value", aggfunc="first"
        ).sort_index()
        pivot = pivot.reindex(calendar)
        return pivot.astype(float)

    @staticmethod
    def _drop_missing_assets(
        weights: pd.DataFrame, close: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        close = close.copy()
        close = close.ffill()
        missing_assets = [col for col in weights.columns if col not in close.columns] + [
            col for col in close.columns if close[col].isna().all()
        ]
        if missing_assets:
            close = close.drop(columns=missing_assets, errors="ignore")
            weights = weights.drop(columns=missing_assets, errors="ignore")
            weights = weights.fillna(0.0)
            row_sum = weights.sum(axis=1)
            weights = weights.div(row_sum.where(row_sum != 0, np.nan), axis=0).fillna(0.0)
        return weights, close, missing_assets

    def _write_outputs(
        self,
        *,
        req: BacktestRequest,
        weights: pd.DataFrame,
        nav: pd.Series,
        returns: pd.Series,
        stats: dict[str, float | None],
    ) -> BacktestArtifacts:
        output_dir = Path(req.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        date_tag = f"{req.start_date.isoformat()}_{req.end_date.isoformat()}"
        weights_csv = output_dir / f"weights_{date_tag}.csv"
        nav_returns_csv = output_dir / f"nav_returns_{date_tag}.csv"
        equity_curve_html = output_dir / f"equity_curve_{date_tag}.html"
        report_md = output_dir / f"report_{date_tag}.md"

        weights.to_csv(weights_csv, index_label="rebalance_date")

        nav_df = pd.DataFrame({"nav": nav, "returns": returns})
        nav_df.index.name = "date"
        nav_df.to_csv(nav_returns_csv)

        fig = nav.vbt.plot(title="Cumulative NAV")
        fig.write_html(str(equity_curve_html))

        report_md.write_text(
            self._build_report(req, stats, weights_csv, nav_returns_csv, equity_curve_html)
        )

        return BacktestArtifacts(
            weights_csv=weights_csv,
            nav_returns_csv=nav_returns_csv,
            equity_curve_html=equity_curve_html,
            report_md=report_md,
        )

    @staticmethod
    def _extract_stats(stats: Any, nav: pd.Series, returns: pd.Series) -> dict[str, float | None]:
        series = stats
        if isinstance(stats, pd.DataFrame):
            series = stats.squeeze()
        if isinstance(series, pd.Series):
            numeric = pd.to_numeric(series, errors="coerce")
            data = {str(k): (float(v) if pd.notna(v) else None) for k, v in numeric.items()}
        elif isinstance(series, dict):
            numeric = pd.to_numeric(pd.Series(series), errors="coerce")
            data = {str(k): (float(v) if pd.notna(v) else None) for k, v in numeric.items()}
        else:
            data = {}

        def pick(*keys: str) -> float | None:
            for key in keys:
                for k, v in data.items():
                    if k.lower() == key.lower():
                        return v
            return None

        max_dd = pick("Max Drawdown", "Max Drawdown [%]")
        annual_ret = pick("Annualized Return", "Annualized Return [%]", "Annual Return")
        annual_vol = pick(
            "Annualized Volatility",
            "Annualized Volatility [%]",
            "Annual Volatility",
        )
        max_recovery = pick("Max Drawdown Duration", "Max Drawdown Duration [days]")
        if max_recovery is None and isinstance(stats, (pd.Series, dict)):
            raw = (
                stats.get("Max Drawdown Duration")
                if isinstance(stats, dict)
                else stats.get("Max Drawdown Duration")
            )
            if isinstance(raw, pd.Timedelta):
                max_recovery = raw / pd.Timedelta(days=1)

        ulcer_index = pick("Ulcer Index")
        if ulcer_index is None:
            dd = nav / nav.cummax() - 1.0
            dd_pct = dd.abs() * 100.0
            ulcer_index = float(np.sqrt((dd_pct**2).mean()))

        return {
            "max_drawdown": max_dd,
            "sharpe_ratio": pick("Sharpe Ratio"),
            "annual_return": annual_ret,
            "calmar_ratio": pick("Calmar Ratio"),
            "max_recovery_time": max_recovery,
            "annual_volatility": annual_vol,
            "ulcer_index": ulcer_index,
        }

    @staticmethod
    def _build_report(
        req: BacktestRequest,
        stats: dict[str, float | None],
        weights_csv: Path,
        nav_returns_csv: Path,
        equity_curve_html: Path,
    ) -> str:
        lines = [
            "# Backtest Report",
            "",
            f"Date range: {req.start_date} -> {req.end_date}",
            f"Strategy: {req.strategy_id} / {req.version} / {req.portfolio_id}",
            "",
            "## Key Metrics",
            "| Metric | Value |",
            "| --- | --- |",
        ]
        for key, value in stats.items():
            lines.append(f"| {key} | {value} |")
        lines.extend(
            [
                "",
                "## Outputs",
                f"- Weights CSV: {weights_csv}",
                f"- NAV & Returns CSV: {nav_returns_csv}",
                f"- Equity Curve HTML: {equity_curve_html}",
            ]
        )
        return "\n".join(lines)
