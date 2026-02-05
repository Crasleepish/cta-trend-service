from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from ..core.config import AppConfig
from ..repo.inputs import (
    BucketRepo,
    FactorRepo,
    MarketRepo,
    TradeCalendarRepo,
)
from ..utils.trading_calendar import TradingCalendar, sample_week_last_trading_day

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoParamResult:
    window_start: date
    window_end: date
    params: dict[str, Any]
    warnings: list[str]
    used_fallback: bool


class AutoParamService:
    def __init__(
        self,
        *,
        bucket_repo: BucketRepo,
        market_repo: MarketRepo,
        factor_repo: FactorRepo,
        calendar_repo: TradeCalendarRepo,
        config: AppConfig,
        output_path: Path | None = None,
    ) -> None:
        self.bucket_repo = bucket_repo
        self.market_repo = market_repo
        self.factor_repo = factor_repo
        self.calendar_repo = calendar_repo
        self.config = config
        self.output_path = output_path or Path(self.config.auto_params.path)

    def compute_and_persist(self, *, as_of: date | None = None) -> AutoParamResult:
        end_date = as_of or date.today()
        window_days = int(self.config.auto_params.auto_param_window_size * 366)
        start_date = end_date - timedelta(days=window_days)

        calendar_rows = self.calendar_repo.get_range(start_date, end_date)
        if not calendar_rows:
            return self._fallback(start_date, end_date, ["trade_calendar empty"])

        calendar = TradingCalendar.from_dates(row["date"] for row in calendar_rows)
        trade_dates = list(calendar.dates)
        if len(trade_dates) < self.config.auto_params.min_points:
            return self._fallback(
                start_date,
                end_date,
                [f"trade_calendar points too few: {len(trade_dates)}"],
            )

        window_start = trade_dates[0].date()
        window_end = trade_dates[-1].date()
        warnings: list[str] = []

        buckets = self.bucket_repo.get_range()
        bucket_proxies = {bucket["bucket_name"]: bucket.get("bucket_proxy") for bucket in buckets}
        proxy_codes = [code for code in bucket_proxies.values() if code]
        if not proxy_codes:
            return self._fallback(window_start, window_end, ["no bucket proxies available"])

        market_rows = self.market_repo.get_range(proxy_codes, window_start, window_end)
        if not market_rows:
            return self._fallback(window_start, window_end, ["no index_hist data"])

        price_series = self._build_price_series(market_rows, bucket_proxies, calendar=calendar)

        trend_by_bucket = self._compute_weekly_trend(price_series, calendar, warnings)
        theta_rate = self._estimate_theta_rate(trend_by_bucket, warnings)
        sigma_target = self._estimate_sigma_target(price_series, warnings)

        path_quality = self._estimate_path_quality_params(price_series, calendar, warnings)
        tilt_lookback = self._estimate_tilt_lookback(window_start, window_end, warnings)

        params = {
            "features": {
                "theta_rate": theta_rate,
                "x0": path_quality["x0"],
                "path_quality_gamma": path_quality["path_quality_gamma"],
            },
            "portfolio": {
                "sigma_target": sigma_target,
            },
            "signals": {
                "tilt_lookback_days": tilt_lookback["tilt_lookback_days"],
            },
        }

        result = AutoParamResult(
            window_start=window_start,
            window_end=window_end,
            params=params,
            warnings=warnings,
            used_fallback=path_quality["fallback"] or tilt_lookback["fallback"],
        )
        self._persist(result)
        return result

    def apply_overrides(self, config: AppConfig, params: Mapping[str, Any]) -> None:
        features = params.get("features", {})
        signals = params.get("signals", {})
        portfolio = params.get("portfolio", {})

        for key in [
            "path_quality_gamma",
            "theta_rate",
            "x0",
        ]:
            if key in features:
                setattr(config.features, key, features[key])

        if "sigma_target" in portfolio:
            config.portfolio.sigma_target = float(portfolio["sigma_target"])

        if "tilt_lookback_days" in signals:
            config.signals.tilt_lookback_days = int(signals["tilt_lookback_days"])

    def _fallback(self, start: date, end: date, warnings: list[str]) -> AutoParamResult:
        params = {
            "features": {
                "path_quality_gamma": self.config.features.path_quality_gamma,
                "x0": self.config.features.x0,
                "theta_rate": self.config.features.theta_rate,
            },
            "portfolio": {"sigma_target": self.config.portfolio.sigma_target},
            "signals": {
                "tilt_lookback_days": self.config.signals.tilt_lookback_days,
            },
        }
        result = AutoParamResult(
            window_start=start,
            window_end=end,
            params=params,
            warnings=warnings,
            used_fallback=True,
        )
        self._persist(result)
        return result

    def _persist(self, result: AutoParamResult) -> None:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window": {
                "start": result.window_start.isoformat(),
                "end": result.window_end.isoformat(),
                "years": self.config.auto_params.auto_param_window_size,
            },
            "params": result.params,
            "warnings": result.warnings,
            "fallback": result.used_fallback,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    def _build_price_series(
        self,
        rows: Iterable[Mapping[str, Any]],
        bucket_proxies: Mapping[str, str | None],
        *,
        calendar: TradingCalendar,
    ) -> dict[str, pd.Series]:
        frames: dict[str, pd.Series] = {}
        for bucket, proxy in bucket_proxies.items():
            if not proxy:
                continue
            data = [row for row in rows if row["index_code"] == proxy]
            if not data:
                continue
            series = (
                pd.DataFrame(data)
                .assign(date=lambda df: pd.to_datetime(df["date"]))
                .set_index("date")["close"]
                .sort_index()
                .rename(bucket)
            )
            frames[bucket] = series.reindex(calendar.dates)
        return frames

    def _compute_weekly_trend(
        self,
        price_series: Mapping[str, pd.Series],
        calendar: TradingCalendar,
        warnings: list[str],
    ) -> dict[str, pd.Series]:
        short_window = self.config.features.short_window
        long_window = self.config.features.long_window
        vol_window = self.config.features.vol_window
        annualize = self.config.features.annualize
        weekly_dates = sample_week_last_trading_day(calendar.dates)

        trend_by_bucket: dict[str, pd.Series] = {}
        for bucket, series in price_series.items():
            if series.dropna().shape[0] < self.config.auto_params.min_points:
                warnings.append(f"{bucket}: insufficient price history")
                continue
            log_ret = np.log(series / series.shift(1))
            sigma = log_ret.rolling(vol_window).std(ddof=1) * np.sqrt(annualize)
            ma_short = series.rolling(short_window).mean()
            ma_long = series.rolling(long_window).mean()
            weekly_sigma = sigma.loc[weekly_dates].dropna()
            weekly_t = (ma_short - ma_long).loc[weekly_dates] / weekly_sigma
            weekly_t = weekly_t.dropna()
            if weekly_t.empty or weekly_sigma.empty:
                warnings.append(f"{bucket}: empty weekly stats")
                continue
            trend_by_bucket[bucket] = weekly_t
        return trend_by_bucket

    def _estimate_theta_rate(
        self, trend_by_bucket: Mapping[str, pd.Series], warnings: list[str]
    ) -> float:
        if "RATE" not in trend_by_bucket:
            warnings.append("RATE trend missing; theta_rate fallback")
            return float(self.config.features.theta_rate)
        series = trend_by_bucket["RATE"].dropna()
        if series.empty:
            warnings.append("RATE trend empty; theta_rate fallback")
            return float(self.config.features.theta_rate)
        return float(series.median())

    def _estimate_sigma_target(
        self, price_series: Mapping[str, pd.Series], warnings: list[str]
    ) -> float:
        risk_buckets = self.config.portfolio.risk_buckets
        series = [price_series[b] for b in risk_buckets if b in price_series]
        if not series:
            warnings.append("risk bucket price series missing; sigma_target fallback")
            return float(self.config.portfolio.sigma_target)
        prices = pd.concat(series, axis=1).dropna(how="all")
        returns = np.log(prices / prices.shift(1)).dropna(how="all")
        if returns.empty:
            warnings.append("risk bucket returns empty; sigma_target fallback")
            return float(self.config.portfolio.sigma_target)
        risk_ret = returns.mean(axis=1)
        annualize = self.config.features.annualize
        if risk_ret.shape[0] < 252:
            warnings.append("risk bucket returns too short; sigma_target fallback")
            return float(self.config.portfolio.sigma_target)
        rolling = risk_ret.rolling(252).std(ddof=1) * np.sqrt(annualize)
        rolling = rolling.dropna()
        if rolling.empty:
            warnings.append("risk bucket rolling vol empty; sigma_target fallback")
            return float(self.config.portfolio.sigma_target)
        return float(rolling.median())

    def _estimate_tilt_lookback(
        self,
        start_date: date,
        end_date: date,
        warnings: list[str],
    ) -> dict[str, Any]:
        fallback = False
        tilt_factors = self.config.signals.tilt_factors
        factor_rows = self.factor_repo.get_range(start_date, end_date)
        factor_df = pd.DataFrame(factor_rows)
        tilt_lookback = self.config.signals.tilt_lookback_days
        if factor_df.empty:
            warnings.append("tilt factors empty; using config tilt_lookback_days")
            fallback = True
        else:
            factor_df = factor_df.set_index("date")[tilt_factors].astype(float)
            rhos = []
            for factor in tilt_factors:
                series = factor_df[factor].dropna()
                if series.shape[0] < 5:
                    continue
                rho = series.autocorr(lag=1)
                if rho is None or rho <= 0 or rho >= 0.99:
                    continue
                half_life = -np.log(2) / np.log(rho)
                if np.isfinite(half_life):
                    rhos.append(half_life)
            if rhos:
                tilt_lookback = int(np.clip(int(round(np.median(rhos))), 20, 120))
            else:
                warnings.append("tilt autocorr invalid; using config tilt_lookback_days")
                fallback = True

        return {
            "tilt_lookback_days": tilt_lookback,
            "fallback": fallback,
        }

    def _estimate_path_quality_params(
        self,
        price_series: Mapping[str, pd.Series],
        calendar: TradingCalendar,
        warnings: list[str],
    ) -> dict[str, Any]:
        fallback = False
        x0_quantile = 0.6
        gamma_quantile = 0.8
        gamma_target = 0.6
        x0_default_raw = self.config.features.x0
        if isinstance(x0_default_raw, dict):
            if "__default__" in x0_default_raw:
                x0_default_raw = x0_default_raw["__default__"]
            else:
                x0_default_raw = next(iter(x0_default_raw.values()))
        x0_default = float(x0_default_raw)

        gamma_default_raw = self.config.features.path_quality_gamma
        if isinstance(gamma_default_raw, dict):
            if "__default__" in gamma_default_raw:
                gamma_default_raw = gamma_default_raw["__default__"]
            else:
                gamma_default_raw = next(iter(gamma_default_raw.values()))
        gamma_default = float(gamma_default_raw)

        path_quality_window = int(self.config.features.path_quality_window_days)
        weekly_dates = sample_week_last_trading_day(calendar.dates)

        z_weekly_by_bucket: dict[str, pd.Series] = {}
        for bucket, series in price_series.items():
            roll_min = series.rolling(path_quality_window, min_periods=path_quality_window).min()
            roll_max = series.rolling(path_quality_window, min_periods=path_quality_window).max()
            runup = np.log(series / roll_min)
            drawdown = np.log(roll_max / series)
            denom = runup + drawdown
            z_daily = runup / denom
            z_daily = z_daily.where(denom != 0, 0.5)
            z_weekly = z_daily.loc[weekly_dates].dropna()
            if not z_weekly.empty:
                z_weekly_by_bucket[bucket] = z_weekly

        if not z_weekly_by_bucket:
            warnings.append("path_quality_z empty; fallback to config x0/gamma")
            return {"x0": x0_default, "path_quality_gamma": gamma_default, "fallback": True}

        x0_values: list[float] = []
        gamma_values: list[float] = []

        for bucket, z_weekly in z_weekly_by_bucket.items():
            z_weekly = z_weekly.dropna()
            if z_weekly.empty:
                continue
            x0_val = float(z_weekly.quantile(x0_quantile))
            x0_values.append(x0_val)
            above = z_weekly[z_weekly > x0_val]
            if above.empty:
                continue
            zq = float(above.quantile(gamma_quantile))
            ratio = (zq - x0_val) / (1 - x0_val) if x0_val < 1 else 0.0
            if ratio <= 0 or ratio >= 1:
                continue
            gamma_val = float(np.log(gamma_target) / np.log(ratio))
            if np.isfinite(gamma_val) and gamma_val > 0:
                gamma_values.append(gamma_val)

        if not x0_values:
            warnings.append("path_quality_z all NaN; fallback to config x0/gamma")
            return {"x0": x0_default, "path_quality_gamma": gamma_default, "fallback": True}

        if len(x0_values) > 2:
            x0_agg = float(pd.Series(x0_values).median())
        else:
            x0_agg = float(np.mean(x0_values))

        if gamma_values:
            if len(x0_values) > 2:
                gamma_agg = float(pd.Series(gamma_values).median())
            else:
                gamma_agg = float(np.mean(gamma_values))
        else:
            warnings.append("path_quality gamma invalid; fallback")
            fallback = True
            gamma_agg = gamma_default

        if np.isfinite(gamma_agg):
            gamma_agg = float(np.clip(gamma_agg, 1.0, 3.0))
        if not np.isfinite(gamma_agg) or gamma_agg <= 0:
            warnings.append("path_quality gamma invalid; fallback")
            fallback = True
            gamma_agg = gamma_default

        return {"x0": x0_agg, "path_quality_gamma": gamma_agg, "fallback": fallback}
