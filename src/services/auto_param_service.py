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
    BetaRepo,
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
        beta_repo: BetaRepo,
        calendar_repo: TradeCalendarRepo,
        config: AppConfig,
        output_path: Path | None = None,
    ) -> None:
        self.bucket_repo = bucket_repo
        self.market_repo = market_repo
        self.factor_repo = factor_repo
        self.beta_repo = beta_repo
        self.calendar_repo = calendar_repo
        self.config = config
        self.output_path = output_path or Path("config/auto_params.json")

    def compute_and_persist(self, *, as_of: date | None = None) -> AutoParamResult:
        end_date = as_of or date.today()
        window_days = int(self.config.auto_params.window_years * 366)
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
        asset_codes = [
            code.strip()
            for bucket in buckets
            for code in (bucket.get("assets") or "").split(",")
            if code.strip()
        ]
        proxy_codes = [code for code in bucket_proxies.values() if code]
        if not proxy_codes:
            return self._fallback(window_start, window_end, ["no bucket proxies available"])

        market_rows = self.market_repo.get_range(proxy_codes, window_start, window_end)
        if not market_rows:
            return self._fallback(window_start, window_end, ["no index_hist data"])

        price_series = self._build_price_series(market_rows, bucket_proxies, calendar=calendar)

        bucket_params = self._estimate_bucket_params(price_series, calendar)
        if bucket_params["fallback"]:
            warnings.extend(bucket_params["warnings"])

        theta_rate = self._estimate_theta_rate(bucket_params["trend"], warnings)
        sigma_target = self._estimate_sigma_target(price_series, warnings)

        tilt_params = self._estimate_tilt_params(window_start, window_end, asset_codes, warnings)

        params = {
            "features": {
                "theta_on": bucket_params["theta_on"],
                "theta_off": bucket_params["theta_off"],
                "theta_minus": bucket_params["theta_minus"],
                "path_quality_gamma": bucket_params["path_quality_gamma"],
                "sigma_min": bucket_params["sigma_min"],
                "sigma_max": bucket_params["sigma_max"],
                "kappa_sigma": bucket_params["kappa_sigma"],
                "x0": bucket_params["x0"],
                "theta_rate": theta_rate,
            },
            "portfolio": {
                "sigma_target": sigma_target,
            },
            "signals": {
                "tilt_lookback_days": tilt_params["tilt_lookback_days"],
                "tilt_scales": tilt_params["tilt_scales"],
                "tilt_eps": tilt_params["tilt_eps"],
                "tilt_temperature": tilt_params["tilt_temperature"],
            },
        }

        result = AutoParamResult(
            window_start=window_start,
            window_end=window_end,
            params=params,
            warnings=warnings,
            used_fallback=bucket_params["fallback"] or tilt_params["fallback"],
        )
        self._persist(result)
        return result

    def apply_overrides(self, config: AppConfig, params: Mapping[str, Any]) -> None:
        features = params.get("features", {})
        signals = params.get("signals", {})
        portfolio = params.get("portfolio", {})

        for key in [
            "theta_on",
            "theta_off",
            "theta_minus",
            "path_quality_gamma",
            "sigma_min",
            "sigma_max",
            "kappa_sigma",
            "theta_rate",
            "x0",
        ]:
            if key in features:
                setattr(config.features, key, features[key])

        if "sigma_target" in portfolio:
            config.portfolio.sigma_target = float(portfolio["sigma_target"])

        if "tilt_lookback_days" in signals:
            config.signals.tilt_lookback_days = int(signals["tilt_lookback_days"])
        if "tilt_scales" in signals:
            config.signals.tilt_scales = dict(signals["tilt_scales"])
        if "tilt_eps" in signals:
            config.signals.tilt_eps = float(signals["tilt_eps"])
        if "tilt_temperature" in signals:
            config.signals.tilt_temperature = float(signals["tilt_temperature"])

    def _fallback(self, start: date, end: date, warnings: list[str]) -> AutoParamResult:
        params = {
            "features": {
                "theta_on": self.config.features.theta_on,
                "theta_off": self.config.features.theta_off,
                "theta_minus": self.config.features.theta_minus,
                "path_quality_gamma": self.config.features.path_quality_gamma,
                "sigma_min": self.config.features.sigma_min,
                "sigma_max": self.config.features.sigma_max,
                "kappa_sigma": self.config.features.kappa_sigma,
                "x0": self.config.features.x0,
                "theta_rate": self.config.features.theta_rate,
            },
            "portfolio": {"sigma_target": self.config.portfolio.sigma_target},
            "signals": {
                "tilt_lookback_days": self.config.signals.tilt_lookback_days,
                "tilt_scales": self.config.signals.tilt_scales,
                "tilt_eps": self.config.signals.tilt_eps,
                "tilt_temperature": self.config.signals.tilt_temperature,
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
                "years": self.config.auto_params.window_years,
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

    def _estimate_bucket_params(
        self,
        price_series: Mapping[str, pd.Series],
        calendar: TradingCalendar,
    ) -> dict[str, Any]:
        q_on = 0.7
        q_off = 0.3
        q_minus = 0.7
        q_min = 0.1
        q_max = 0.8
        q_hi = 0.95
        eps = 0.1
        c_min = 0.5

        short_window = self.config.features.short_window
        long_window = self.config.features.long_window
        vol_window = self.config.features.vol_window
        annualize = self.config.features.annualize

        theta_on: dict[str, float] = {}
        theta_off: dict[str, float] = {}
        theta_minus: dict[str, float] = {}
        sigma_min: dict[str, float] = {}
        sigma_max: dict[str, float] = {}
        kappa_sigma: dict[str, float] = {}
        x0: dict[str, float] = {}
        path_quality_gamma: dict[str, float] = {}

        gamma_default = self.config.features.path_quality_gamma
        if isinstance(gamma_default, dict):
            if "__default__" in gamma_default:
                gamma_default = gamma_default["__default__"]
            else:
                gamma_default = next(iter(gamma_default.values()))
        gamma_default = float(gamma_default)

        trend_by_bucket: dict[str, pd.Series] = {}
        warnings: list[str] = []
        fallback = False

        weekly_dates = sample_week_last_trading_day(calendar.dates)
        path_quality_window = int(self.config.features.path_quality_window_days)
        x0_quantile = 0.65
        gamma_quantile = 0.9
        gamma_target = 0.6

        for bucket, series in price_series.items():
            if series.dropna().shape[0] < self.config.auto_params.min_points:
                warnings.append(f"{bucket}: insufficient price history")
                fallback = True
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
                fallback = True
                continue

            trend_by_bucket[bucket] = weekly_t

            positive = weekly_t[weekly_t > 0]
            if positive.empty:
                warnings.append(f"{bucket}: no positive trend samples")
                fallback = True
            else:
                theta_on[bucket] = float(positive.quantile(q_on))
                theta_off[bucket] = float(positive.quantile(q_off))
                if theta_off[bucket] <= 0 or theta_on[bucket] <= 0:
                    warnings.append(f"{bucket}: non-positive theta_on/off")
                    fallback = True

            negative = weekly_t[weekly_t < 0].abs()
            if negative.empty:
                warnings.append(f"{bucket}: no negative trend samples")
                fallback = True
            else:
                theta_minus[bucket] = float(negative.quantile(q_minus))

            sigma_vals = weekly_sigma.dropna()
            sigma_min_val = max(
                float(sigma_vals.quantile(q_min)),
                float(c_min * sigma_vals.median()),
            )
            sigma_min[bucket] = sigma_min_val
            sigma_max_val = float(sigma_vals.quantile(q_max))
            sigma_max[bucket] = sigma_max_val
            sigma_hi = float(sigma_vals.quantile(q_hi))
            if sigma_hi > sigma_max_val:
                kappa = (sigma_hi - sigma_max_val) / np.log(1 / eps - 1)
                if kappa > 0:
                    kappa_sigma[bucket] = float(kappa)
                else:
                    warnings.append(f"{bucket}: invalid kappa_sigma")
                    fallback = True
            else:
                warnings.append(f"{bucket}: sigma_hi <= sigma_max")
                fallback = True

            roll_min = series.rolling(path_quality_window, min_periods=path_quality_window).min()
            roll_max = series.rolling(path_quality_window, min_periods=path_quality_window).max()
            runup = np.log(series / roll_min)
            drawdown = np.log(roll_max / series)
            denom = runup + drawdown
            z_daily = runup / denom
            z_daily = z_daily.where(denom != 0, 0.5)
            z_weekly = z_daily.loc[weekly_dates].dropna()
            if z_weekly.empty:
                warnings.append(f"{bucket}: empty path_quality_z series")
                fallback = True
                continue
            x0_val = float(z_weekly.quantile(x0_quantile))
            x0[bucket] = x0_val

            above = z_weekly[z_weekly > x0_val]
            if above.empty:
                warnings.append(f"{bucket}: no samples above x0 for gamma")
                fallback = True
                path_quality_gamma[bucket] = gamma_default
            else:
                zq = float(above.quantile(gamma_quantile))
                ratio = (zq - x0_val) / (1 - x0_val) if x0_val < 1 else 0.0
                if ratio <= 0 or ratio >= 1:
                    warnings.append(f"{bucket}: invalid gamma ratio; fallback gamma")
                    fallback = True
                    path_quality_gamma[bucket] = gamma_default
                else:
                    gamma_val = float(np.log(gamma_target) / np.log(ratio))
                    if not np.isfinite(gamma_val) or gamma_val <= 0:
                        warnings.append(f"{bucket}: non-positive gamma; fallback")
                        fallback = True
                        path_quality_gamma[bucket] = gamma_default
                    else:
                        path_quality_gamma[bucket] = gamma_val

        return {
            "theta_on": theta_on,
            "theta_off": theta_off,
            "theta_minus": theta_minus,
            "path_quality_gamma": path_quality_gamma,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "kappa_sigma": kappa_sigma,
            "x0": x0,
            "trend": trend_by_bucket,
            "warnings": warnings,
            "fallback": fallback,
        }

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

    def _estimate_tilt_params(
        self,
        start_date: date,
        end_date: date,
        asset_codes: list[str],
        warnings: list[str],
    ) -> dict[str, Any]:
        fallback = False
        tilt_factors = self.config.signals.tilt_factors
        factor_rows = self.factor_repo.get_range(start_date, end_date)
        factor_df = pd.DataFrame(factor_rows)
        tilt_lookback = self.config.signals.tilt_lookback_days
        if factor_df.empty:
            warnings.append("tilt factors empty; using config tilt params")
            fallback = True
            tilt_scales = dict(self.config.signals.tilt_scales)
            tilt_eps = float(self.config.signals.tilt_eps)
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

            beta_rows = []
            if asset_codes:
                beta_rows = self.beta_repo.get_range(asset_codes, start_date, end_date)
            beta_df = pd.DataFrame(beta_rows)
            if beta_df.empty:
                warnings.append("fund_beta empty; using config tilt_scales")
                tilt_scales = dict(self.config.signals.tilt_scales)
                fallback = True
            else:
                beta_df = beta_df.sort_values("date").groupby("code").tail(1)
                iqr = beta_df[tilt_factors].quantile(0.75) - beta_df[tilt_factors].quantile(0.25)
                tilt_scales = {k: float(v) if v > 0 else 1.0 for k, v in iqr.items()}

            tilt_eps = float(self.config.signals.tilt_eps)

        return {
            "tilt_lookback_days": tilt_lookback,
            "tilt_scales": tilt_scales,
            "tilt_eps": tilt_eps,
            "tilt_temperature": float(self.config.signals.tilt_temperature),
            "fallback": fallback,
        }
