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
from ..repo.inputs import BetaRepo, BucketRepo, MarketRepo, TradeCalendarRepo
from ..utils.trading_calendar import TradingCalendar

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DynamicParamResult:
    window_start: date
    window_end: date
    params: dict[str, Any]
    warnings: list[str]
    used_fallback: bool


class DynamicParamService:
    def __init__(
        self,
        *,
        bucket_repo: BucketRepo,
        market_repo: MarketRepo,
        beta_repo: BetaRepo,
        calendar_repo: TradeCalendarRepo,
        config: AppConfig,
        output_path: Path | None = None,
    ) -> None:
        self.bucket_repo = bucket_repo
        self.market_repo = market_repo
        self.beta_repo = beta_repo
        self.calendar_repo = calendar_repo
        self.config = config
        self.output_path = output_path or Path(self.config.dynamic_params.path)

    def compute_and_persist(self, *, as_of: date | None = None) -> DynamicParamResult:
        end_date = as_of or date.today()
        window_days = int(self.config.auto_params.auto_param_window_size * 366)
        start_date = end_date - timedelta(days=window_days)

        calendar_rows = self.calendar_repo.get_range(start_date, end_date)
        if not calendar_rows:
            raise ValueError("dynamic params: trade_calendar empty")

        calendar = TradingCalendar.from_dates(row["date"] for row in calendar_rows)
        trade_dates = list(calendar.dates)
        if len(trade_dates) < self.config.auto_params.min_points:
            raise ValueError(
                f"dynamic params: trade_calendar points too few: {len(trade_dates)}"
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
            raise ValueError("dynamic params: no bucket proxies available")

        market_rows = self.market_repo.get_range(proxy_codes, window_start, window_end)
        if not market_rows:
            raise ValueError("dynamic params: no index_hist data")

        price_series = self._build_price_series(market_rows, bucket_proxies, calendar=calendar)
        bucket_params = self._estimate_bucket_params(price_series, calendar, warnings)

        tilt_scales = self._estimate_tilt_scales(window_start, window_end, asset_codes, warnings)

        params = {
            "features": {
                "theta_on": bucket_params["theta_on"],
                "theta_off": bucket_params["theta_off"],
                "theta_minus": bucket_params["theta_minus"],
                "sigma_min": bucket_params["sigma_min"],
                "sigma_max": bucket_params["sigma_max"],
                "kappa_sigma": bucket_params["kappa_sigma"],
            },
            "signals": {
                "tilt_scales": tilt_scales,
            },
        }

        result = DynamicParamResult(
            window_start=window_start,
            window_end=window_end,
            params=params,
            warnings=warnings,
            used_fallback=False,
        )
        if bucket_params["fallback"]:
            raise ValueError(
                "dynamic params: insufficient data for bucket stats; see warnings"
            )
        self._persist(result)
        return result

    def apply_overrides(self, config: AppConfig, params: Mapping[str, Any]) -> None:
        features = params.get("features", {})
        signals = params.get("signals", {})

        for key in [
            "theta_on",
            "theta_off",
            "theta_minus",
            "sigma_min",
            "sigma_max",
            "kappa_sigma",
        ]:
            if key in features:
                setattr(config.features, key, features[key])

        if "tilt_scales" in signals:
            config.signals.tilt_scales = dict(signals["tilt_scales"])

    def _persist(self, result: DynamicParamResult) -> None:
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

    def _estimate_bucket_params(
        self,
        price_series: Mapping[str, pd.Series],
        calendar: TradingCalendar,
        warnings: list[str],
    ) -> dict[str, Any]:
        q_on = self.config.dynamic_params.q_on
        q_off = self.config.dynamic_params.q_off
        q_minus = self.config.dynamic_params.q_minus
        q_min = self.config.dynamic_params.q_min
        q_max = self.config.dynamic_params.q_max
        q_hi = self.config.dynamic_params.q_hi
        eps = self.config.dynamic_params.eps
        c_min = self.config.dynamic_params.c_min

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

        fallback = False
        for bucket, series in price_series.items():
            if series.dropna().shape[0] < self.config.auto_params.min_points:
                warnings.append(f"{bucket}: insufficient price history")
                fallback = True
                continue

            log_ret = np.log(series / series.shift(1))
            sigma = log_ret.rolling(vol_window).std(ddof=1) * np.sqrt(annualize)
            ma_short = series.rolling(short_window).mean()
            ma_long = series.rolling(long_window).mean()

            daily_sigma = sigma.dropna()
            daily_t = (ma_short - ma_long) / sigma
            daily_t = daily_t.dropna()
            if daily_t.empty or daily_sigma.empty:
                warnings.append(f"{bucket}: empty daily stats")
                fallback = True
                continue

            positive = daily_t[daily_t > 0]
            if positive.empty:
                warnings.append(f"{bucket}: no positive trend samples")
                fallback = True
            else:
                theta_on[bucket] = float(positive.quantile(q_on))
                theta_off[bucket] = float(positive.quantile(q_off))
                if theta_off[bucket] <= 0 or theta_on[bucket] <= 0:
                    warnings.append(f"{bucket}: non-positive theta_on/off")
                    fallback = True

            negative = daily_t[daily_t < 0].abs()
            if negative.empty:
                warnings.append(f"{bucket}: no negative trend samples")
                fallback = True
            else:
                theta_minus[bucket] = float(negative.quantile(q_minus))

            sigma_vals = daily_sigma.dropna()
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

        return {
            "theta_on": theta_on,
            "theta_off": theta_off,
            "theta_minus": theta_minus,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "kappa_sigma": kappa_sigma,
            "warnings": warnings,
            "fallback": fallback,
        }

    def _estimate_tilt_scales(
        self,
        start_date: date,
        end_date: date,
        asset_codes: list[str],
        warnings: list[str],
    ) -> dict[str, float]:
        tilt_factors = self.config.signals.tilt_factors
        beta_rows = []
        if asset_codes:
            beta_rows = self.beta_repo.get_range(asset_codes, start_date, end_date)
        beta_df = pd.DataFrame(beta_rows)
        if beta_df.empty:
            warnings.append("fund_beta empty; using config tilt_scales")
            return dict(self.config.signals.tilt_scales)

        beta_df = beta_df.sort_values("date").groupby("code").tail(1)
        iqr = beta_df[tilt_factors].quantile(0.75) - beta_df[tilt_factors].quantile(0.25)
        return {k: float(v) if v > 0 else 1.0 for k, v in iqr.items()}
