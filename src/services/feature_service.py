from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Mapping, cast

import pandas as pd

from ..core.config import FeatureConfig
from ..features import computer, sampler
from ..features.registry import FeatureRegistry, FeatureSpec
from ..repo.inputs import BucketRepo, MarketRepo, TradeCalendarRepo
from ..repo.outputs import FeatureRepo, FeatureRow, FeatureWeeklySampleRepo, FeatureWeeklySampleRow
from ..utils.trading_calendar import TradingCalendar

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSetSpec:
    enabled_features: list[str]
    feature_params: dict[str, float]


@dataclass(frozen=True)
class FeatureRunSummary:
    calc_date_range: tuple[date, date]
    instruments_count: int
    features_generated: list[str]
    rows_upserted_daily: int
    rows_upserted_weekly: int
    warnings: list[str]
    coverage: dict[str, dict[str, Any]]


@dataclass
class FeatureService:
    bucket_repo: BucketRepo
    market_repo: MarketRepo
    calendar_repo: TradeCalendarRepo
    feature_repo: FeatureRepo
    feature_weekly_repo: FeatureWeeklySampleRepo
    config: FeatureConfig

    def registry(self) -> FeatureRegistry:
        specs = [
            FeatureSpec("r_log_daily", "daily", ("prices",), lambda *_: pd.DataFrame()),
            FeatureSpec("sigma_ann", "daily", ("prices",), lambda *_: pd.DataFrame()),
            FeatureSpec("T", "daily", ("prices",), lambda *_: pd.DataFrame()),
            FeatureSpec("sigma_eff", "weekly", ("prices",), lambda *_: pd.DataFrame()),
            FeatureSpec("f_sigma", "weekly", ("prices",), lambda *_: pd.DataFrame()),
            FeatureSpec("gate_state", "weekly", ("prices",), lambda *_: pd.DataFrame()),
            FeatureSpec("down_drift", "weekly", ("prices",), lambda *_: pd.DataFrame()),
            FeatureSpec("T_RATE", "weekly", ("prices",), lambda *_: pd.DataFrame()),
            FeatureSpec("rate_pref", "weekly", ("prices",), lambda *_: pd.DataFrame()),
        ]
        return FeatureRegistry(specs)

    def compute(
        self,
        calc_start: date,
        calc_end: date,
        universe: dict[str, Any],
        snapshot_id: str | None,
        dry_run: bool,
    ) -> list[FeatureRow]:
        feature_set = FeatureSetSpec(enabled_features=[], feature_params={})
        summary, daily_rows, _weekly_rows = self._compute_features(
            run_id="",
            strategy_id="",
            version="",
            snapshot_id=snapshot_id,
            rebalance_date=calc_end,
            calc_start=calc_start,
            calc_end=calc_end,
            universe=universe,
            feature_set=feature_set,
        )
        _ = summary
        return daily_rows

    def compute_and_persist(
        self,
        *,
        run_id: str,
        strategy_id: str,
        version: str,
        snapshot_id: str | None,
        rebalance_date: date,
        calc_start: date,
        calc_end: date,
        universe: dict[str, Any],
        feature_set: FeatureSetSpec | None,
        dry_run: bool,
        force_recompute: bool,
    ) -> FeatureRunSummary:
        t0 = time.perf_counter()
        summary, daily_rows, weekly_rows = self._compute_features(
            run_id=run_id,
            strategy_id=strategy_id,
            version=version,
            snapshot_id=snapshot_id,
            rebalance_date=rebalance_date,
            calc_start=calc_start,
            calc_end=calc_end,
            universe=universe,
            feature_set=feature_set or FeatureSetSpec([], {}),
        )
        logger.info(
            "feature.compute run_id=%s rebalance_date=%s elapsed=%.3fs "
            "rows_daily=%s rows_weekly=%s",
            run_id,
            rebalance_date,
            time.perf_counter() - t0,
            len(daily_rows),
            len(weekly_rows),
        )

        if dry_run:
            return summary

        t1 = time.perf_counter()
        daily_count = self.feature_repo.upsert_many(daily_rows) if daily_rows else 0
        logger.info(
            "feature.upsert_daily run_id=%s rebalance_date=%s elapsed=%.3fs rows=%s",
            run_id,
            rebalance_date,
            time.perf_counter() - t1,
            daily_count,
        )
        t2 = time.perf_counter()
        weekly_count = self.feature_weekly_repo.upsert_many(weekly_rows) if weekly_rows else 0
        logger.info(
            "feature.upsert_weekly run_id=%s rebalance_date=%s elapsed=%.3fs rows=%s",
            run_id,
            rebalance_date,
            time.perf_counter() - t2,
            weekly_count,
        )
        return FeatureRunSummary(
            calc_date_range=summary.calc_date_range,
            instruments_count=summary.instruments_count,
            features_generated=summary.features_generated,
            rows_upserted_daily=daily_count,
            rows_upserted_weekly=weekly_count,
            warnings=summary.warnings,
            coverage=summary.coverage,
        )

    def _compute_features(
        self,
        *,
        run_id: str,
        strategy_id: str,
        version: str,
        snapshot_id: str | None,
        rebalance_date: date,
        calc_start: date,
        calc_end: date,
        universe: dict[str, Any],
        feature_set: FeatureSetSpec,
    ) -> tuple[FeatureRunSummary, list[FeatureRow], list[FeatureWeeklySampleRow]]:
        registry = self.registry()
        enabled = feature_set.enabled_features or self.config.enabled_features
        unknown = [name for name in enabled if name not in registry.list_features()]
        if unknown:
            raise ValueError(f"unknown features: {', '.join(unknown)}")

        t0 = time.perf_counter()
        buckets = self.bucket_repo.get_range(
            [int(b) for b in universe.get("bucket_ids", [])] or None
        )
        logger.info(
            "feature.io bucket_repo.get_range run_id=%s rebalance_date=%s elapsed=%.3fs buckets=%s",
            run_id,
            rebalance_date,
            time.perf_counter() - t0,
            len(buckets),
        )
        if not buckets:
            raise ValueError("bucket universe is empty")

        bucket_proxies: dict[str, str | None] = {
            bucket["bucket_name"]: bucket.get("bucket_proxy") for bucket in buckets
        }
        warnings: list[str] = []
        proxy_codes = [code for code in bucket_proxies.values() if code]
        if not proxy_codes:
            raise ValueError("no bucket proxies available")

        t1 = time.perf_counter()
        rows = self.market_repo.get_range(proxy_codes, calc_start, calc_end)
        logger.info(
            "feature.io market_repo.get_range run_id=%s rebalance_date=%s elapsed=%.3fs rows=%s",
            run_id,
            rebalance_date,
            time.perf_counter() - t1,
            len(rows),
        )
        if not rows:
            raise ValueError("no proxy price data available")

        calendar_rows = self.calendar_repo.get_range(calc_start, calc_end)
        if not calendar_rows:
            raise ValueError("trade_calendar is empty for calc range")
        calendar = TradingCalendar.from_dates(row["date"] for row in calendar_rows)

        price_series = self._build_price_series(rows, bucket_proxies, calendar=calendar)
        prices = pd.concat(price_series.values(), axis=1)
        coverage = self._coverage(price_series, calc_start, calc_end)

        params = self._params_dict()
        params.update(feature_set.feature_params)
        params_hash = computer.feature_params_hash(params)

        short_window = int(self._param_number(params["short_window"]))
        long_window = int(self._param_number(params["long_window"]))
        vol_window = int(self._param_number(params["vol_window"]))
        annualize = int(self._param_number(params["annualize"]))
        theta_on = params["theta_on"]
        theta_off = params["theta_off"]
        theta_minus = float(self._param_number(params["theta_minus"]))
        sigma_min = params["sigma_min"]
        sigma_max = params["sigma_max"]
        kappa_sigma = params["kappa_sigma"]
        rate_k = float(self._param_number(params["rate_k"]))
        theta_rate = float(self._param_number(params["theta_rate"]))

        returns_map = {
            name: computer.log_returns(series.to_frame(name))[name]
            for name, series in price_series.items()
        }
        sigma_map = {
            name: computer.sigma_annualized(
                returns_map[name].to_frame(name),
                window=vol_window,
                annualize=annualize,
            )[name]
            for name in returns_map
        }
        trend_map = {
            name: computer.trend_strength(
                price_series[name].to_frame(name),
                sigma_map[name].to_frame(name),
                short_window=short_window,
                long_window=long_window,
            )[name]
            for name in price_series
        }
        returns = pd.concat(returns_map, axis=1)
        sigma_ann = pd.concat(sigma_map, axis=1)
        trend = pd.concat(trend_map, axis=1)

        daily_frames: dict[str, pd.DataFrame] = {
            "r_log_daily": returns,
            "sigma_ann": sigma_ann,
            "T": trend,
        }

        weekly_history = sampler.weekly_history(
            trend, calendar=calendar.dates, rebalance_date=rebalance_date
        )
        weekly_sigma = sampler.weekly_history(
            sigma_ann, calendar=calendar.dates, rebalance_date=rebalance_date
        )

        gate_state = self._gate_state_by_bucket(
            weekly_history,
            theta_on=theta_on,
            theta_off=theta_off,
        )
        down_drift = computer.down_drift(weekly_history, theta_minus=theta_minus)
        sigma_eff = self._sigma_eff_by_bucket(weekly_sigma, sigma_min=sigma_min)
        f_sigma = self._f_sigma_by_bucket(
            weekly_sigma, sigma_max=sigma_max, kappa_sigma=kappa_sigma
        )

        weekly_frames: dict[str, pd.DataFrame] = {
            "T": weekly_history,
            "sigma_eff": sigma_eff,
            "f_sigma": f_sigma,
            "gate_state": gate_state,
            "down_drift": down_drift,
        }

        if "RATE" in weekly_history.columns:
            t_rate_series = weekly_history["RATE"]
            rate_pref_series = computer.rate_preference(
                t_rate_series,
                k=rate_k,
                theta_rate=theta_rate,
            )
            weekly_frames["T_RATE"] = t_rate_series.to_frame("RATE")
            weekly_frames["rate_pref"] = rate_pref_series.to_frame("RATE")
        else:
            warnings.append("RATE bucket not found for T_RATE/rate_pref")

        daily_rows = self._build_daily_rows(
            daily_frames,
            enabled,
            strategy_id,
            version,
            run_id,
            snapshot_id,
            rebalance_date,
            params_hash,
        )
        weekly_rows = self._build_weekly_rows(
            weekly_frames,
            enabled,
            strategy_id,
            version,
            run_id,
            snapshot_id,
            rebalance_date,
            params_hash,
        )

        features_generated = [
            name for name in enabled if name in daily_frames or name in weekly_frames
        ]
        summary = FeatureRunSummary(
            calc_date_range=(calc_start, calc_end),
            instruments_count=len(prices.columns),
            features_generated=features_generated,
            rows_upserted_daily=0,
            rows_upserted_weekly=0,
            warnings=warnings,
            coverage=coverage,
        )
        return summary, daily_rows, weekly_rows

    def _params_dict(self) -> dict[str, object]:
        return {
            "short_window": float(self.config.short_window),
            "long_window": float(self.config.long_window),
            "vol_window": float(self.config.vol_window),
            "annualize": float(self.config.annualize),
            "theta_on": self.config.theta_on,
            "theta_off": self.config.theta_off,
            "theta_minus": float(self.config.theta_minus),
            "sigma_min": self.config.sigma_min,
            "sigma_max": self.config.sigma_max,
            "kappa_sigma": self.config.kappa_sigma,
            "rate_k": float(self.config.rate_k),
            "theta_rate": float(self.config.theta_rate),
            "x0": self.config.x0,
        }

    def _param_number(self, value: object) -> float:
        if isinstance(value, dict):
            if "__default__" in value:
                return float(value["__default__"])
            return float(next(iter(value.values())))
        return float(cast(Any, value))

    def _resolve_bucket_param(self, value: object, bucket: str) -> float:
        if isinstance(value, dict):
            if bucket in value:
                return float(value[bucket])
            if "__default__" in value:
                return float(value["__default__"])
            return float(next(iter(value.values())))
        return float(cast(Any, value))

    def _gate_state_by_bucket(
        self, weekly_history: pd.DataFrame, *, theta_on: object, theta_off: object
    ) -> pd.DataFrame:
        gate_state = pd.DataFrame(index=weekly_history.index, columns=weekly_history.columns)
        for bucket in weekly_history.columns:
            gate_state[bucket] = computer.hysteresis_gate(
                weekly_history[[bucket]],
                theta_on=self._resolve_bucket_param(theta_on, bucket),
                theta_off=self._resolve_bucket_param(theta_off, bucket),
            )[bucket]
        return gate_state

    def _sigma_eff_by_bucket(
        self, weekly_sigma: pd.DataFrame, *, sigma_min: object
    ) -> pd.DataFrame:
        sigma_eff = pd.DataFrame(index=weekly_sigma.index, columns=weekly_sigma.columns)
        for bucket in weekly_sigma.columns:
            sigma_eff[bucket] = computer.sigma_eff(
                weekly_sigma[[bucket]],
                sigma_min=self._resolve_bucket_param(sigma_min, bucket),
            )[bucket]
        return sigma_eff

    def _f_sigma_by_bucket(
        self, weekly_sigma: pd.DataFrame, *, sigma_max: object, kappa_sigma: object
    ) -> pd.DataFrame:
        f_sigma = pd.DataFrame(index=weekly_sigma.index, columns=weekly_sigma.columns)
        for bucket in weekly_sigma.columns:
            f_sigma[bucket] = computer.tradability_filter(
                weekly_sigma[[bucket]],
                sigma_max=self._resolve_bucket_param(sigma_max, bucket),
                kappa=self._resolve_bucket_param(kappa_sigma, bucket),
            )[bucket]
        return f_sigma

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
                raise ValueError(f"missing proxy data for {bucket}")
            series = (
                pd.DataFrame(data)
                .assign(date=lambda df: pd.to_datetime(df["date"]))
                .set_index("date")["close"]
                .sort_index()
                .rename(bucket)
            )
            frames[bucket] = series.reindex(calendar.dates)
        if not frames:
            raise ValueError("no proxy series available")
        return frames

    def _coverage(
        self,
        series_map: Mapping[str, pd.Series],
        start_date: date,
        end_date: date,
    ) -> dict[str, dict[str, Any]]:
        coverage: dict[str, dict[str, Any]] = {}
        for col, series in series_map.items():
            series = series.dropna()
            if series.empty:
                raise ValueError(f"missing price coverage for {col}")
            window = series.loc[
                (series.index >= pd.Timestamp(start_date))
                & (series.index <= pd.Timestamp(end_date))
            ]
            if window.empty:
                raise ValueError(f"missing price coverage for {col}")
            first_trade_date = window.index.min().date()
            min_date = series.index.min().date()
            max_date = series.index.max().date()
            if min_date > first_trade_date or max_date < end_date:
                raise ValueError(f"insufficient price coverage for {col}")
            coverage[col] = {
                "count": int(series.shape[0]),
                "min": min_date.isoformat(),
                "max": max_date.isoformat(),
                "latest": max_date.isoformat(),
            }
        return coverage

    def _build_daily_rows(
        self,
        frames: Mapping[str, pd.DataFrame],
        enabled: list[str],
        strategy_id: str,
        version: str,
        run_id: str,
        snapshot_id: str | None,
        rebalance_date: date,
        params_hash: str,
    ) -> list[FeatureRow]:
        rows: list[FeatureRow] = []
        for name, frame in frames.items():
            if name not in enabled:
                continue
            stacked = computer.build_feature_frame(name, frame)
            for _, row in stacked.iterrows():
                rows.append(
                    {
                        "strategy_id": strategy_id,
                        "version": version,
                        "instrument_id": row["instrument_id"],
                        "calc_date": row["calc_date"].date(),
                        "feature_name": name,
                        "value": float(row["value"]),
                        "meta_json": {
                            "run_id": run_id,
                            "snapshot_id": snapshot_id,
                            "rebalance_date": rebalance_date.isoformat(),
                            "params_hash": params_hash,
                        },
                    }
                )
        return rows

    def _build_weekly_rows(
        self,
        frames: Mapping[str, pd.DataFrame],
        enabled: list[str],
        strategy_id: str,
        version: str,
        run_id: str,
        snapshot_id: str | None,
        rebalance_date: date,
        params_hash: str,
    ) -> list[FeatureWeeklySampleRow]:
        rows: list[FeatureWeeklySampleRow] = []
        for name, frame in frames.items():
            if name not in enabled:
                continue
            if frame.empty:
                continue
            latest = frame.loc[[frame.index.max()]]
            stacked = computer.build_feature_frame(name, latest)
            for _, row in stacked.iterrows():
                rows.append(
                    {
                        "strategy_id": strategy_id,
                        "version": version,
                        "instrument_id": row["instrument_id"],
                        "rebalance_date": rebalance_date,
                        "feature_name": name,
                        "value": float(row["value"]),
                        "meta_json": {
                            "run_id": run_id,
                            "snapshot_id": snapshot_id,
                            "rebalance_date": rebalance_date.isoformat(),
                            "params_hash": params_hash,
                        },
                    }
                )
        return rows
