from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Mapping

import pandas as pd

from ..core.config import FeatureConfig
from ..features import computer, sampler
from ..features.registry import FeatureRegistry, FeatureSpec
from ..repo.inputs import BucketRepo, MarketRepo
from ..repo.outputs import FeatureRepo, FeatureRow, FeatureWeeklySampleRepo, FeatureWeeklySampleRow


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

        if dry_run:
            return summary

        daily_count = self.feature_repo.upsert_many(daily_rows) if daily_rows else 0
        weekly_count = self.feature_weekly_repo.upsert_many(weekly_rows) if weekly_rows else 0
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

        buckets = self.bucket_repo.get_range(
            [int(b) for b in universe.get("bucket_ids", [])] or None
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

        rows = self.market_repo.get_range(proxy_codes, calc_start, calc_end)
        if not rows:
            raise ValueError("no proxy price data available")

        prices = self._build_prices(rows, bucket_proxies)
        coverage = self._coverage(prices, calc_start, calc_end)

        params = self._params_dict()
        params.update(feature_set.feature_params)
        params_hash = computer.feature_params_hash(params)

        short_window = int(params["short_window"])
        long_window = int(params["long_window"])
        vol_window = int(params["vol_window"])
        annualize = int(params["annualize"])
        theta_on = float(params["theta_on"])
        theta_off = float(params["theta_off"])
        theta_minus = float(params["theta_minus"])
        sigma_min = float(params["sigma_min"])
        sigma_max = float(params["sigma_max"])
        kappa_sigma = float(params["kappa_sigma"])
        rate_k = float(params["rate_k"])
        theta_rate = float(params["theta_rate"])

        returns = computer.log_returns(prices)
        sigma_ann = computer.sigma_annualized(returns, window=vol_window, annualize=annualize)
        trend = computer.trend_strength(
            prices,
            sigma_ann,
            short_window=short_window,
            long_window=long_window,
        )

        daily_frames: dict[str, pd.DataFrame] = {
            "r_log_daily": returns,
            "sigma_ann": sigma_ann,
            "T": trend,
        }

        weekly_history = sampler.weekly_history(trend, rebalance_date=rebalance_date)
        weekly_sigma = sampler.weekly_history(sigma_ann, rebalance_date=rebalance_date)

        gate_state = computer.hysteresis_gate(
            weekly_history,
            theta_on=theta_on,
            theta_off=theta_off,
        )
        down_drift = computer.down_drift(weekly_history, theta_minus=theta_minus)
        sigma_eff = computer.sigma_eff(weekly_sigma, sigma_min=sigma_min)
        f_sigma = computer.tradability_filter(weekly_sigma, sigma_max=sigma_max, kappa=kappa_sigma)

        weekly_frames: dict[str, pd.DataFrame] = {
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

    def _params_dict(self) -> dict[str, float]:
        return {
            "short_window": float(self.config.short_window),
            "long_window": float(self.config.long_window),
            "vol_window": float(self.config.vol_window),
            "annualize": float(self.config.annualize),
            "theta_on": float(self.config.theta_on),
            "theta_off": float(self.config.theta_off),
            "theta_minus": float(self.config.theta_minus),
            "sigma_min": float(self.config.sigma_min),
            "sigma_max": float(self.config.sigma_max),
            "kappa_sigma": float(self.config.kappa_sigma),
            "rate_k": float(self.config.rate_k),
            "theta_rate": float(self.config.theta_rate),
        }

    def _build_prices(
        self,
        rows: Iterable[Mapping[str, Any]],
        bucket_proxies: Mapping[str, str | None],
    ) -> pd.DataFrame:
        frames = []
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
            frames.append(series)
        if not frames:
            raise ValueError("no proxy series available")
        return pd.concat(frames, axis=1)

    def _coverage(
        self,
        prices: pd.DataFrame,
        start_date: date,
        end_date: date,
    ) -> dict[str, dict[str, Any]]:
        coverage: dict[str, dict[str, Any]] = {}
        for col in prices.columns:
            series = prices[col].dropna()
            if series.empty:
                raise ValueError(f"missing price coverage for {col}")
            min_date = series.index.min().date()
            max_date = series.index.max().date()
            if min_date > start_date or max_date < end_date:
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
