from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd

from ..core.config import PortfolioConfig
from ..repo.inputs import BucketRepo
from ..repo.outputs import SignalRepo, WeightRepo, WeightRow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortfolioRunSummary:
    rebalance_date: date
    rows_upserted: int
    checks: dict[str, Any]
    warnings: list[str]


@dataclass
class PortfolioService:
    DEFENSIVE_BUCKETS: ClassVar[tuple[str, str]] = ("RATE", "CASH")

    bucket_repo: BucketRepo
    signal_repo: SignalRepo
    weight_repo: WeightRepo
    config: PortfolioConfig

    def compute_and_persist_weights_from_signals(
        self,
        *,
        run_id: str,
        strategy_id: str,
        version: str,
        snapshot_id: str | None,
        portfolio_id: str,
        rebalance_date: date,
        universe: dict[str, Any],
        dry_run: bool,
        force_recompute: bool,
    ) -> PortfolioRunSummary:
        _ = (run_id, snapshot_id, force_recompute)
        t0 = time.perf_counter()
        buckets = self.bucket_repo.get_range(
            [int(b) for b in universe.get("bucket_ids", [])] or None
        )
        logger.info(
            "portfolio.io bucket_repo.get_range run_id=%s rebalance_date=%s elapsed=%.3fs "
            "buckets=%s",
            run_id,
            rebalance_date,
            time.perf_counter() - t0,
            len(buckets),
        )
        if not buckets:
            raise ValueError("bucket universe is empty")

        bucket_assets: dict[str, list[str]] = {}
        for bucket in buckets:
            assets = [a.strip() for a in bucket["assets"].split(",") if a.strip()]
            bucket_assets[bucket["bucket_name"]] = assets

        bucket_names = sorted(bucket_assets.keys())
        t1 = time.perf_counter()
        signals = self.signal_repo.get_range(
            strategy_id=strategy_id,
            version=version,
            rebalance_date=rebalance_date,
        )
        logger.info(
            "portfolio.io signal_repo.get_range run_id=%s rebalance_date=%s elapsed=%.3fs rows=%s",
            run_id,
            rebalance_date,
            time.perf_counter() - t1,
            len(signals),
        )
        if not signals:
            raise ValueError("signal_weekly is empty")

        signal_df = pd.DataFrame(signals)
        if signal_df.empty:
            raise ValueError("signal_weekly is empty")

        bucket_signal_names = signal_df[
            signal_df["signal_name"].isin(self.config.bucket_signal_names)
        ]
        bucket_signal_buckets = sorted(bucket_signal_names["instrument_id"].unique())
        if bucket_signal_buckets != bucket_names:
            raise ValueError("bucket list mismatch between config and signals")

        risk_buckets = self._risk_buckets(bucket_names)
        if not risk_buckets:
            raise ValueError("no risk buckets available")

        raw_weights = self._build_raw_bucket_weights(signal_df, risk_buckets)
        w0 = self._normalize(raw_weights)
        sigma_eff = self._bucket_signal(signal_df, risk_buckets, "sigma_eff")

        implied = self._implied_risk_vol(w0, sigma_eff)
        if implied <= 0:
            alpha = 0.0
        else:
            alpha = min(1.0, self.config.sigma_target / implied)

        risk_weights = {b: alpha * w0[b] for b in risk_buckets}
        w_def = 1.0 - sum(risk_weights.values())
        if w_def < -1e-8:
            raise ValueError("risk weights exceed 1")

        w_rate, w_cash = self._defensive_split(signal_df, w_def)

        bucket_total: dict[str, float] = {**risk_weights}
        if "RATE" in bucket_names:
            bucket_total["RATE"] = w_rate
        elif w_rate > 0:
            raise ValueError("RATE bucket missing")
        if "CASH" in bucket_names:
            bucket_total["CASH"] = w_cash
        elif w_cash > 0:
            raise ValueError("CASH bucket missing")

        tilt_weights = self._tilt_weights(signal_df, bucket_assets)

        target_weights = self._build_weights(
            strategy_id=strategy_id,
            version=version,
            portfolio_id=portfolio_id,
            rebalance_date=rebalance_date,
            bucket_total=bucket_total,
            tilt_weights=tilt_weights,
            bucket_assets=bucket_assets,
            run_id=run_id,
        )

        executed_weights = self._apply_execution_controls(
            target_weights=target_weights,
            signal_df=signal_df,
            bucket_assets=bucket_assets,
            strategy_id=strategy_id,
            version=version,
            portfolio_id=portfolio_id,
            rebalance_date=rebalance_date,
        )

        checks = {"weights_sum": sum(w["target_weight"] for w in executed_weights)}
        if not np.isclose(checks["weights_sum"], 1.0, atol=1e-8):
            raise ValueError("portfolio weights do not sum to 1")

        if dry_run:
            return PortfolioRunSummary(
                rebalance_date=rebalance_date,
                rows_upserted=0,
                checks=checks,
                warnings=[],
            )

        t2 = time.perf_counter()
        rows_upserted = self.weight_repo.upsert_many(executed_weights)
        logger.info(
            "portfolio.upsert run_id=%s rebalance_date=%s elapsed=%.3fs rows=%s",
            run_id,
            rebalance_date,
            time.perf_counter() - t2,
            rows_upserted,
        )
        return PortfolioRunSummary(
            rebalance_date=rebalance_date,
            rows_upserted=rows_upserted,
            checks=checks,
            warnings=[],
        )

    def _risk_buckets(self, bucket_names: list[str]) -> list[str]:
        configured = [bucket for bucket in self.config.risk_buckets if bucket in bucket_names]
        if not configured:
            return []
        defensive = set(self.DEFENSIVE_BUCKETS)
        if any(bucket in defensive for bucket in configured):
            raise ValueError("risk_buckets cannot include defensive buckets")
        return configured

    def _build_raw_bucket_weights(
        self, signal_df: pd.DataFrame, risk_buckets: list[str]
    ) -> dict[str, float]:
        required = [
            "raw_weight_component_risk_budget",
            "raw_weight_component_gate",
            "raw_weight_component_trend",
            "raw_weight_component_inv_sigma_eff",
            "raw_weight_component_f_sigma",
            "raw_weight_component_path_quality",
        ]
        raw_weights: dict[str, float] = {}
        for bucket in risk_buckets:
            subset = signal_df[signal_df["instrument_id"] == bucket]
            values: list[float] = []
            for name in required:
                row = subset[subset["signal_name"] == name]
                if row.empty:
                    raise ValueError(f"missing signal {name} for bucket {bucket}")
                values.append(float(row["value"].iloc[0]))
            raw = float(np.prod(values))
            raw_weights[bucket] = raw
        return raw_weights

    @staticmethod
    def _normalize(values: dict[str, float]) -> dict[str, float]:
        total = sum(values.values())
        if total <= 0:
            return {k: 0.0 for k in values}
        return {k: v / total for k, v in values.items()}

    @staticmethod
    def _bucket_signal(signal_df: pd.DataFrame, buckets: list[str], name: str) -> dict[str, float]:
        values: dict[str, float] = {}
        for bucket in buckets:
            row = signal_df[
                (signal_df["instrument_id"] == bucket) & (signal_df["signal_name"] == name)
            ]
            if row.empty:
                raise ValueError(f"missing signal {name} for bucket {bucket}")
            values[bucket] = float(row["value"].iloc[0])
        return values

    def _implied_risk_vol(self, weights: dict[str, float], sigma_eff: dict[str, float]) -> float:
        total = 0.0
        for bucket, weight in weights.items():
            total += (weight * sigma_eff[bucket]) ** 2
        return float(np.sqrt(total))

    def _defensive_split(self, signal_df: pd.DataFrame, w_def: float) -> tuple[float, float]:
        if w_def <= 0:
            return 0.0, 0.0
        row = signal_df[
            (signal_df["instrument_id"] == "RATE") & (signal_df["signal_name"] == "rate_pref")
        ]
        if row.empty:
            raise ValueError("missing rate_pref signal")
        rate_pref = float(row["value"].iloc[0])
        w_rate = w_def * rate_pref
        w_cash = w_def - w_rate
        return w_rate, w_cash

    def _apply_execution_controls(
        self,
        *,
        target_weights: list[WeightRow],
        signal_df: pd.DataFrame,
        bucket_assets: dict[str, list[str]],
        strategy_id: str,
        version: str,
        portfolio_id: str,
        rebalance_date: date,
    ) -> list[WeightRow]:
        prev = self._previous_weights(
            strategy_id=strategy_id,
            version=version,
            portfolio_id=portfolio_id,
            rebalance_date=rebalance_date,
        )
        if prev is None:
            return target_weights
        prev_weights = {row["instrument_id"]: float(row["target_weight"]) for row in prev}
        bucket_by_asset = {
            asset: bucket for bucket, assets in bucket_assets.items() for asset in assets
        }
        down_drift = self._bucket_signal(signal_df, list(bucket_assets.keys()), "down_drift")
        adjusted: list[WeightRow] = []
        dead_band_hits = 0
        for row in target_weights:
            instr = row["instrument_id"]
            bucket = bucket_by_asset.get(instr)
            prev_w = prev_weights.get(instr, 0.0)
            target_w = float(row["target_weight"])
            if abs(target_w - prev_w) <= self.config.dead_band:
                target_w = prev_w
                dead_band_hits += 1
            if bucket and down_drift.get(bucket, 0.0) == 1.0:
                alpha = self.config.alpha_off
            else:
                alpha = self.config.alpha_on
            new_w = (1.0 - alpha) * prev_w + alpha * target_w
            adjusted.append({**row, "target_weight": new_w})
        if dead_band_hits:
            logger.info(
                "portfolio.exec.dead_band rebalance_date=%s hits=%s threshold=%.6f",
                rebalance_date,
                dead_band_hits,
                self.config.dead_band,
            )

        capped = self._apply_caps(
            weights=adjusted,
            bucket_by_asset=bucket_by_asset,
        )

        total = sum(float(row["target_weight"]) for row in capped)
        if total > 0:
            for row in capped:
                row["target_weight"] = float(row["target_weight"]) / total
        return capped

    def _apply_caps(
        self,
        *,
        weights: list[WeightRow],
        bucket_by_asset: dict[str, str],
    ) -> list[WeightRow]:
        max_asset = self.config.max_weight_asset
        max_bucket = self.config.max_weight_bucket
        if max_asset >= 1.0 and max_bucket >= 1.0:
            return weights
        weights = cast(list[WeightRow], [dict(row) for row in weights])
        excess = 0.0
        capped_assets = 0
        capped_buckets = 0
        for row in weights:
            if row["target_weight"] > max_asset:
                excess += row["target_weight"] - max_asset
                row["target_weight"] = max_asset
                capped_assets += 1
        bucket_totals: dict[str, float] = {}
        for row in weights:
            bucket = bucket_by_asset.get(row["instrument_id"], "")
            bucket_totals[bucket] = bucket_totals.get(bucket, 0.0) + float(row["target_weight"])
        for bucket, total in list(bucket_totals.items()):
            if total > max_bucket and bucket:
                scale = max_bucket / total
                for row in weights:
                    if bucket_by_asset.get(row["instrument_id"]) == bucket:
                        row["target_weight"] *= scale
                excess += total - max_bucket
                capped_buckets += 1
        if capped_assets or capped_buckets:
            logger.info(
                "portfolio.exec.caps capped_assets=%s capped_buckets=%s max_asset=%.6f "
                "max_bucket=%.6f excess=%.6f",
                capped_assets,
                capped_buckets,
                max_asset,
                max_bucket,
                excess,
            )
        if excess > 0:
            rate = None
            cash = None
            for row in weights:
                bucket = bucket_by_asset.get(row["instrument_id"], "")
                if bucket == "RATE":
                    rate = row
                if bucket == "CASH":
                    cash = row
            if rate is None and cash is None:
                return weights
            rate_w = float(rate["target_weight"]) if rate else 0.0
            cash_w = float(cash["target_weight"]) if cash else 0.0
            total_def = rate_w + cash_w
            if total_def <= 0:
                if cash is not None:
                    cash["target_weight"] += excess
            else:
                if rate is not None:
                    rate["target_weight"] += excess * (rate_w / total_def)
                if cash is not None:
                    cash["target_weight"] += excess * (cash_w / total_def)
            logger.info(
                "portfolio.exec.caps.redistribute excess=%.6f rate=%.6f cash=%.6f",
                excess,
                float(rate["target_weight"]) if rate else 0.0,
                float(cash["target_weight"]) if cash else 0.0,
            )
        return weights

    def _previous_weights(
        self,
        *,
        strategy_id: str,
        version: str,
        portfolio_id: str,
        rebalance_date: date,
    ) -> list[WeightRow] | None:
        prev_date = self.weight_repo.get_latest_date_before(
            strategy_id=strategy_id,
            version=version,
            portfolio_id=portfolio_id,
            rebalance_date=rebalance_date,
        )
        if prev_date is None:
            return None
        return self.weight_repo.get_by_date(
            strategy_id=strategy_id,
            version=version,
            portfolio_id=portfolio_id,
            rebalance_date=prev_date,
        )

    def _tilt_weights(
        self,
        signal_df: pd.DataFrame,
        bucket_assets: dict[str, list[str]],
    ) -> dict[str, dict[str, float]]:
        tilt_df = signal_df[signal_df["signal_name"] == "tilt_weight"]
        if tilt_df.empty:
            raise ValueError("missing tilt_weight signals")
        weights: dict[str, dict[str, float]] = {}
        for _, row in tilt_df.iterrows():
            bucket_id = row.get("bucket_id")
            if not bucket_id:
                raise ValueError("tilt_weight missing bucket_id")
            weights.setdefault(bucket_id, {})[row["instrument_id"]] = float(row["value"])

        for bucket, assets in bucket_assets.items():
            if bucket not in weights:
                raise ValueError(f"missing tilt_weight for bucket {bucket}")
            bucket_weights = weights[bucket]
            if set(bucket_weights.keys()) != set(assets):
                raise ValueError(f"tilt_weight incomplete for bucket {bucket}")
            total = sum(bucket_weights.values())
            if not np.isclose(total, 1.0, atol=1e-8):
                raise ValueError(f"tilt_weight sum != 1 for bucket {bucket}")
            if any(weight < 0 for weight in bucket_weights.values()):
                raise ValueError(f"tilt_weight negative for bucket {bucket}")
        return weights

    def _build_weights(
        self,
        *,
        strategy_id: str,
        version: str,
        portfolio_id: str,
        rebalance_date: date,
        bucket_total: dict[str, float],
        tilt_weights: dict[str, dict[str, float]],
        bucket_assets: dict[str, list[str]],
        run_id: str,
    ) -> list[WeightRow]:
        rows: list[WeightRow] = []
        for bucket, assets in bucket_assets.items():
            bucket_weight = bucket_total.get(bucket, 0.0)
            for asset in assets:
                rows.append(
                    {
                        "strategy_id": strategy_id,
                        "version": version,
                        "portfolio_id": portfolio_id,
                        "rebalance_date": rebalance_date,
                        "instrument_id": asset,
                        "target_weight": bucket_weight * tilt_weights[bucket][asset],
                        "bucket": bucket,
                        "meta_json": {"run_id": run_id},
                    }
                )
        return rows
