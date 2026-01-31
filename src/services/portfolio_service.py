from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ..core.config import PortfolioConfig
from ..repo.inputs import BucketRepo
from ..repo.outputs import SignalRepo, WeightRepo, WeightRow


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
        buckets = self.bucket_repo.get_range(
            [int(b) for b in universe.get("bucket_ids", [])] or None
        )
        if not buckets:
            raise ValueError("bucket universe is empty")

        bucket_assets: dict[str, list[str]] = {}
        for bucket in buckets:
            assets = [a.strip() for a in bucket["assets"].split(",") if a.strip()]
            bucket_assets[bucket["bucket_name"]] = assets

        bucket_names = sorted(bucket_assets.keys())
        signals = self.signal_repo.get_range(
            strategy_id=strategy_id,
            version=version,
            rebalance_date=rebalance_date,
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

        weights = self._build_weights(
            strategy_id=strategy_id,
            version=version,
            portfolio_id=portfolio_id,
            rebalance_date=rebalance_date,
            bucket_total=bucket_total,
            tilt_weights=tilt_weights,
            bucket_assets=bucket_assets,
            run_id=run_id,
        )

        checks = {"weights_sum": sum(w["target_weight"] for w in weights)}
        if not np.isclose(checks["weights_sum"], 1.0, atol=1e-8):
            raise ValueError("portfolio weights do not sum to 1")

        if dry_run:
            return PortfolioRunSummary(
                rebalance_date=rebalance_date,
                rows_upserted=0,
                checks=checks,
                warnings=[],
            )

        rows_upserted = self.weight_repo.upsert_many(weights)
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
