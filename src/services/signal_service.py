from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..core.config import SignalConfig
from ..repo.inputs import BetaRepo, BucketRepo, FactorRepo
from ..repo.outputs import FeatureWeeklySampleRepo, SignalRepo, SignalRow


@dataclass(frozen=True)
class SignalRunSummary:
    rebalance_date: date
    rows_upserted: int
    warnings: list[str]
    checks: dict[str, Any]


@dataclass
class SignalService:
    bucket_repo: BucketRepo
    feature_weekly_repo: FeatureWeeklySampleRepo
    factor_repo: FactorRepo
    beta_repo: BetaRepo
    signal_repo: SignalRepo
    config: SignalConfig

    def compute_and_persist_signals(
        self,
        *,
        run_id: str,
        strategy_id: str,
        version: str,
        snapshot_id: str | None,
        rebalance_date: date,
        universe: dict[str, Any],
        dry_run: bool,
        force_recompute: bool,
    ) -> SignalRunSummary:
        _ = force_recompute
        buckets = self.bucket_repo.get_range(
            [int(b) for b in universe.get("bucket_ids", [])] or None
        )
        if not buckets:
            raise ValueError("bucket universe is empty")

        bucket_assets: dict[str, list[str]] = {}
        for bucket in buckets:
            assets = [a.strip() for a in bucket["assets"].split(",") if a.strip()]
            bucket_assets[bucket["bucket_name"]] = assets

        bucket_names = list(bucket_assets.keys())
        features_needed = [
            "T",
            "gate_state",
            "sigma_ann",
            "sigma_eff",
            "f_sigma",
            "T_RATE",
            "rate_pref",
        ]
        feature_rows = self.feature_weekly_repo.get_range(
            strategy_id=strategy_id,
            version=version,
            rebalance_date=rebalance_date,
            instrument_ids=bucket_names,
            feature_names=features_needed,
        )
        if not feature_rows:
            raise ValueError("feature_weekly_sample is empty")

        feature_df = pd.DataFrame(feature_rows)
        if feature_df.empty:
            raise ValueError("feature_weekly_sample is empty")

        bucket_signals = self._build_bucket_signals(
            feature_df,
            run_id,
            snapshot_id,
            bucket_names,
        )

        tilt_rows, checks, warnings = self._build_tilt_signals(
            bucket_assets=bucket_assets,
            rebalance_date=rebalance_date,
            run_id=run_id,
            snapshot_id=snapshot_id,
            strategy_id=strategy_id,
            version=version,
        )

        all_rows = bucket_signals + tilt_rows
        if dry_run:
            return SignalRunSummary(
                rebalance_date=rebalance_date,
                rows_upserted=0,
                warnings=warnings,
                checks=checks,
            )

        rows_upserted = self.signal_repo.upsert_many(all_rows)
        return SignalRunSummary(
            rebalance_date=rebalance_date,
            rows_upserted=rows_upserted,
            warnings=warnings,
            checks=checks,
        )

    def _build_bucket_signals(
        self,
        feature_df: pd.DataFrame,
        run_id: str,
        snapshot_id: str | None,
        bucket_names: list[str],
    ) -> list[SignalRow]:
        signals: list[SignalRow] = []
        feature_df = feature_df[feature_df["instrument_id"].isin(bucket_names)]
        for _, row in feature_df.iterrows():
            signals.append(
                {
                    "strategy_id": row["strategy_id"],
                    "version": row["version"],
                    "instrument_id": row["instrument_id"],
                    "rebalance_date": row["rebalance_date"],
                    "signal_name": row["feature_name"],
                    "bucket_id": row["instrument_id"],
                    "value": float(row["value"]),
                    "meta_json": {
                        "run_id": run_id,
                        "snapshot_id": snapshot_id,
                        "bucket_id": row["instrument_id"],
                    },
                }
            )

        # raw weight components for audit
        required = ["gate_state", "T", "sigma_eff", "f_sigma"]
        for bucket in bucket_names:
            bucket_slice = feature_df[feature_df["instrument_id"] == bucket]
            values: dict[str, float] = {}
            for name in required:
                subset = bucket_slice[bucket_slice["feature_name"] == name]
                if subset.empty:
                    raise ValueError(f"missing feature {name} for bucket {bucket}")
                values[name] = float(subset["value"].iloc[0])
            if values["sigma_eff"] == 0:
                raise ValueError(f"sigma_eff is zero for bucket {bucket}")
            raw_components = {
                "raw_weight_component_risk_budget": 1.0,
                "raw_weight_component_gate": values["gate_state"],
                "raw_weight_component_trend": values["T"],
                "raw_weight_component_inv_sigma_eff": 1.0 / values["sigma_eff"],
                "raw_weight_component_f_sigma": values["f_sigma"],
            }
            for name, value in raw_components.items():
                signals.append(
                    {
                        "strategy_id": bucket_slice["strategy_id"].iloc[0],
                        "version": bucket_slice["version"].iloc[0],
                        "instrument_id": bucket,
                        "rebalance_date": bucket_slice["rebalance_date"].iloc[0],
                        "signal_name": name,
                        "bucket_id": bucket,
                        "value": float(value),
                        "meta_json": {
                            "run_id": run_id,
                            "snapshot_id": snapshot_id,
                            "bucket_id": bucket,
                        },
                    }
                )
        return signals

    def _build_tilt_signals(
        self,
        bucket_assets: dict[str, list[str]],
        rebalance_date: date,
        run_id: str,
        snapshot_id: str | None,
        strategy_id: str,
        version: str,
    ) -> tuple[list[SignalRow], dict[str, Any], list[str]]:
        warnings: list[str] = []
        checks: dict[str, Any] = {}
        tilt_rows: list[SignalRow] = []

        factors = self.config.tilt_factors
        lookback_days = self.config.tilt_lookback_days
        start_date = rebalance_date - timedelta(days=lookback_days)
        factor_rows = self.factor_repo.get_range(start_date, rebalance_date)
        factor_df = pd.DataFrame(factor_rows)
        if factor_df.empty:
            raise ValueError("factor returns are empty for tilt")
        s_t = factor_df[factors].sum().astype(float)
        scales = pd.Series(self.config.tilt_scales)
        s_t = s_t / scales.reindex(factors).fillna(1.0)
        s_norm = np.linalg.norm(s_t.values)
        if s_norm == 0:
            warnings.append("tilt tendency vector is zero")

        beta_rows = self.beta_repo.get_range(
            [code for assets in bucket_assets.values() for code in assets],
            rebalance_date,
            rebalance_date,
        )
        beta_df = pd.DataFrame(beta_rows)
        if beta_df.empty:
            raise ValueError("fund_beta missing for rebalance_date")

        exposures = beta_df.set_index("code")[factors].astype(float)

        for bucket_id, assets in bucket_assets.items():
            if bucket_id in {"RATE", "CASH"}:
                for asset in assets:
                    tilt_rows.extend(
                        self._tilt_rows(
                            asset=asset,
                            bucket_id=bucket_id,
                            rebalance_date=rebalance_date,
                            run_id=run_id,
                            snapshot_id=snapshot_id,
                            strategy_id=strategy_id,
                            version=version,
                            score=0.0,
                            weight=1.0,
                        )
                    )
                checks[bucket_id] = {"tilt_weight_sum": 1.0}
                continue

            scores: dict[str, float] = {}
            eligible: dict[str, bool] = {}
            for asset in assets:
                if asset not in exposures.index:
                    eligible[asset] = False
                    continue
                vector = exposures.loc[asset].values
                if np.any(pd.isna(vector)):
                    eligible[asset] = False
                    continue
                v_norm = np.linalg.norm(vector)
                if v_norm == 0 or s_norm == 0:
                    eligible[asset] = False
                    continue
                score = float(np.dot(vector, s_t.values) / (v_norm * s_norm + self.config.tilt_eps))
                scores[asset] = score
                eligible[asset] = True

            if not any(eligible.values()):
                raise ValueError(f"no eligible assets for bucket {bucket_id}")

            score_vec = np.array([scores.get(asset, 0.0) for asset in assets])
            weights = self._softmax(score_vec, self.config.tilt_temperature)
            weight_map = {
                asset: float(weight) if eligible.get(asset, False) else 0.0
                for asset, weight in zip(assets, weights)
            }
            total = sum(weight_map.values())
            if total <= 0:
                raise ValueError(f"tilt weights sum to zero for bucket {bucket_id}")
            weight_map = {asset: weight / total for asset, weight in weight_map.items()}
            weight_sum = sum(weight_map.values())
            if not np.isclose(weight_sum, 1.0, atol=1e-8):
                raise ValueError(f"tilt_weight sum != 1 for bucket {bucket_id}")
            if any(weight < 0 for weight in weight_map.values()):
                raise ValueError(f"tilt_weight negative for bucket {bucket_id}")
            checks[bucket_id] = {"tilt_weight_sum": weight_sum}

            for asset in assets:
                tilt_rows.extend(
                    self._tilt_rows(
                        asset=asset,
                        bucket_id=bucket_id,
                        rebalance_date=rebalance_date,
                        run_id=run_id,
                        snapshot_id=snapshot_id,
                        strategy_id=strategy_id,
                        version=version,
                        score=scores.get(asset, 0.0),
                        weight=weight_map[asset],
                    )
                )

        return tilt_rows, checks, warnings

    def _tilt_rows(
        self,
        asset: str,
        bucket_id: str,
        rebalance_date: date,
        run_id: str,
        snapshot_id: str | None,
        strategy_id: str,
        version: str,
        *,
        score: float,
        weight: float,
    ) -> list[SignalRow]:
        return [
            {
                "strategy_id": strategy_id,
                "version": version,
                "instrument_id": asset,
                "rebalance_date": rebalance_date,
                "signal_name": "tilt_score",
                "bucket_id": bucket_id,
                "value": float(score),
                "meta_json": {
                    "run_id": run_id,
                    "snapshot_id": snapshot_id,
                    "bucket_id": bucket_id,
                },
            },
            {
                "strategy_id": strategy_id,
                "version": version,
                "instrument_id": asset,
                "rebalance_date": rebalance_date,
                "signal_name": "tilt_weight",
                "bucket_id": bucket_id,
                "value": float(weight),
                "meta_json": {
                    "run_id": run_id,
                    "snapshot_id": snapshot_id,
                    "bucket_id": bucket_id,
                },
            },
        ]

    @staticmethod
    def _softmax(scores: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
        scaled = scores / max(temperature, 1e-12)
        scaled -= scaled.max()
        exp = np.exp(scaled)
        denom = float(exp.sum())
        return cast(NDArray[np.float64], np.asarray(exp / denom, dtype=np.float64))
