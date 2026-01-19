from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..repo.inputs import AuxRepo, BetaRepo, BucketRepo, FactorRepo, MarketRepo, NavRepo
from ..repo.outputs import (
    FeatureRepo,
    FeatureRow,
    SignalRepo,
    SignalRow,
    WeightRepo,
    WeightRow,
)
from .contracts import JobType, RunContext, RunRequest, RunResult, RunStatus
from .run_audit_service import RunAuditService


class FeatureService(Protocol):
    def compute(
        self,
        calc_start: date,
        calc_end: date,
        universe: dict[str, Any],
        snapshot_id: str | None,
        dry_run: bool,
    ) -> list[FeatureRow]: ...


class SignalService(Protocol):
    def compute(
        self,
        rebalance_date: date,
        universe: dict[str, Any],
        snapshot_id: str | None,
        dry_run: bool,
    ) -> list[SignalRow]: ...


class PortfolioService(Protocol):
    def compute_weights(
        self,
        rebalance_date: date,
        universe: dict[str, Any],
        snapshot_id: str | None,
        dry_run: bool,
    ) -> list[WeightRow]: ...


Clock = Callable[[], datetime]


@dataclass(frozen=True)
class JobRunner:
    audit: RunAuditService
    bucket_repo: BucketRepo
    market_repo: MarketRepo
    factor_repo: FactorRepo
    nav_repo: NavRepo
    beta_repo: BetaRepo
    aux_repo: AuxRepo
    feature_repo: FeatureRepo
    signal_repo: SignalRepo
    weight_repo: WeightRepo
    feature_service: FeatureService
    signal_service: SignalService
    portfolio_service: PortfolioService
    clock: Clock = lambda: datetime.now(timezone.utc)

    def run_feature(self, ctx: RunContext, req: RunRequest) -> RunResult:
        if ctx.job_type != JobType.FEATURE:
            raise ValueError("ctx.job_type must be FEATURE")
        return self._run_with_audit(ctx, req, self._feature_flow)

    def run_signal(self, ctx: RunContext, req: RunRequest) -> RunResult:
        if ctx.job_type != JobType.SIGNAL:
            raise ValueError("ctx.job_type must be SIGNAL")
        return self._run_with_audit(ctx, req, self._signal_flow)

    def run_portfolio(self, ctx: RunContext, req: RunRequest) -> RunResult:
        if ctx.job_type != JobType.PORTFOLIO:
            raise ValueError("ctx.job_type must be PORTFOLIO")
        return self._run_with_audit(ctx, req, self._portfolio_flow)

    def run_full(self, ctx: RunContext, req: RunRequest) -> RunResult:
        if ctx.job_type != JobType.FULL:
            raise ValueError("ctx.job_type must be FULL")
        return self._run_with_audit(ctx, req, self._full_flow)

    def _run_with_audit(
        self,
        ctx: RunContext,
        req: RunRequest,
        flow: Callable[[RunContext, RunRequest], RunResult],
    ) -> RunResult:
        self.audit.start_run(ctx, req)
        try:
            result = flow(ctx, req)
        except Exception as err:  # pragma: no cover - top-level guard
            self.audit.finish_failed(ctx.run_id, err)
            return RunResult(
                run_id=ctx.run_id,
                status=RunStatus.FAILED,
                job_type=ctx.job_type,
                time_start=ctx.created_at,
                time_end=self.clock(),
                input_range={},
                outputs={},
                error={"message": str(err), "stack": str(err)},
            )
        return result

    def _feature_flow(self, ctx: RunContext, req: RunRequest) -> RunResult:
        calc_start, calc_end = self._resolve_calc_range(req)
        universe = self._resolve_universe(req)
        coverage = self._build_input_coverage(calc_start, calc_end, universe)
        self._assert_coverage(coverage, req.dry_run)

        input_range = {
            "calc_range": [calc_start.isoformat(), calc_end.isoformat()],
            "universe": universe,
            "coverage": coverage,
        }
        self.audit.update_inputs(ctx.run_id, input_range)

        rows = self.feature_service.compute(
            calc_start, calc_end, universe, ctx.snapshot_id, req.dry_run
        )
        rows_upserted = 0
        if not req.dry_run:
            rows_upserted = self.feature_repo.upsert_many(rows)

        outputs = {
            "rows_upserted": {"feature_daily": rows_upserted},
            "date_coverage": coverage,
            "checks": {},
        }
        self.audit.finish_success(ctx.run_id, outputs)
        return RunResult(
            run_id=ctx.run_id,
            status=RunStatus.SUCCESS,
            job_type=ctx.job_type,
            time_start=ctx.created_at,
            time_end=self.clock(),
            input_range=input_range,
            outputs=outputs,
            error=None,
        )

    def _signal_flow(self, ctx: RunContext, req: RunRequest) -> RunResult:
        universe = self._resolve_universe(req)
        input_range = {
            "rebalance_date": req.rebalance_date.isoformat(),
            "universe": universe,
        }
        self.audit.update_inputs(ctx.run_id, input_range)

        rows = self.signal_service.compute(
            req.rebalance_date, universe, ctx.snapshot_id, req.dry_run
        )
        rows_upserted = 0
        if not req.dry_run:
            rows_upserted = self.signal_repo.upsert_many(rows)

        outputs = {"rows_upserted": {"signal_weekly": rows_upserted}, "checks": {}}
        self.audit.finish_success(ctx.run_id, outputs)
        return RunResult(
            run_id=ctx.run_id,
            status=RunStatus.SUCCESS,
            job_type=ctx.job_type,
            time_start=ctx.created_at,
            time_end=self.clock(),
            input_range=input_range,
            outputs=outputs,
            error=None,
        )

    def _portfolio_flow(self, ctx: RunContext, req: RunRequest) -> RunResult:
        universe = self._resolve_universe(req)
        input_range = {
            "rebalance_date": req.rebalance_date.isoformat(),
            "universe": universe,
        }
        self.audit.update_inputs(ctx.run_id, input_range)

        rows = self.portfolio_service.compute_weights(
            req.rebalance_date, universe, ctx.snapshot_id, req.dry_run
        )
        rows_upserted = 0
        if not req.dry_run:
            rows_upserted = self.weight_repo.upsert_many(rows)

        outputs = {
            "rows_upserted": {"portfolio_weight_weekly": rows_upserted},
            "checks": {},
        }
        self.audit.finish_success(ctx.run_id, outputs)
        return RunResult(
            run_id=ctx.run_id,
            status=RunStatus.SUCCESS,
            job_type=ctx.job_type,
            time_start=ctx.created_at,
            time_end=self.clock(),
            input_range=input_range,
            outputs=outputs,
            error=None,
        )

    def _full_flow(self, ctx: RunContext, req: RunRequest) -> RunResult:
        calc_start, calc_end = self._resolve_calc_range(req)
        universe = self._resolve_universe(req)
        coverage = self._build_input_coverage(calc_start, calc_end, universe)
        self._assert_coverage(coverage, req.dry_run)

        input_range = {
            "calc_range": [calc_start.isoformat(), calc_end.isoformat()],
            "universe": universe,
            "coverage": coverage,
        }
        self.audit.update_inputs(ctx.run_id, input_range)

        step_outputs: dict[str, Any] = {}

        feature_rows = self.feature_service.compute(
            calc_start, calc_end, universe, ctx.snapshot_id, req.dry_run
        )
        feature_upsert = 0
        if not req.dry_run:
            feature_upsert = self.feature_repo.upsert_many(feature_rows)
        step_outputs["feature"] = {"rows_upserted": feature_upsert}

        signal_rows = self.signal_service.compute(
            req.rebalance_date, universe, ctx.snapshot_id, req.dry_run
        )
        signal_upsert = 0
        if not req.dry_run:
            signal_upsert = self.signal_repo.upsert_many(signal_rows)
        step_outputs["signal"] = {"rows_upserted": signal_upsert}

        weight_rows = self.portfolio_service.compute_weights(
            req.rebalance_date, universe, ctx.snapshot_id, req.dry_run
        )
        weight_upsert = 0
        if not req.dry_run:
            weight_upsert = self.weight_repo.upsert_many(weight_rows)
        step_outputs["portfolio"] = {"rows_upserted": weight_upsert}

        outputs = {
            "rows_upserted": {
                "feature_daily": feature_upsert,
                "signal_weekly": signal_upsert,
                "portfolio_weight_weekly": weight_upsert,
            },
            "steps": step_outputs,
            "date_coverage": coverage,
            "checks": {},
        }
        self.audit.finish_success(ctx.run_id, outputs)
        return RunResult(
            run_id=ctx.run_id,
            status=RunStatus.SUCCESS,
            job_type=ctx.job_type,
            time_start=ctx.created_at,
            time_end=self.clock(),
            input_range=input_range,
            outputs=outputs,
            error=None,
        )

    def _resolve_calc_range(self, req: RunRequest) -> tuple[date, date]:
        calc_end = req.calc_end or req.rebalance_date
        if req.calc_start:
            return req.calc_start, calc_end
        if req.lookback and "market_days" in req.lookback:
            days = int(req.lookback["market_days"])
            return calc_end - timedelta(days=days), calc_end
        raise ValueError("calc_start or lookback.market_days must be provided")

    def _resolve_universe(self, req: RunRequest) -> dict[str, Any]:
        universe = req.universe or {}
        portfolio_id = universe.get("portfolio_id", "main")
        bucket_ids_raw = universe.get("bucket_ids")
        instrument_ids = universe.get("instrument_ids")

        bucket_ids: list[int] | None = None
        if bucket_ids_raw:
            bucket_ids = [int(b) for b in bucket_ids_raw]

        if instrument_ids:
            instruments = list(instrument_ids)
        else:
            buckets = self.bucket_repo.get_range(bucket_ids)
            if not buckets:
                raise ValueError("bucket universe is empty")
            instruments = []
            for bucket in buckets:
                assets = [a.strip() for a in bucket["assets"].split(",") if a.strip()]
                instruments.extend(assets)

        return {
            "portfolio_id": portfolio_id,
            "bucket_ids": bucket_ids_raw or [],
            "instrument_ids": instruments,
        }

    def _build_input_coverage(
        self, calc_start: date, calc_end: date, universe: dict[str, Any]
    ) -> dict[str, Any]:
        instruments = universe.get("instrument_ids", [])
        market_rows = self.market_repo.get_range(["Au99.99.SGE"], calc_start, calc_end)
        factor_rows = self.factor_repo.get_range(calc_start, calc_end)
        nav_rows = self.nav_repo.get_range(instruments, calc_start, calc_end)
        beta_rows = self.beta_repo.get_range(instruments, calc_start, calc_end)
        gold_rows = self.aux_repo.get_gold_range(calc_start, calc_end)

        return {
            "market": self._coverage(market_rows, "date"),
            "factors": self._coverage(factor_rows, "date"),
            "nav": self._coverage(nav_rows, "date"),
            "beta": self._coverage(beta_rows, "date"),
            "gold": self._coverage(gold_rows, "date"),
        }

    @staticmethod
    def _coverage(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
        if not rows:
            return {"count": 0, "min": None, "max": None, "latest": None}
        dates = sorted(row[field] for row in rows if row.get(field) is not None)
        if not dates:
            return {"count": len(rows), "min": None, "max": None, "latest": None}
        return {
            "count": len(rows),
            "min": dates[0].isoformat(),
            "max": dates[-1].isoformat(),
            "latest": dates[-1].isoformat(),
        }

    @staticmethod
    def _assert_coverage(coverage: dict[str, Any], dry_run: bool) -> None:
        if dry_run:
            return
        missing = [name for name, info in coverage.items() if info["count"] == 0]
        if missing:
            raise ValueError(f"missing input coverage: {', '.join(missing)}")
