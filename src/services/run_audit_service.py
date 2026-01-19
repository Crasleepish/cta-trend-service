from __future__ import annotations

import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import uuid

from ..repo.outputs import JobRunRow, RunRepo
from .contracts import RunContext, RunRequest, RunStatus


@dataclass(frozen=True)
class RunAuditService:
    repo: RunRepo

    @staticmethod
    def generate_run_id(now: datetime | None = None) -> str:
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
        token = uuid.uuid4().hex[:4]
        return f"RUN_{stamp}_{token}"

    def start_run(self, ctx: RunContext, req: RunRequest) -> str:
        now = ctx.created_at
        row: JobRunRow = {
            "run_id": ctx.run_id,
            "job_type": ctx.job_type.value,
            "strategy_id": ctx.strategy_id,
            "version": ctx.version,
            "snapshot_id": ctx.snapshot_id,
            "status": RunStatus.RUNNING.value,
            "time_start": now,
            "time_end": None,
            "input_range_json": {
                "rebalance_date": req.rebalance_date.isoformat(),
                "calc_start": req.calc_start.isoformat() if req.calc_start else None,
                "calc_end": req.calc_end.isoformat() if req.calc_end else None,
                "lookback": req.lookback,
                "universe": req.universe,
                "tags": req.tags,
            },
            "output_summary_json": None,
            "error_stack": None,
        }
        self.repo.upsert_many([row])
        return ctx.run_id

    def update_inputs(self, run_id: str, input_range: dict[str, Any]) -> None:
        self.repo.update_fields(run_id, {"input_range_json": input_range})

    def finish_success(self, run_id: str, output_summary: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        self.repo.update_fields(
            run_id,
            {
                "status": RunStatus.SUCCESS.value,
                "time_end": now,
                "output_summary_json": output_summary,
                "error_stack": None,
            },
        )

    def finish_failed(
        self,
        run_id: str,
        err: Exception,
        output_summary: dict[str, Any] | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        self.repo.update_fields(
            run_id,
            {
                "status": RunStatus.FAILED.value,
                "time_end": now,
                "output_summary_json": output_summary,
                "error_stack": "".join(traceback.format_exception(err)),
            },
        )
