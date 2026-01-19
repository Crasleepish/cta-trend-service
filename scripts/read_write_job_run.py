from __future__ import annotations

import argparse
import json
import uuid
from datetime import date, datetime

from src.core.config import load_app_config
from src.core.db import create_engine_from_config
from src.repo.inputs import AuxRepo, MarketRepo
from src.repo.outputs import JobRunRow, RunRepo


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _build_run_id(now: datetime) -> str:
    token = uuid.uuid4().hex[:4]
    return f"RUN_{now:%Y%m%d_%H%M%S}_{token}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Read a slice of data and write one job_run.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--rebalance-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--strategy-id", default="cta_trend_v1")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--job-type", default="FULL")
    args = parser.parse_args()

    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)
    rebalance_date = _parse_date(args.rebalance_date)

    config = load_app_config()
    engine = create_engine_from_config(config)

    market_repo = MarketRepo(engine, schema=config.db.schema_in)
    aux_repo = AuxRepo(engine, schema=config.db.schema_in)
    run_repo = RunRepo(engine, schema=config.db.schema_out)

    gold_rows = aux_repo.get_gold_range(start_date, end_date)
    index_rows = market_repo.get_range(["Au99.99.SGE"], start_date, end_date)

    now = datetime.now()
    run_row: JobRunRow = {
        "run_id": _build_run_id(now),
        "job_type": args.job_type,
        "strategy_id": args.strategy_id,
        "version": args.version,
        "snapshot_id": None,
        "status": "SUCCESS",
        "time_start": now,
        "time_end": datetime.now(),
        "input_range_json": {
            "rebalance_date": rebalance_date.isoformat(),
            "calc_date_range": [start_date.isoformat(), end_date.isoformat()],
            "instruments": ["Au99.99.SGE"],
        },
        "output_summary_json": {
            "gold_rows": len(gold_rows),
            "index_rows": len(index_rows),
        },
        "error_stack": None,
    }

    run_repo.upsert_many([run_row])
    print(json.dumps({"run_id": run_row["run_id"], "status": run_row["status"]}))


if __name__ == "__main__":
    main()
