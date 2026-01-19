from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock

from sqlalchemy.dialects import postgresql

from src.repo.outputs import JobRunRow, RunRepo, WeightRepo, _upsert_statement
from src.repo.outputs import _feature_table


def test_upsert_statement_contains_on_conflict() -> None:
    table = _feature_table(schema="cta")
    stmt = _upsert_statement(
        table,
        [
            {
                "strategy_id": "s",
                "version": "1",
                "instrument_id": "i",
                "calc_date": date(2026, 1, 1),
                "feature_name": "f",
                "value": 1.0,
                "meta_json": None,
            }
        ],
    )
    compiled = str(stmt.compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT" in compiled
    assert "strategy_id" in compiled


def test_run_repo_upsert_many_executes() -> None:
    engine = MagicMock()
    connection = MagicMock()
    engine.begin.return_value.__enter__.return_value = connection

    repo = RunRepo(engine, schema="cta")
    row: JobRunRow = {
        "run_id": "RUN_1",
        "job_type": "FULL",
        "strategy_id": "cta_trend_v1",
        "version": "1.0.0",
        "snapshot_id": None,
        "status": "SUCCESS",
        "time_start": datetime(2026, 1, 1, 0, 0, 0),
        "time_end": datetime(2026, 1, 1, 0, 1, 0),
        "input_range_json": None,
        "output_summary_json": None,
        "error_stack": None,
    }

    repo.upsert_many([row])
    assert connection.execute.called


def test_empty_upsert_returns_zero() -> None:
    engine = MagicMock()
    repo = WeightRepo(engine, schema="cta")
    assert repo.upsert_many([]) == 0
