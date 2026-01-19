from __future__ import annotations

from datetime import date, datetime
from typing import Mapping, Sequence, TypedDict

from sqlalchemy import Column, Date, DateTime, Float, MetaData, String, Table, Text
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy import update
from sqlalchemy.engine import Engine

from .base import BaseRepo


class FeatureRow(TypedDict):
    strategy_id: str
    version: str
    instrument_id: str
    calc_date: date
    feature_name: str
    value: float
    meta_json: dict[str, object] | None


class SignalRow(TypedDict):
    strategy_id: str
    version: str
    instrument_id: str
    rebalance_date: date
    signal_name: str
    value: float
    meta_json: dict[str, object] | None


class WeightRow(TypedDict):
    strategy_id: str
    version: str
    portfolio_id: str
    rebalance_date: date
    instrument_id: str
    target_weight: float
    bucket: str
    meta_json: dict[str, object] | None


class JobRunRow(TypedDict):
    run_id: str
    job_type: str
    strategy_id: str
    version: str
    snapshot_id: str | None
    status: str
    time_start: datetime
    time_end: datetime | None
    input_range_json: dict[str, object] | None
    output_summary_json: dict[str, object] | None
    error_stack: str | None


def _feature_table(schema: str | None) -> Table:
    metadata = MetaData(schema=schema)
    return Table(
        "feature_daily",
        metadata,
        Column("strategy_id", String(64), primary_key=True),
        Column("version", String(32), primary_key=True),
        Column("instrument_id", String(64), primary_key=True),
        Column("calc_date", Date, primary_key=True),
        Column("feature_name", String(64), primary_key=True),
        Column("value", Float, nullable=False),
        Column("meta_json", JSONB, nullable=True),
    )


def _signal_table(schema: str | None) -> Table:
    metadata = MetaData(schema=schema)
    return Table(
        "signal_weekly",
        metadata,
        Column("strategy_id", String(64), primary_key=True),
        Column("version", String(32), primary_key=True),
        Column("instrument_id", String(64), primary_key=True),
        Column("rebalance_date", Date, primary_key=True),
        Column("signal_name", String(64), primary_key=True),
        Column("value", Float, nullable=False),
        Column("meta_json", JSONB, nullable=True),
    )


def _weight_table(schema: str | None) -> Table:
    metadata = MetaData(schema=schema)
    return Table(
        "portfolio_weight_weekly",
        metadata,
        Column("strategy_id", String(64), primary_key=True),
        Column("version", String(32), primary_key=True),
        Column("portfolio_id", String(64), primary_key=True),
        Column("rebalance_date", Date, primary_key=True),
        Column("instrument_id", String(64), primary_key=True),
        Column("target_weight", Float, nullable=False),
        Column("bucket", String(64), nullable=False),
        Column("meta_json", JSONB, nullable=True),
    )


def _job_run_table(schema: str | None) -> Table:
    metadata = MetaData(schema=schema)
    return Table(
        "job_run",
        metadata,
        Column("run_id", String(64), primary_key=True),
        Column("job_type", String(32), nullable=False),
        Column("strategy_id", String(64), nullable=False),
        Column("version", String(32), nullable=False),
        Column("snapshot_id", String(64), nullable=True),
        Column("status", String(16), nullable=False),
        Column("time_start", DateTime, nullable=False),
        Column("time_end", DateTime, nullable=True),
        Column("input_range_json", JSONB, nullable=True),
        Column("output_summary_json", JSONB, nullable=True),
        Column("error_stack", Text, nullable=True),
    )


def _upsert_statement(table: Table, rows: Sequence[Mapping[str, object]]):
    stmt = insert(table).values(rows)
    pk_cols = [col.name for col in table.primary_key.columns]
    update_cols = {
        col.name: stmt.excluded[col.name]
        for col in table.columns
        if col.name not in pk_cols
    }
    return stmt.on_conflict_do_update(index_elements=pk_cols, set_=update_cols)


class FeatureRepo(BaseRepo):
    def __init__(self, engine: Engine, schema: str | None = "cta") -> None:
        super().__init__(engine)
        self._table = _feature_table(schema)

    def upsert_many(self, rows: Sequence[FeatureRow]) -> int:
        stmt = _upsert_statement(self._table, rows)
        return self._execute_many(stmt, rows)


class SignalRepo(BaseRepo):
    def __init__(self, engine: Engine, schema: str | None = "cta") -> None:
        super().__init__(engine)
        self._table = _signal_table(schema)

    def upsert_many(self, rows: Sequence[SignalRow]) -> int:
        stmt = _upsert_statement(self._table, rows)
        return self._execute_many(stmt, rows)


class WeightRepo(BaseRepo):
    def __init__(self, engine: Engine, schema: str | None = "cta") -> None:
        super().__init__(engine)
        self._table = _weight_table(schema)

    def upsert_many(self, rows: Sequence[WeightRow]) -> int:
        stmt = _upsert_statement(self._table, rows)
        return self._execute_many(stmt, rows)


class RunRepo(BaseRepo):
    def __init__(self, engine: Engine, schema: str | None = "cta") -> None:
        super().__init__(engine)
        self._table = _job_run_table(schema)

    def upsert_many(self, rows: Sequence[JobRunRow]) -> int:
        stmt = _upsert_statement(self._table, rows)
        return self._execute_many(stmt, rows)

    def update_fields(self, run_id: str, fields: Mapping[str, object]) -> int:
        if not fields:
            return 0
        stmt = (
            update(self._table).where(self._table.c.run_id == run_id).values(**fields)
        )
        return self._execute(stmt)
