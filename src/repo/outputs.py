from __future__ import annotations

from datetime import date, datetime
from typing import Mapping, Sequence, TypedDict, cast

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    MetaData,
    String,
    Table,
    Text,
    and_,
    desc,
    or_,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.engine import Engine
from sqlalchemy.sql import Insert

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
    bucket_id: str | None
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


class FeatureWeeklySampleRow(TypedDict):
    strategy_id: str
    version: str
    instrument_id: str
    rebalance_date: date
    feature_name: str
    value: float
    meta_json: dict[str, object] | None


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
        Column("bucket_id", String(64)),
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


def _upsert_statement(
    table: Table,
    rows: Sequence[Mapping[str, object]],
) -> Insert:
    stmt = insert(table).values(rows)
    pk_cols = [col.name for col in table.primary_key.columns]
    update_cols = {
        col.name: stmt.excluded[col.name] for col in table.columns if col.name not in pk_cols
    }
    return stmt.on_conflict_do_update(index_elements=pk_cols, set_=update_cols)


class FeatureRepo(BaseRepo):
    def __init__(self, engine: Engine, schema: str | None = "cta") -> None:
        super().__init__(engine)
        self._table = _feature_table(schema)

    def upsert_many(self, rows: Sequence[FeatureRow]) -> int:
        stmt = _upsert_statement(self._table, rows)
        return self._execute_many(stmt, rows)


def _feature_weekly_sample_table(schema: str | None) -> Table:
    metadata = MetaData(schema=schema)
    return Table(
        "feature_weekly_sample",
        metadata,
        Column("strategy_id", String(64), primary_key=True),
        Column("version", String(32), primary_key=True),
        Column("instrument_id", String(64), primary_key=True),
        Column("rebalance_date", Date, primary_key=True),
        Column("feature_name", String(64), primary_key=True),
        Column("value", Float, nullable=False),
        Column("meta_json", JSONB, nullable=True),
    )


class FeatureWeeklySampleRepo(BaseRepo):
    def __init__(self, engine: Engine, schema: str | None = "cta") -> None:
        super().__init__(engine)
        self._table = _feature_weekly_sample_table(schema)

    def upsert_many(self, rows: Sequence[FeatureWeeklySampleRow]) -> int:
        stmt = _upsert_statement(self._table, rows)
        return self._execute_many(stmt, rows)

    def get_range(
        self,
        *,
        strategy_id: str,
        version: str,
        rebalance_date: date,
        instrument_ids: Sequence[str] | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> list[FeatureWeeklySampleRow]:
        stmt = (
            select(self._table)
            .where(self._table.c.strategy_id == strategy_id)
            .where(self._table.c.version == version)
            .where(self._table.c.rebalance_date == rebalance_date)
        )
        if instrument_ids:
            stmt = stmt.where(self._table.c.instrument_id.in_(instrument_ids))
        if feature_names:
            stmt = stmt.where(self._table.c.feature_name.in_(feature_names))
        return cast(list[FeatureWeeklySampleRow], self._fetch_all(stmt))


class SignalRepo(BaseRepo):
    def __init__(self, engine: Engine, schema: str | None = "cta") -> None:
        super().__init__(engine)
        self._table = _signal_table(schema)

    def upsert_many(self, rows: Sequence[SignalRow]) -> int:
        stmt = _upsert_statement(self._table, rows)
        return self._execute_many(stmt, rows)

    def get_range(
        self,
        *,
        strategy_id: str,
        version: str,
        rebalance_date: date,
        instrument_ids: Sequence[str] | None = None,
        signal_names: Sequence[str] | None = None,
        signal_name_prefix: str | None = None,
    ) -> list[SignalRow]:
        stmt = (
            select(self._table)
            .where(self._table.c.strategy_id == strategy_id)
            .where(self._table.c.version == version)
            .where(self._table.c.rebalance_date == rebalance_date)
        )
        if instrument_ids:
            stmt = stmt.where(self._table.c.instrument_id.in_(instrument_ids))
        if signal_names:
            stmt = stmt.where(self._table.c.signal_name.in_(signal_names))
        if signal_name_prefix:
            stmt = stmt.where(self._table.c.signal_name.like(f"{signal_name_prefix}%"))
        return cast(list[SignalRow], self._fetch_all(stmt))

    def get_latest(
        self,
        *,
        strategy_id: str,
        version: str,
        instrument_ids: Sequence[str] | None = None,
        signal_names: Sequence[str] | None = None,
    ) -> tuple[date, list[SignalRow]] | None:
        stmt = (
            select(self._table.c.rebalance_date)
            .where(self._table.c.strategy_id == strategy_id)
            .where(self._table.c.version == version)
            .order_by(desc(self._table.c.rebalance_date))
            .limit(1)
        )
        row = self._fetch_one(stmt)
        if not row:
            return None
        rebalance_date = cast(date, row["rebalance_date"])
        rows = self.get_range(
            strategy_id=strategy_id,
            version=version,
            rebalance_date=rebalance_date,
            instrument_ids=instrument_ids,
            signal_names=signal_names,
        )
        return rebalance_date, rows


class WeightRepo(BaseRepo):
    def __init__(self, engine: Engine, schema: str | None = "cta") -> None:
        super().__init__(engine)
        self._table = _weight_table(schema)

    def upsert_many(self, rows: Sequence[WeightRow]) -> int:
        stmt = _upsert_statement(self._table, rows)
        return self._execute_many(stmt, rows)

    def get_by_date(
        self,
        *,
        strategy_id: str,
        version: str,
        portfolio_id: str,
        rebalance_date: date,
    ) -> list[WeightRow]:
        stmt = (
            select(self._table)
            .where(self._table.c.strategy_id == strategy_id)
            .where(self._table.c.version == version)
            .where(self._table.c.portfolio_id == portfolio_id)
            .where(self._table.c.rebalance_date == rebalance_date)
        )
        return cast(list[WeightRow], self._fetch_all(stmt))

    def get_latest(
        self,
        *,
        strategy_id: str,
        version: str,
        portfolio_id: str,
    ) -> tuple[date, list[WeightRow]] | None:
        stmt = (
            select(self._table.c.rebalance_date)
            .where(self._table.c.strategy_id == strategy_id)
            .where(self._table.c.version == version)
            .where(self._table.c.portfolio_id == portfolio_id)
            .order_by(desc(self._table.c.rebalance_date))
            .limit(1)
        )
        row = self._fetch_one(stmt)
        if not row:
            return None
        rebalance_date = cast(date, row["rebalance_date"])
        rows = self.get_by_date(
            strategy_id=strategy_id,
            version=version,
            portfolio_id=portfolio_id,
            rebalance_date=rebalance_date,
        )
        return rebalance_date, rows

    def get_history(
        self,
        *,
        strategy_id: str,
        version: str,
        portfolio_id: str,
        start_date: date,
        end_date: date,
    ) -> list[WeightRow]:
        stmt = (
            select(self._table)
            .where(self._table.c.strategy_id == strategy_id)
            .where(self._table.c.version == version)
            .where(self._table.c.portfolio_id == portfolio_id)
            .where(self._table.c.rebalance_date >= start_date)
            .where(self._table.c.rebalance_date <= end_date)
            .order_by(self._table.c.rebalance_date, self._table.c.instrument_id)
        )
        return cast(list[WeightRow], self._fetch_all(stmt))


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
        stmt = update(self._table).where(self._table.c.run_id == run_id).values(**fields)
        return self._execute(stmt)

    def get_by_id(self, run_id: str) -> JobRunRow | None:
        stmt = select(self._table).where(self._table.c.run_id == run_id)
        row = self._fetch_one(stmt)
        return cast(JobRunRow | None, row)

    def list_runs(
        self,
        *,
        limit: int,
        cursor: tuple[datetime, str] | None = None,
        status: str | None = None,
        job_type: str | None = None,
        strategy_id: str | None = None,
        version: str | None = None,
    ) -> list[JobRunRow]:
        stmt = select(self._table)
        if status:
            stmt = stmt.where(self._table.c.status == status)
        if job_type:
            stmt = stmt.where(self._table.c.job_type == job_type)
        if strategy_id:
            stmt = stmt.where(self._table.c.strategy_id == strategy_id)
        if version:
            stmt = stmt.where(self._table.c.version == version)
        if cursor:
            time_start, run_id = cursor
            stmt = stmt.where(
                or_(
                    self._table.c.time_start < time_start,
                    and_(
                        self._table.c.time_start == time_start,
                        self._table.c.run_id < run_id,
                    ),
                )
            )
        stmt = stmt.order_by(desc(self._table.c.time_start), desc(self._table.c.run_id)).limit(
            limit
        )
        return cast(list[JobRunRow], self._fetch_all(stmt))

    def get_by_idempotency_key(
        self,
        *,
        idempotency_key: str,
        job_type: str,
        strategy_id: str,
        version: str,
    ) -> JobRunRow | None:
        tags = self._table.c.input_range_json["tags"]["idempotency_key"].astext
        stmt = (
            select(self._table)
            .where(self._table.c.job_type == job_type)
            .where(self._table.c.strategy_id == strategy_id)
            .where(self._table.c.version == version)
            .where(tags == idempotency_key)
            .order_by(desc(self._table.c.time_start))
            .limit(1)
        )
        row = self._fetch_one(stmt)
        return cast(JobRunRow | None, row)
