from __future__ import annotations

from datetime import date
from typing import Sequence, TypedDict, cast

from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    Float,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    select,
)
from sqlalchemy.engine import Engine
from .base import BaseRepo


class BucketRow(TypedDict):
    id: int
    bucket_name: str
    assets: str


class IndexOhlcRow(TypedDict):
    index_code: str
    date: date
    open: float | None
    close: float | None
    high: float | None
    low: float | None
    volume: int | None
    amount: float | None
    change_percent: float | None
    change: float | None


class FactorRow(TypedDict):
    date: date
    MKT: float | None
    SMB: float | None
    HML: float | None
    QMJ: float | None


class FundNavRow(TypedDict):
    fund_code: str
    date: date
    net_value: float | None


class FundBetaRow(TypedDict):
    code: str
    date: date
    MKT: float | None
    SMB: float | None
    HML: float | None
    QMJ: float | None
    const: float | None


class AuxGoldRow(TypedDict):
    index_code: str
    date: date
    open: float | None
    close: float | None
    high: float | None
    low: float | None
    volume: int | None
    amount: float | None
    change_percent: float | None
    change: float | None


def _bucket_table(schema: str | None) -> Table:
    metadata = MetaData(schema=schema)
    return Table(
        "bucket",
        metadata,
        Column("id", BigInteger, primary_key=True),
        Column("bucket_name", String(64), nullable=False),
        Column("assets", Text, nullable=False),
    )


def _index_hist_table(schema: str | None) -> Table:
    metadata = MetaData(schema=schema)
    return Table(
        "index_hist",
        metadata,
        Column("index_code", String(64), primary_key=True),
        Column("date", Date, primary_key=True),
        Column("open", Float),
        Column("close", Float),
        Column("high", Float),
        Column("low", Float),
        Column("volume", BigInteger),
        Column("amount", Float),
        Column("change_percent", Float),
        Column("change", Float),
    )


def _market_factors_table(schema: str | None) -> Table:
    metadata = MetaData(schema=schema)
    return Table(
        "market_factors",
        metadata,
        Column("date", Date, primary_key=True),
        Column("MKT", Numeric),
        Column("SMB", Numeric),
        Column("HML", Numeric),
        Column("QMJ", Numeric),
    )


def _fund_hist_table(schema: str | None) -> Table:
    metadata = MetaData(schema=schema)
    return Table(
        "fund_hist",
        metadata,
        Column("fund_code", String(64), primary_key=True),
        Column("date", Date, primary_key=True),
        Column("net_value", Float),
    )


def _fund_beta_table(schema: str | None) -> Table:
    metadata = MetaData(schema=schema)
    return Table(
        "fund_beta",
        metadata,
        Column("code", String(64), primary_key=True),
        Column("date", Date, primary_key=True),
        Column("MKT", Float),
        Column("SMB", Float),
        Column("HML", Float),
        Column("QMJ", Float),
        Column("const", Float),
    )


class BucketRepo(BaseRepo):
    def __init__(self, engine: Engine, schema: str | None = "cta") -> None:
        super().__init__(engine)
        self._table = _bucket_table(schema)

    def get_range(self, bucket_ids: Sequence[int] | None = None) -> list[BucketRow]:
        stmt = select(self._table)
        if bucket_ids:
            if len(bucket_ids) == 1:
                stmt = stmt.where(self._table.c.id == bucket_ids[0])
            else:
                stmt = stmt.where(self._table.c.id.in_(bucket_ids))
        return cast(list[BucketRow], self._fetch_all(stmt))


class MarketRepo(BaseRepo):
    def __init__(self, engine: Engine, schema: str | None = "public") -> None:
        super().__init__(engine)
        self._table = _index_hist_table(schema)

    def get_range(
        self,
        index_codes: Sequence[str],
        start_date: date,
        end_date: date,
        order_by_date: bool = True,
    ) -> list[IndexOhlcRow]:
        stmt = (
            select(self._table)
            .where(self._table.c.index_code.in_(index_codes))
            .where(self._table.c.date.between(start_date, end_date))
        )
        if order_by_date:
            stmt = stmt.order_by(self._table.c.date.asc())
        return cast(list[IndexOhlcRow], self._fetch_all(stmt))


class FactorRepo(BaseRepo):
    def __init__(self, engine: Engine, schema: str | None = "public") -> None:
        super().__init__(engine)
        self._table = _market_factors_table(schema)

    def get_range(
        self,
        start_date: date,
        end_date: date,
        order_by_date: bool = True,
    ) -> list[FactorRow]:
        stmt = select(self._table).where(
            self._table.c.date.between(start_date, end_date)
        )
        if order_by_date:
            stmt = stmt.order_by(self._table.c.date.asc())
        return cast(list[FactorRow], self._fetch_all(stmt))


class NavRepo(BaseRepo):
    def __init__(self, engine: Engine, schema: str | None = "public") -> None:
        super().__init__(engine)
        self._table = _fund_hist_table(schema)

    def get_range(
        self,
        fund_codes: Sequence[str],
        start_date: date,
        end_date: date,
        order_by_date: bool = True,
    ) -> list[FundNavRow]:
        stmt = (
            select(self._table)
            .where(self._table.c.fund_code.in_(fund_codes))
            .where(self._table.c.date.between(start_date, end_date))
        )
        if order_by_date:
            stmt = stmt.order_by(self._table.c.date.asc())
        return cast(list[FundNavRow], self._fetch_all(stmt))


class BetaRepo(BaseRepo):
    def __init__(self, engine: Engine, schema: str | None = "public") -> None:
        super().__init__(engine)
        self._table = _fund_beta_table(schema)

    def get_range(
        self,
        fund_codes: Sequence[str],
        start_date: date,
        end_date: date,
        order_by_date: bool = True,
    ) -> list[FundBetaRow]:
        stmt = (
            select(self._table)
            .where(self._table.c.code.in_(fund_codes))
            .where(self._table.c.date.between(start_date, end_date))
        )
        if order_by_date:
            stmt = stmt.order_by(self._table.c.date.asc())
        return cast(list[FundBetaRow], self._fetch_all(stmt))


class AuxRepo(BaseRepo):
    _gold_code: str = "Au99.99.SGE"

    def __init__(self, engine: Engine, schema: str | None = "public") -> None:
        super().__init__(engine)
        self._index_table = _index_hist_table(schema)

    def get_gold_range(
        self,
        start_date: date,
        end_date: date,
        order_by_date: bool = True,
    ) -> list[AuxGoldRow]:
        stmt = (
            select(self._index_table)
            .where(self._index_table.c.index_code == self._gold_code)
            .where(self._index_table.c.date.between(start_date, end_date))
        )
        if order_by_date:
            stmt = stmt.order_by(self._index_table.c.date.asc())
        return cast(list[AuxGoldRow], self._fetch_all(stmt))

    def get_usd_index_range(
        self,
        start_date: date,
        end_date: date,
        order_by_date: bool = True,
    ) -> list[dict[str, object]]:
        return []
