from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from sqlalchemy import Column, Integer, MetaData, Table
from sqlalchemy.dialects.postgresql import dialect

from src.repo.base import BaseRepo
from src.repo.outputs import _upsert_statement


@dataclass
class _ExecuteCall:
    stmt: object
    rows: Sequence[Mapping[str, object]]


class _FakeConnection:
    def __init__(self) -> None:
        self.calls: list[_ExecuteCall] = []
        self._rowcount = 0

    def execute(self, stmt: object, rows: Sequence[Mapping[str, object]]) -> object:
        self.calls.append(_ExecuteCall(stmt=stmt, rows=rows))
        self._rowcount = len(rows)

        class _Result:
            def __init__(self, rowcount: int) -> None:
                self.rowcount = rowcount

        return _Result(self._rowcount)

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeEngine:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    def begin(self) -> _FakeConnection:
        return self._connection


class _FakeRepo(BaseRepo):
    pass


def test_upsert_statement_does_not_embed_rows() -> None:
    metadata = MetaData()
    table = Table(
        "t",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("value", Integer),
    )
    stmt = _upsert_statement(table)
    compiled = stmt.compile(dialect=dialect())
    assert compiled.params == {"id": None, "value": None}


def test_execute_many_runs_once_with_rows() -> None:
    connection = _FakeConnection()
    engine = _FakeEngine(connection)
    repo = _FakeRepo(engine)  # type: ignore[arg-type]

    metadata = MetaData()
    table = Table(
        "t",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("value", Integer),
    )
    stmt = _upsert_statement(table)
    rows = [{"id": 1, "value": 10}, {"id": 2, "value": 20}]
    count = repo._execute_many(stmt, rows)

    assert count == len(rows)
    assert len(connection.calls) == 1
    assert connection.calls[0].rows == rows
