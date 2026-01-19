from __future__ import annotations

from typing import Mapping, Sequence

from sqlalchemy.engine import Engine
from sqlalchemy.sql import Executable


class BaseRepo:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def _fetch_all(self, stmt: Executable) -> list[dict[str, object]]:
        with self._engine.connect() as connection:
            result = connection.execute(stmt)
            return [dict(row._mapping) for row in result]

    def _execute(self, stmt: Executable) -> int:
        with self._engine.begin() as connection:
            result = connection.execute(stmt)
            return int(result.rowcount or 0)

    def _execute_many(self, stmt: Executable, rows: Sequence[Mapping[str, object]]) -> int:
        if not rows:
            return 0
        with self._engine.begin() as connection:
            result = connection.execute(stmt, rows)
            return int(result.rowcount or 0)
