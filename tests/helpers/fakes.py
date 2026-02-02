from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Sequence


@dataclass
class FakeUpsertRepo:
    key_fields: Sequence[str]
    rows: dict[tuple[Any, ...], Mapping[str, Any]] = field(default_factory=dict)

    def upsert_many(self, rows: Sequence[Mapping[str, Any]]) -> int:
        for row in rows:
            key = tuple(row[field] for field in self.key_fields)
            self.rows[key] = dict(row)
        return len(rows)


@dataclass
class FakeRunRepo(FakeUpsertRepo):
    def update_fields(self, run_id: str, fields: Mapping[str, Any]) -> int:
        key = (run_id,)
        if key not in self.rows:
            return 0
        merged = dict(self.rows[key])
        merged.update(fields)
        self.rows[key] = merged
        return 1


@dataclass
class FakeBucketRepo:
    buckets: list[Mapping[str, Any]]

    def get_range(self, bucket_ids: list[int] | None) -> list[Mapping[str, Any]]:
        return list(self.buckets)


@dataclass
class FakeDateRepo:
    dates: list[date]

    def get_range(self, *_args, **_kwargs) -> list[Mapping[str, Any]]:
        return [{"date": d} for d in self.dates]

    def get_nth_before(self, anchor: date, days: int) -> date:
        _ = (anchor, days)
        return min(self.dates)


@dataclass
class FakeAuxRepo:
    dates: list[date]

    def get_gold_range(self, *_args, **_kwargs) -> list[Mapping[str, Any]]:
        return [{"date": d} for d in self.dates]
