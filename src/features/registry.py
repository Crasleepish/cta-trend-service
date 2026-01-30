from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Iterable, Mapping

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    frequency: str
    dependencies: tuple[str, ...]
    compute: Callable[["DataBundle", Mapping[str, float]], pd.DataFrame]


@dataclass(frozen=True)
class DataBundle:
    prices: pd.DataFrame
    calendar: pd.DatetimeIndex
    rebalance_date: date


class FeatureRegistry:
    def __init__(self, specs: Iterable[FeatureSpec]) -> None:
        self._specs = {spec.name: spec for spec in specs}

    def get(self, name: str) -> FeatureSpec:
        return self._specs[name]

    def list_features(self) -> list[str]:
        return sorted(self._specs.keys())
