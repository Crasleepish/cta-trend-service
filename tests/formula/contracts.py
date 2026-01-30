"""Formula test contracts (signatures only)."""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence


class TrendStrengthFn(Protocol):
    def __call__(
        self, prices: Sequence[float], short: int, long: int, vol_window: int
    ) -> float: ...


class HysteresisGateFn(Protocol):
    def __call__(self, trends: Iterable[float], theta_on: float, theta_off: float) -> list[int]: ...


class SigmaAnnualizedFn(Protocol):
    def __call__(self, returns: Iterable[float], window: int, annualize: int = 252) -> float: ...


class WeeklySamplerFn(Protocol):
    def __call__(self, dates: Iterable[object]) -> list[object]: ...
