from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

import src.services.backtest_service as service_mod
from src.services.backtest_service import BacktestRequest, BacktestService


class FakeWeightRepo:
    def __init__(self, rows):
        self._rows = rows

    def get_history(self, **_kwargs):
        return list(self._rows)


class FakeNavRepo:
    def __init__(self, rows):
        self._rows = rows

    def get_range(self, *_args, **_kwargs):
        return list(self._rows)


class FakeCalendarRepo:
    def __init__(self, dates):
        self._dates = dates

    def get_range(self, *_args, **_kwargs):
        return [{"date": d} for d in self._dates]


class DummySeries:
    def __init__(self, index: pd.Index):
        self._index = index

    @property
    def vbt(self):
        class _Plot:
            def __init__(self, index: pd.Index):
                self.index = index

            def plot(self, **_kwargs):
                class _Fig:
                    def write_html(self, _path: str) -> None:
                        Path(_path).write_text("<html></html>")

                return _Fig()

        return _Plot(self._index)


class DummyResult:
    def __init__(self, nav: pd.Series, returns: pd.Series):
        self.nav = nav
        self.returns = returns


def _fake_run_backtest(weights: pd.DataFrame, close: pd.DataFrame, cfg):
    nav = pd.Series([1.0, 1.01], index=close.index[:2])
    returns = pd.Series([0.0, 0.01], index=close.index[:2])
    nav = nav.copy()
    nav.__class__ = pd.Series  # ensure pandas series
    nav.vbt = DummySeries(nav.index).vbt  # type: ignore[attr-defined]
    return {
        "nav": nav,
        "returns": returns,
        "stats": {"Sharpe Ratio": 1.23},
    }


def test_backtest_service_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(service_mod, "run_backtest", _fake_run_backtest)

    weights_rows = [
        {
            "strategy_id": "s1",
            "version": "v1",
            "portfolio_id": "p1",
            "rebalance_date": date(2019, 1, 4),
            "instrument_id": "004253.OF",
            "target_weight": 1.0,
            "bucket": "GOLD",
            "meta_json": None,
        }
    ]
    nav_rows = [
        {"fund_code": "004253.OF", "date": date(2019, 1, 2), "net_value": 1.0808},
        {"fund_code": "004253.OF", "date": date(2019, 1, 3), "net_value": 1.0864},
        {"fund_code": "004253.OF", "date": date(2019, 1, 4), "net_value": 1.0876},
    ]
    calendar = [date(2019, 1, 2), date(2019, 1, 3), date(2019, 1, 4)]

    service = BacktestService(
        weight_repo=FakeWeightRepo(weights_rows),
        nav_repo=FakeNavRepo(nav_rows),
        calendar_repo=FakeCalendarRepo(calendar),
    )

    req = BacktestRequest(
        start_date=date(2019, 1, 2),
        end_date=date(2019, 1, 4),
        strategy_id="s1",
        version="v1",
        portfolio_id="p1",
        output_dir=tmp_path,
    )

    result = service.run_backtest_task(req)

    assert result.artifacts.weights_csv.exists()
    assert result.artifacts.nav_returns_csv.exists()
    assert result.artifacts.equity_curve_html.exists()
    assert result.artifacts.report_md.exists()

    weights = pd.read_csv(result.artifacts.weights_csv)
    assert list(weights.columns) == ["rebalance_date", "004253.OF"]
    assert weights.iloc[0]["004253.OF"] == 1.0


def test_backtest_service_drops_missing_assets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(service_mod, "run_backtest", _fake_run_backtest)

    weights_rows = [
        {
            "strategy_id": "s1",
            "version": "v1",
            "portfolio_id": "p1",
            "rebalance_date": date(2019, 1, 4),
            "instrument_id": "A",
            "target_weight": 0.6,
            "bucket": "B1",
            "meta_json": None,
        },
        {
            "strategy_id": "s1",
            "version": "v1",
            "portfolio_id": "p1",
            "rebalance_date": date(2019, 1, 4),
            "instrument_id": "B",
            "target_weight": 0.4,
            "bucket": "B2",
            "meta_json": None,
        },
    ]
    nav_rows = [
        {"fund_code": "A", "date": date(2019, 1, 2), "net_value": 1.0},
        {"fund_code": "A", "date": date(2019, 1, 3), "net_value": 1.01},
        {"fund_code": "A", "date": date(2019, 1, 4), "net_value": 1.02},
    ]
    calendar = [date(2019, 1, 2), date(2019, 1, 3), date(2019, 1, 4)]

    service = BacktestService(
        weight_repo=FakeWeightRepo(weights_rows),
        nav_repo=FakeNavRepo(nav_rows),
        calendar_repo=FakeCalendarRepo(calendar),
    )

    req = BacktestRequest(
        start_date=date(2019, 1, 2),
        end_date=date(2019, 1, 4),
        strategy_id="s1",
        version="v1",
        portfolio_id="p1",
        output_dir=tmp_path,
    )

    result = service.run_backtest_task(req)
    assert result.warnings
    assert "B" in result.warnings[0]

    weights = pd.read_csv(result.artifacts.weights_csv)
    assert "B" not in weights.columns
