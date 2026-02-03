from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

from src.core.config import AppConfig
from src.services.auto_param_service import AutoParamService

# Sampled from production DB (public.trade_calendar / index_hist / market_factors / fund_beta)
# Window: 2019-01-02 to 2019-02-28
DATES = [
    "2019-01-02",
    "2019-01-03",
    "2019-01-04",
    "2019-01-07",
    "2019-01-08",
    "2019-01-09",
    "2019-01-10",
    "2019-01-11",
    "2019-01-14",
    "2019-01-15",
    "2019-01-16",
    "2019-01-17",
    "2019-01-18",
    "2019-01-21",
    "2019-01-22",
    "2019-01-23",
    "2019-01-24",
    "2019-01-25",
    "2019-01-28",
    "2019-01-29",
    "2019-01-30",
    "2019-01-31",
    "2019-02-01",
    "2019-02-11",
    "2019-02-12",
    "2019-02-13",
    "2019-02-14",
    "2019-02-15",
    "2019-02-18",
    "2019-02-19",
    "2019-02-20",
    "2019-02-21",
    "2019-02-22",
    "2019-02-25",
    "2019-02-26",
    "2019-02-27",
    "2019-02-28",
]

INDEX_ROWS = [
    ("000805.CSI", "2019-01-02", 2994.4657),
    ("000805.CSI", "2019-01-03", 3011.4385),
    ("000805.CSI", "2019-01-04", 3078.983),
    ("000805.CSI", "2019-01-07", 3128.4999),
    ("000805.CSI", "2019-01-08", 3098.0349),
    ("000805.CSI", "2019-01-09", 3094.6502),
    ("000805.CSI", "2019-01-10", 3066.2398),
    ("000805.CSI", "2019-01-11", 3099.2966),
    ("000805.CSI", "2019-01-14", 3041.4655),
    ("000805.CSI", "2019-01-15", 3050.1356),
    ("000805.CSI", "2019-01-16", 3030.1885),
    ("000805.CSI", "2019-01-17", 2965.3578),
    ("000805.CSI", "2019-01-18", 3007.5872),
    ("000805.CSI", "2019-01-21", 3074.7778),
    ("000805.CSI", "2019-01-22", 3042.2237),
    ("000805.CSI", "2019-01-23", 3038.2876),
    ("000805.CSI", "2019-01-24", 3075.2124),
    ("000805.CSI", "2019-01-25", 3127.5266),
    ("000805.CSI", "2019-01-28", 3086.2457),
    ("000805.CSI", "2019-01-29", 3038.1002),
    ("000805.CSI", "2019-01-30", 3061.0152),
    ("000805.CSI", "2019-01-31", 3070.0772),
    ("000805.CSI", "2019-02-01", 3084.5394),
    ("000805.CSI", "2019-02-11", 3231.144),
    ("000805.CSI", "2019-02-12", 3288.2025),
    ("000805.CSI", "2019-02-13", 3329.7148),
    ("000805.CSI", "2019-02-14", 3320.6911),
    ("000805.CSI", "2019-02-15", 3369.2154),
    ("000805.CSI", "2019-02-18", 3404.7167),
    ("000805.CSI", "2019-02-19", 3356.462),
    ("000805.CSI", "2019-02-20", 3438.6848),
    ("000805.CSI", "2019-02-21", 3492.9667),
    ("000805.CSI", "2019-02-22", 3532.8356),
    ("000805.CSI", "2019-02-25", 3643.2379),
    ("000805.CSI", "2019-02-26", 3613.2906),
    ("000805.CSI", "2019-02-27", 3644.6922),
    ("000805.CSI", "2019-02-28", 3565.9911),
    ("399006.SZ", "2019-01-02", 934.833),
    ("399006.SZ", "2019-01-03", 937.817),
    ("399006.SZ", "2019-01-04", 959.563),
    ("399006.SZ", "2019-01-07", 975.501),
    ("399006.SZ", "2019-01-08", 967.912),
    ("399006.SZ", "2019-01-09", 963.707),
    ("399006.SZ", "2019-01-10", 956.057),
    ("399006.SZ", "2019-01-11", 960.988),
    ("399006.SZ", "2019-01-14", 944.529),
    ("399006.SZ", "2019-01-15", 946.208),
    ("399006.SZ", "2019-01-16", 939.752),
    ("399006.SZ", "2019-01-17", 920.984),
    ("399006.SZ", "2019-01-18", 937.593),
    ("399006.SZ", "2019-01-21", 958.477),
    ("399006.SZ", "2019-01-22", 948.923),
    ("399006.SZ", "2019-01-23", 945.762),
    ("399006.SZ", "2019-01-24", 959.341),
    ("399006.SZ", "2019-01-25", 979.802),
    ("399006.SZ", "2019-01-28", 966.264),
    ("399006.SZ", "2019-01-29", 952.443),
    ("399006.SZ", "2019-01-30", 957.647),
    ("399006.SZ", "2019-01-31", 960.736),
    ("399006.SZ", "2019-02-01", 965.102),
    ("399006.SZ", "2019-02-11", 1027.001),
    ("399006.SZ", "2019-02-12", 1046.584),
    ("399006.SZ", "2019-02-13", 1061.592),
    ("399006.SZ", "2019-02-14", 1065.19),
    ("399006.SZ", "2019-02-15", 1085.96),
    ("399006.SZ", "2019-02-18", 1101.866),
    ("399006.SZ", "2019-02-19", 1089.283),
    ("399006.SZ", "2019-02-20", 1125.108),
    ("399006.SZ", "2019-02-21", 1146.283),
    ("399006.SZ", "2019-02-22", 1158.719),
    ("399006.SZ", "2019-02-25", 1205.389),
    ("399006.SZ", "2019-02-26", 1193.123),
    ("399006.SZ", "2019-02-27", 1200.255),
    ("399006.SZ", "2019-02-28", 1171.406),
]

FACTORS = [
    ("2019-01-02", -0.009609, 0.014014, -0.001802, -0.00194),
    ("2019-01-03", -0.003609, -0.001802, 0.017119, -0.000133),
    ("2019-01-04", 0.024549, 0.003195, 0.002898, -0.003373),
]

BETAS = [
    ("002236.OF", "2019-01-02", 0.8054588543597426, -0.0562055746865235),
    ("002236.OF", "2019-01-03", 0.8157371681544833, -0.046318618703427794),
    ("004744.OF", "2019-01-02", 0.4864391294525387, -0.2154098890942462),
]


@dataclass
class FakeBucketRepo:
    buckets: list[dict[str, object]]

    def get_range(self, bucket_ids: Sequence[int] | None = None) -> list[dict[str, object]]:
        return self.buckets


@dataclass
class FakeMarketRepo:
    rows: list[tuple[str, str, float]]

    def get_range(
        self,
        index_codes: Sequence[str],
        start_date: date,
        end_date: date,
        order_by_date: bool = True,
    ) -> list[dict[str, object]]:
        rows = []
        for code, dt, close in self.rows:
            if code not in index_codes:
                continue
            d = date.fromisoformat(dt)
            if start_date <= d <= end_date:
                rows.append({"index_code": code, "date": d, "close": close})
        if order_by_date:
            rows.sort(key=lambda r: (r["index_code"], r["date"]))
        return rows


@dataclass
class FakeFactorRepo:
    rows: list[tuple[str, float, float, float, float]]

    def get_range(
        self,
        start_date: date,
        end_date: date,
        order_by_date: bool = True,
    ) -> list[dict[str, object]]:
        rows = []
        for dt, mkt, smb, hml, qmj in self.rows:
            d = date.fromisoformat(dt)
            if start_date <= d <= end_date:
                rows.append({"date": d, "MKT": mkt, "SMB": smb, "HML": hml, "QMJ": qmj})
        if order_by_date:
            rows.sort(key=lambda r: r["date"])
        return rows


@dataclass
class FakeBetaRepo:
    rows: list[tuple[str, str, float, float]]

    def get_range(
        self,
        fund_codes: Sequence[str],
        start_date: date,
        end_date: date,
        order_by_date: bool = True,
    ) -> list[dict[str, object]]:
        rows = []
        for code, dt, smb, qmj in self.rows:
            if fund_codes and code not in fund_codes:
                continue
            d = date.fromisoformat(dt)
            if start_date <= d <= end_date:
                rows.append({"code": code, "date": d, "SMB": smb, "QMJ": qmj})
        if order_by_date:
            rows.sort(key=lambda r: (r["code"], r["date"]))
        return rows


@dataclass
class FakeCalendarRepo:
    rows: list[str]

    def get_range(self, start_date: date, end_date: date) -> list[dict[str, object]]:
        out = []
        for dt in self.rows:
            d = date.fromisoformat(dt)
            if start_date <= d <= end_date:
                out.append({"date": d})
        out.sort(key=lambda r: r["date"])
        return out


def test_auto_param_service_computes_and_persists(tmp_path: Path) -> None:
    config = AppConfig.model_validate(
        {
            "db": {"dsn": "postgresql://example", "schema_in": "public", "schema_out": "cta"},
            "auto_params": {"enabled": True, "window_years": 1, "min_points": 5},
            "features": {"short_window": 3, "long_window": 5, "vol_window": 3},
        }
    )
    buckets = [
        {
            "bucket_name": "GROWTH",
            "bucket_proxy": "399006.SZ",
            "assets": "002236.OF,004744.OF",
        },
        {
            "bucket_name": "CYCLE",
            "bucket_proxy": "000805.CSI",
            "assets": "002236.OF",
        },
    ]

    service = AutoParamService(
        bucket_repo=FakeBucketRepo(buckets),
        market_repo=FakeMarketRepo(INDEX_ROWS),
        factor_repo=FakeFactorRepo(FACTORS),
        beta_repo=FakeBetaRepo(BETAS),
        calendar_repo=FakeCalendarRepo(DATES),
        config=config,
        output_path=tmp_path / "auto_params.json",
    )

    result = service.compute_and_persist(as_of=date(2019, 2, 28))
    assert (tmp_path / "auto_params.json").exists()
    assert "features" in result.params
    assert "theta_on" in result.params["features"]
    assert "sigma_min" in result.params["features"]
    assert isinstance(result.params["features"]["theta_on"], dict)
    assert "GROWTH" in result.params["features"]["theta_on"]
    assert "signals" in result.params
    assert result.params["signals"]["tilt_scales"]


def test_auto_param_service_fallback_when_insufficient(tmp_path: Path) -> None:
    config = AppConfig.model_validate(
        {
            "db": {"dsn": "postgresql://example", "schema_in": "public", "schema_out": "cta"},
            "auto_params": {"enabled": True, "window_years": 1, "min_points": 9999},
        }
    )
    service = AutoParamService(
        bucket_repo=FakeBucketRepo([]),
        market_repo=FakeMarketRepo([]),
        factor_repo=FakeFactorRepo([]),
        beta_repo=FakeBetaRepo([]),
        calendar_repo=FakeCalendarRepo(DATES),
        config=config,
        output_path=tmp_path / "auto_params.json",
    )

    result = service.compute_and_persist(as_of=date(2019, 2, 28))
    assert result.used_fallback is True
    assert result.params["features"]["theta_on"] == config.features.theta_on


def test_apply_overrides_updates_config() -> None:
    config = AppConfig.model_validate(
        {
            "db": {"dsn": "postgresql://example", "schema_in": "public", "schema_out": "cta"},
            "auto_params": {"enabled": True, "window_years": 1, "min_points": 5},
        }
    )
    service = AutoParamService(
        bucket_repo=FakeBucketRepo([]),
        market_repo=FakeMarketRepo([]),
        factor_repo=FakeFactorRepo([]),
        beta_repo=FakeBetaRepo([]),
        calendar_repo=FakeCalendarRepo([]),
        config=config,
        output_path=Path("config/auto_params.json"),
    )
    params = {
        "features": {"theta_on": {"GROWTH": 0.8}, "sigma_min": {"GROWTH": 0.1}},
        "portfolio": {"sigma_target": 0.12},
        "signals": {"tilt_lookback_days": 45, "tilt_scales": {"SMB": 0.9}, "tilt_eps": 1e-9},
    }
    service.apply_overrides(config, params)
    assert isinstance(config.features.theta_on, dict)
    assert config.features.theta_on["GROWTH"] == 0.8
    assert config.portfolio.sigma_target == 0.12
    assert config.signals.tilt_lookback_days == 45
