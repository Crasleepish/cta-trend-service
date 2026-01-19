from __future__ import annotations

from datetime import date

from sqlalchemy import BigInteger, Column, Date, Float, MetaData, String, Table, text

from src.repo.inputs import (
    AuxRepo,
    BetaRepo,
    BucketRepo,
    FactorRepo,
    MarketRepo,
    NavRepo,
)


def _setup_tables(engine) -> None:
    metadata_public = MetaData()
    metadata_cta = MetaData(schema="cta")
    Table(
        "bucket",
        metadata_cta,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("bucket_name", String(64), nullable=False),
        Column("assets", String(256), nullable=False),
        Column("bucket_proxy", String(64), nullable=False),
        Column("bucket_proxy_name", String(128)),
    )
    Table(
        "index_hist",
        metadata_public,
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
    Table(
        "market_factors",
        metadata_public,
        Column("date", Date, primary_key=True),
        Column("MKT", Float),
        Column("SMB", Float),
        Column("HML", Float),
        Column("QMJ", Float),
    )
    Table(
        "fund_hist",
        metadata_public,
        Column("fund_code", String(64), primary_key=True),
        Column("date", Date, primary_key=True),
        Column("net_value", Float),
    )
    Table(
        "fund_beta",
        metadata_public,
        Column("code", String(64), primary_key=True),
        Column("date", Date, primary_key=True),
        Column("MKT", Float),
        Column("SMB", Float),
        Column("HML", Float),
        Column("QMJ", Float),
        Column("const", Float),
    )
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS cta"))
    metadata_public.create_all(engine)
    metadata_cta.create_all(engine)


def test_input_repos_get_range(pg_engine) -> None:
    _setup_tables(pg_engine)

    with pg_engine.begin() as conn:
        conn.execute(
            Table("bucket", MetaData(schema="cta"), autoload_with=pg_engine).insert(),
            [
                {
                    "id": 1,
                    "bucket_name": "GROWTH",
                    "assets": "F1,F2",
                    "bucket_proxy": "IDX_GROWTH",
                    "bucket_proxy_name": "Growth Index",
                }
            ],
        )
        conn.execute(
            Table("index_hist", MetaData(), autoload_with=pg_engine).insert(),
            [
                {
                    "index_code": "Au99.99.SGE",
                    "date": date(2026, 1, 1),
                    "open": 1.0,
                    "close": 1.1,
                    "high": 1.2,
                    "low": 0.9,
                    "volume": 100,
                    "amount": 200.0,
                    "change_percent": 0.1,
                    "change": 0.01,
                }
            ],
        )
        conn.execute(
            Table("market_factors", MetaData(), autoload_with=pg_engine).insert(),
            [
                {
                    "date": date(2026, 1, 1),
                    "MKT": 0.1,
                    "SMB": 0.2,
                    "HML": 0.3,
                    "QMJ": 0.4,
                }
            ],
        )
        conn.execute(
            Table("fund_hist", MetaData(), autoload_with=pg_engine).insert(),
            [{"fund_code": "F1", "date": date(2026, 1, 1), "net_value": 1.23}],
        )
        conn.execute(
            Table("fund_beta", MetaData(), autoload_with=pg_engine).insert(),
            [
                {
                    "code": "F1",
                    "date": date(2026, 1, 1),
                    "MKT": 0.1,
                    "SMB": 0.2,
                    "HML": 0.3,
                    "QMJ": 0.4,
                    "const": 1.0,
                }
            ],
        )

    bucket_repo = BucketRepo(pg_engine, schema="cta")
    market_repo = MarketRepo(pg_engine, schema="public")
    factor_repo = FactorRepo(pg_engine, schema="public")
    nav_repo = NavRepo(pg_engine, schema="public")
    beta_repo = BetaRepo(pg_engine, schema="public")
    aux_repo = AuxRepo(pg_engine, schema="public")

    assert bucket_repo.get_range([1])[0]["bucket_name"] == "GROWTH"
    assert market_repo.get_range(["Au99.99.SGE"], date(2026, 1, 1), date(2026, 1, 1))
    assert factor_repo.get_range(date(2026, 1, 1), date(2026, 1, 1))
    assert nav_repo.get_range(["F1"], date(2026, 1, 1), date(2026, 1, 1))
    assert beta_repo.get_range(["F1"], date(2026, 1, 1), date(2026, 1, 1))
    assert aux_repo.get_gold_range(date(2026, 1, 1), date(2026, 1, 1))
    assert aux_repo.get_usd_index_range(date(2026, 1, 1), date(2026, 1, 1)) == []
