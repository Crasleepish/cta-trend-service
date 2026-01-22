from __future__ import annotations

from sqlalchemy import BigInteger, Column, Date, Float, LargeBinary, MetaData, String, Table
from sqlalchemy.engine import Engine


def create_fund_beta_table(engine: Engine, schema: str = "public") -> Table:
    metadata = MetaData(schema=schema)
    table = Table(
        "fund_beta",
        metadata,
        Column("code", String(64), primary_key=True),
        Column("date", Date, primary_key=True),
        Column("MKT", Float),
        Column("SMB", Float),
        Column("HML", Float),
        Column("QMJ", Float),
        Column("const", Float),
        Column("gamma", Float),
        Column("P_bin", LargeBinary),
    )
    metadata.create_all(engine)
    return table


def create_index_hist_table(engine: Engine, schema: str = "public") -> Table:
    metadata = MetaData(schema=schema)
    table = Table(
        "index_hist",
        metadata,
        Column("index_code", String(64), primary_key=True),
        Column("date", Date, primary_key=True),
        Column("close", Float),
    )
    metadata.create_all(engine)
    return table


def create_fund_hist_table(engine: Engine, schema: str = "public") -> Table:
    metadata = MetaData(schema=schema)
    table = Table(
        "fund_hist",
        metadata,
        Column("fund_code", String(64), primary_key=True),
        Column("date", Date, primary_key=True),
        Column("net_value", Float),
    )
    metadata.create_all(engine)
    return table


def create_bucket_table(engine: Engine, schema: str = "cta") -> Table:
    metadata = MetaData(schema=schema)
    table = Table(
        "bucket",
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("bucket_name", String(64), nullable=False),
        Column("assets", String(256), nullable=False),
        Column("bucket_proxy", String(64)),
        Column("bucket_proxy_name", String(128)),
    )
    metadata.create_all(engine)
    return table
