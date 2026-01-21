from __future__ import annotations

from sqlalchemy import Column, Date, Float, LargeBinary, MetaData, String, Table
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
