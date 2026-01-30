from __future__ import annotations

from sqlalchemy import Column, Date, DateTime, Float, MetaData, String, Table, Text
from sqlalchemy.dialects.postgresql import JSONB


def build_metadata(schema: str | None) -> MetaData:
    metadata = MetaData(schema=schema)

    Table(
        "feature_daily",
        metadata,
        Column("strategy_id", String(64), primary_key=True),
        Column("version", String(32), primary_key=True),
        Column("instrument_id", String(64), primary_key=True),
        Column("calc_date", Date, primary_key=True),
        Column("feature_name", String(64), primary_key=True),
        Column("value", Float, nullable=False),
        Column("meta_json", JSONB, nullable=True),
    )

    Table(
        "feature_weekly_sample",
        metadata,
        Column("strategy_id", String(64), primary_key=True),
        Column("version", String(32), primary_key=True),
        Column("instrument_id", String(64), primary_key=True),
        Column("rebalance_date", Date, primary_key=True),
        Column("feature_name", String(64), primary_key=True),
        Column("value", Float, nullable=False),
        Column("meta_json", JSONB, nullable=True),
    )

    Table(
        "signal_weekly",
        metadata,
        Column("strategy_id", String(64), primary_key=True),
        Column("version", String(32), primary_key=True),
        Column("instrument_id", String(64), primary_key=True),
        Column("rebalance_date", Date, primary_key=True),
        Column("signal_name", String(64), primary_key=True),
        Column("value", Float, nullable=False),
        Column("meta_json", JSONB, nullable=True),
    )

    Table(
        "portfolio_weight_weekly",
        metadata,
        Column("strategy_id", String(64), primary_key=True),
        Column("version", String(32), primary_key=True),
        Column("portfolio_id", String(64), primary_key=True),
        Column("rebalance_date", Date, primary_key=True),
        Column("instrument_id", String(64), primary_key=True),
        Column("target_weight", Float, nullable=False),
        Column("bucket", String(64), nullable=False),
        Column("meta_json", JSONB, nullable=True),
    )

    Table(
        "job_run",
        metadata,
        Column("run_id", String(64), primary_key=True),
        Column("job_type", String(32), nullable=False),
        Column("strategy_id", String(64), nullable=False),
        Column("version", String(32), nullable=False),
        Column("snapshot_id", String(64), nullable=True),
        Column("status", String(16), nullable=False),
        Column("time_start", DateTime, nullable=False),
        Column("time_end", DateTime, nullable=True),
        Column("input_range_json", JSONB, nullable=True),
        Column("output_summary_json", JSONB, nullable=True),
        Column("error_stack", Text, nullable=True),
    )

    return metadata
