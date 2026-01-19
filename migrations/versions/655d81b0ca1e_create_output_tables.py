"""create output tables

Revision ID: 655d81b0ca1e
Revises:
Create Date: 2026-01-19 16:53:25.259117

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "655d81b0ca1e"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE SCHEMA IF NOT EXISTS cta")

    op.create_table(
        "feature_daily",
        sa.Column("strategy_id", sa.String(length=64), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("instrument_id", sa.String(length=64), nullable=False),
        sa.Column("calc_date", sa.Date(), nullable=False),
        sa.Column("feature_name", sa.String(length=64), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("meta_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint(
            "strategy_id",
            "version",
            "instrument_id",
            "calc_date",
            "feature_name",
        ),
        schema="cta",
    )

    op.create_table(
        "signal_weekly",
        sa.Column("strategy_id", sa.String(length=64), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("instrument_id", sa.String(length=64), nullable=False),
        sa.Column("rebalance_date", sa.Date(), nullable=False),
        sa.Column("signal_name", sa.String(length=64), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("meta_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint(
            "strategy_id",
            "version",
            "instrument_id",
            "rebalance_date",
            "signal_name",
        ),
        schema="cta",
    )

    op.create_table(
        "portfolio_weight_weekly",
        sa.Column("strategy_id", sa.String(length=64), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("portfolio_id", sa.String(length=64), nullable=False),
        sa.Column("rebalance_date", sa.Date(), nullable=False),
        sa.Column("instrument_id", sa.String(length=64), nullable=False),
        sa.Column("target_weight", sa.Float(), nullable=False),
        sa.Column("bucket", sa.String(length=64), nullable=False),
        sa.Column("meta_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint(
            "strategy_id",
            "version",
            "portfolio_id",
            "rebalance_date",
            "instrument_id",
        ),
        schema="cta",
    )

    op.create_table(
        "job_run",
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("job_type", sa.String(length=32), nullable=False),
        sa.Column("strategy_id", sa.String(length=64), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("snapshot_id", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("time_start", sa.DateTime(), nullable=False),
        sa.Column("time_end", sa.DateTime(), nullable=True),
        sa.Column(
            "input_range_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "output_summary_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("error_stack", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("run_id"),
        schema="cta",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("job_run", schema="cta")
    op.drop_table("portfolio_weight_weekly", schema="cta")
    op.drop_table("signal_weekly", schema="cta")
    op.drop_table("feature_daily", schema="cta")
