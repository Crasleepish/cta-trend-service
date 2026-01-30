"""add feature_weekly_sample

Revision ID: 2c6b9b9a87d4
Revises: 9784c5d24020
Create Date: 2026-01-30
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "2c6b9b9a87d4"
down_revision = "9784c5d24020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "feature_weekly_sample",
        sa.Column("strategy_id", sa.String(length=64), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("instrument_id", sa.String(length=64), nullable=False),
        sa.Column("rebalance_date", sa.Date(), nullable=False),
        sa.Column("feature_name", sa.String(length=64), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("meta_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint(
            "strategy_id",
            "version",
            "instrument_id",
            "rebalance_date",
            "feature_name",
        ),
        schema="cta",
    )


def downgrade() -> None:
    op.drop_table("feature_weekly_sample", schema="cta")
