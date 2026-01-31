"""add bucket_id to signal_weekly

Revision ID: 7c1f5b1a9b8a
Revises: 2c6b9b9a87d4
Create Date: 2026-01-31 00:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "7c1f5b1a9b8a"
down_revision: Union[str, Sequence[str], None] = "2c6b9b9a87d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "signal_weekly", sa.Column("bucket_id", sa.String(64), nullable=True), schema="cta"
    )
    op.execute(
        "UPDATE cta.signal_weekly "
        "SET bucket_id = (meta_json->>'bucket_id') "
        "WHERE bucket_id IS NULL AND meta_json IS NOT NULL"
    )


def downgrade() -> None:
    op.drop_column("signal_weekly", "bucket_id", schema="cta")
