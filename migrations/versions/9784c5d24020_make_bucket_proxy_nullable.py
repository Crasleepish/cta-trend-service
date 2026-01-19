"""make bucket_proxy nullable

Revision ID: 9784c5d24020
Revises: b95bfe62110c
Create Date: 2026-01-19 22:58:12.626293

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9784c5d24020"
down_revision: Union[str, Sequence[str], None] = "b95bfe62110c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column(
        "bucket",
        "bucket_proxy",
        existing_type=sa.String(length=64),
        nullable=True,
        schema="cta",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column(
        "bucket",
        "bucket_proxy",
        existing_type=sa.String(length=64),
        nullable=False,
        schema="cta",
    )
