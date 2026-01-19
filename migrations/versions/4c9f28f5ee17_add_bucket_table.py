"""add bucket table

Revision ID: 4c9f28f5ee17
Revises: 655d81b0ca1e
Create Date: 2026-01-19 21:10:22.997592

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "4c9f28f5ee17"
down_revision: Union[str, Sequence[str], None] = "655d81b0ca1e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "bucket",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("bucket_name", sa.String(length=64), nullable=False),
        sa.Column("assets", sa.Text(), nullable=False),
        schema="cta",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("bucket", schema="cta")
