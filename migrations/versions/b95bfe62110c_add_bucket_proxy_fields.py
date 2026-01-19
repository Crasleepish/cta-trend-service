"""add bucket proxy fields

Revision ID: b95bfe62110c
Revises: 4c9f28f5ee17
Create Date: 2026-01-19 22:38:46.133136

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b95bfe62110c"
down_revision: Union[str, Sequence[str], None] = "4c9f28f5ee17"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "bucket",
        sa.Column("bucket_proxy", sa.String(length=64), nullable=False),
        schema="cta",
    )
    op.add_column(
        "bucket",
        sa.Column("bucket_proxy_name", sa.String(length=128), nullable=True),
        schema="cta",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("bucket", "bucket_proxy_name", schema="cta")
    op.drop_column("bucket", "bucket_proxy", schema="cta")
