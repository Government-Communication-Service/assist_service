"""add notified_at to chat_share_user_mapping

Revision ID: c7e2a91d5b38
Revises: b3d1f7c42a90
Create Date: 2026-07-12 10:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c7e2a91d5b38"
down_revision: Union[str, None] = "b3d1f7c42a90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("chat_share_user_mapping", sa.Column("notified_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("chat_share_user_mapping", "notified_at")
