"""add message raw_messages

Revision ID: a8faf7d10994
Revises: 2c48faa99745
Create Date: 2026-09-28 10:04:32.996575

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a8faf7d10994"
down_revision: Union[str, None] = "2c48faa99745"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("message", sa.Column("raw_messages", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("message", "raw_messages")
