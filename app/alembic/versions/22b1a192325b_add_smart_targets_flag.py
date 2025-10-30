"""Add smart targets data and sources column

Revision ID: 22b1a192325b
Revises: b463d4cca67d
Create Date: 2025-10-27 14:16:26.230827

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "22b1a192325b"
down_revision: Union[str, None] = "b463d4cca67d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("chat", sa.Column("use_smart_targets", sa.Boolean(), server_default="false", nullable=True))
    op.add_column("message", sa.Column("sources", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("chat", "use_smart_targets")
    op.drop_column("message", "sources")
