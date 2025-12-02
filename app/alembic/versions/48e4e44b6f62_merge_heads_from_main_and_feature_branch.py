"""merge heads from main and feature branch

Revision ID: 48e4e44b6f62
Revises: 2a0b079dc0b2, 887fa86569b8
Create Date: 2025-11-24 08:19:37.017470

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "48e4e44b6f62"
down_revision: Union[str, None] = ("2a0b079dc0b2", "887fa86569b8")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
