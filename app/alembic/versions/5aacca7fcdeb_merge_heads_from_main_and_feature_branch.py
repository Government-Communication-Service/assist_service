"""merge heads from main and feature branch

Revision ID: 5aacca7fcdeb
Revises: 48e4e44b6f62, fba39601e3d6
Create Date: 2025-11-24 09:36:13.471367

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "5aacca7fcdeb"
down_revision: Union[str, None] = ("48e4e44b6f62", "fba39601e3d6")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
