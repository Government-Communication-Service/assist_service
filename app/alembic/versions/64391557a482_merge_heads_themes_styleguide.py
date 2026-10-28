"""merge_heads_themes_styleguide

Revision ID: 64391557a482
Revises: 4236f563df36, 70852de1f37e
Create Date: 2026-04-08 17:50:02.253638

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "64391557a482"
down_revision: Union[str, None] = ("4236f563df36", "70852de1f37e")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
