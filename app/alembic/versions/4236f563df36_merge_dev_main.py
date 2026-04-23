"""merge_dev_main

Revision ID: 4236f563df36
Revises: 1d082f609ec7, 87807b4451a2
Create Date: 2026-02-20 16:02:56.285477

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "4236f563df36"
down_revision: Union[str, None] = ("1d082f609ec7", "87807b4451a2")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
