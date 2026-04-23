"""added banner fields to theme and use cases

Revision ID: 1d082f609ec7
Revises: afc2607e4aaf
Create Date: 2026-02-03 17:11:38.135932

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1d082f609ec7"
down_revision: Union[str, None] = "afc2607e4aaf"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add banner columns to theme table
    op.add_column(
        "theme",
        sa.Column("show_update_banner", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.add_column(
        "theme",
        sa.Column("banner_type", sa.String(20), nullable=True),
    )
    op.add_column(
        "theme",
        sa.Column("banner_until", sa.DateTime(), nullable=True),
    )

    # Add banner columns to use_case table
    op.add_column(
        "use_case",
        sa.Column("show_update_banner", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.add_column(
        "use_case",
        sa.Column("banner_type", sa.String(20), nullable=True),
    )
    op.add_column(
        "use_case",
        sa.Column("banner_until", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    # Remove banner columns from use_case table
    op.drop_column("use_case", "banner_until")
    op.drop_column("use_case", "banner_type")
    op.drop_column("use_case", "show_update_banner")

    # Remove banner columns from theme table
    op.drop_column("theme", "banner_until")
    op.drop_column("theme", "banner_type")
    op.drop_column("theme", "show_update_banner")
