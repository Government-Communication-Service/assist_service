"""migration message

Revision ID: 518255ab4b99
Revises: 8ca7db7bfc4c
Create Date: 2024-11-04 11:49:45.622658

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "518255ab4b99"
down_revision: Union[str, None] = "8ea05d0d078e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("document", sa.Column("is_central", sa.Boolean(), server_default=sa.false(), nullable=False))
    op.execute(
        "UPDATE document SET is_central = true WHERE "
        "name in ("
        "'The Modern Communications Operating Model (MCOM) 3.0', "
        "'Accessible by default', 'The Labour Manifesto 2024', "
        "'Inclusive Communications Template', "
        "'British Sign Language Act and guidance')"
        ""
    )

    op.create_index("ix_document_is_central", "document", ["is_central"], unique=False)

    # expired_at field changes
    op.add_column(
        "document_user_mapping",
        sa.Column("expired_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP + INTERVAL '90 days'")),
    )
    op.create_index("idx_document_user_mapping_expired_at", "document_user_mapping", ["expired_at"])

    # update existing records
    op.execute("UPDATE document_user_mapping SET expired_at = created_at + interval '90 days';")

    # make  expired_at column non-null
    op.execute("ALTER TABLE document_user_mapping ALTER COLUMN expired_at SET NOT NULL;")

    # ### end Alembic commands ###


def downgrade() -> None:
    op.drop_column("document", "is_central")

    # expired_at field drop
    op.drop_column("document_user_mapping", "expired_at")
    # ### end Alembic commands ###
