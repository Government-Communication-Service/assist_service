"""add chat_compaction_lock table, add compaction cache columns

Revision ID: 1818b315958c
Revises: 05a22cfabf12
Create Date: 2026-08-25 12:42:25.158240

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1818b315958c"
down_revision: Union[str, None] = "05a22cfabf12"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("chat_compaction", sa.Column("summary_cache_read_tokens", sa.Integer(), nullable=True))
    op.add_column("chat_compaction", sa.Column("summary_cache_write_tokens", sa.Integer(), nullable=True))

    op.create_table(
        "chat_compaction_lock",
        sa.Column("chat_id", sa.Integer(), nullable=False),
        sa.Column("compaction_lock", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("locked_at", sa.DateTime(), nullable=True),
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("uuid", sa.UUID(), server_default=sa.text("uuid_generate_v4()"), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["chat_id"], ["chat.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chat_id", name="uq_chat_compaction_lock_chat_id"),
    )


def downgrade() -> None:
    op.drop_table("chat_compaction_lock")

    op.drop_column("chat_compaction", "summary_cache_write_tokens")
    op.drop_column("chat_compaction", "summary_cache_read_tokens")
