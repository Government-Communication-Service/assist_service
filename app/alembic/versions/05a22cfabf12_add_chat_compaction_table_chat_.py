"""add chat_compaction table

Revision ID: 05a22cfabf12
Revises: 4fd83184f219
Create Date: 2026-08-24 14:36:55.188906

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "05a22cfabf12"
down_revision: Union[str, None] = "4fd83184f219"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chat_compaction",
        sa.Column("chat_id", sa.Integer(), nullable=False),
        sa.Column("up_to_message_id", sa.Integer(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("prefix_tokens_at_compaction", sa.Integer(), nullable=True),
        sa.Column("summary_tokens", sa.Integer(), nullable=True),
        sa.Column("llm_internal_response_id", sa.Integer(), nullable=True),
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("uuid", sa.UUID(), server_default=sa.text("uuid_generate_v4()"), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["chat_id"], ["chat.id"]),
        sa.ForeignKeyConstraint(["up_to_message_id"], ["message.id"]),
        sa.ForeignKeyConstraint(["llm_internal_response_id"], ["llm_internal_response.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_chat_compaction_chat_deleted_created",
        "chat_compaction",
        ["chat_id", "deleted_at", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_chat_compaction_chat_deleted_created", table_name="chat_compaction")
    op.drop_table("chat_compaction")
