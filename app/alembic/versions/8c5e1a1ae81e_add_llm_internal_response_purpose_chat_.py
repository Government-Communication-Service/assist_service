"""add llm_internal_response purpose chat_id message_id

Revision ID: 8c5e1a1ae81e
Revises: 3d16b7cc8911
Create Date: 2026-09-29 13:59:58.258394

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8c5e1a1ae81e"
down_revision: Union[str, None] = "3d16b7cc8911"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("llm_internal_response", sa.Column("purpose", sa.Text(), nullable=True))
    op.add_column("llm_internal_response", sa.Column("chat_id", sa.Integer(), nullable=True))
    op.add_column("llm_internal_response", sa.Column("message_id", sa.Integer(), nullable=True))
    op.create_foreign_key("llm_internal_response_chat_id_fkey", "llm_internal_response", "chat", ["chat_id"], ["id"])
    op.create_foreign_key(
        "llm_internal_response_message_id_fkey", "llm_internal_response", "message", ["message_id"], ["id"]
    )
    op.create_index("ix_llm_internal_response_chat_id", "llm_internal_response", ["chat_id"])


def downgrade() -> None:
    op.drop_index("ix_llm_internal_response_chat_id", table_name="llm_internal_response")
    op.drop_constraint("llm_internal_response_message_id_fkey", "llm_internal_response", type_="foreignkey")
    op.drop_constraint("llm_internal_response_chat_id_fkey", "llm_internal_response", type_="foreignkey")
    op.drop_column("llm_internal_response", "message_id")
    op.drop_column("llm_internal_response", "chat_id")
    op.drop_column("llm_internal_response", "purpose")
