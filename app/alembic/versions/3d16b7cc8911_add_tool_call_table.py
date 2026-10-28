"""add tool_call table

Revision ID: 3d16b7cc8911
Revises: a8faf7d10994
Create Date: 2026-09-28 11:13:59.390768

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3d16b7cc8911"
down_revision: Union[str, None] = "a8faf7d10994"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tool_call",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("uuid", sa.dialects.postgresql.UUID(), server_default=sa.text("uuid_generate_v4()"), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("llm_internal_response_id", sa.Integer(), nullable=True),
        sa.Column("iteration", sa.Integer(), nullable=False),
        sa.Column("tool_use_id", sa.Text(), nullable=False),
        sa.Column("tool_name", sa.Text(), nullable=False),
        sa.Column("args", sa.JSON(), nullable=True),
        sa.Column("is_error", sa.Boolean(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("result_chars", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["message_id"], ["message.id"]),
        sa.ForeignKeyConstraint(["llm_internal_response_id"], ["llm_internal_response.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_tool_call_message_id", "tool_call", ["message_id"])


def downgrade() -> None:
    op.drop_index("idx_tool_call_message_id", table_name="tool_call")
    op.drop_table("tool_call")
