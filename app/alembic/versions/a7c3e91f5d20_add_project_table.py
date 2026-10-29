"""add project table and chat.project_id

Revision ID: a7c3e91f5d20
Revises: c3f9880a07dd
Create Date: 2026-10-28 10:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7c3e91f5d20"
down_revision: Union[str, None] = "c3f9880a07dd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Fail fast rather than queue behind a long transaction on chat, which would stall all chat queries.
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.create_table(
        "project",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("uuid", sa.UUID(), server_default=sa.text("uuid_generate_v4()"), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_project_uuid", "project", ["uuid"], unique=False)
    op.create_index("ix_project_user_id", "project", ["user_id"], unique=False)
    # Project names are unique per user, case-insensitively, among live (non-deleted) projects only.
    op.create_index(
        "uq_project_user_id_lower_name",
        "project",
        ["user_id", sa.text("lower(name)")],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    op.add_column("chat", sa.Column("project_id", sa.Integer(), nullable=True))
    # NOT VALID skips scanning chat under a write lock; the new column is all NULL so there is nothing to check.
    op.create_foreign_key(
        "fk_chat_project_id_project", "chat", "project", ["project_id"], ["id"], postgresql_not_valid=True
    )
    op.create_index(
        "ix_chat_project_id",
        "chat",
        ["project_id"],
        unique=False,
        postgresql_where=sa.text("project_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_chat_project_id", table_name="chat")
    op.drop_constraint("fk_chat_project_id_project", "chat", type_="foreignkey")
    op.drop_column("chat", "project_id")

    op.drop_index("uq_project_user_id_lower_name", table_name="project")
    op.drop_index("ix_project_user_id", table_name="project")
    op.drop_index("ix_project_uuid", table_name="project")
    op.drop_table("project")
