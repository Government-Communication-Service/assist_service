import os

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import text as sa_text

# Revision identifiers, used by Alembic.
revision = "c732af2e465a"
down_revision = None
branch_labels = None
depends_on = None

print(f"initial_migration.py alembic POSTGRES_DB = {os.getenv('POSTGRES_DB')}")


def common_columns():
    """Return common columns used in multiple tables."""
    return [
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "uuid",
            UUID(as_uuid=True),
            server_default=sa_text("uuid_generate_v4()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa_text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa_text("CURRENT_TIMESTAMP"),
            onupdate=sa_text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
    ]


def upgrade():
    # Enable UUID extension in PostgreSQL
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    op.create_table(
        "llm",
        *common_columns(),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("provider", sa.String(length=255), nullable=False),
    )

    # Create tables in dependency order, referencing common columns
    op.create_table(
        "theme",
        *common_columns(),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("subtitle", sa.Text(), nullable=False),
    )

    op.create_table("user", *common_columns())

    op.create_table(
        "user_group",
        *common_columns(),
        sa.Column("group", sa.String(length=255), nullable=False),
    )

    op.create_table(
        "feedback_score",
        *common_columns(),
        sa.Column("score", sa.String(length=255), nullable=False),
    )

    op.create_table(
        "feedback_label",
        *common_columns(),
        sa.Column("label", sa.String(length=255), nullable=False),
    )

    op.create_table(
        "auth_session",
        *common_columns(),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id"), nullable=False),
    )

    op.create_table(
        "use_case",
        *common_columns(),
        sa.Column("theme_id", sa.Integer(), sa.ForeignKey("theme.id"), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("system_prompt", sa.Text(), nullable=True),
        sa.Column("instruction", sa.Text(), nullable=False),
        sa.Column("user_input_form", sa.Text(), nullable=False),
    )

    op.create_table(
        "redaction",
        *common_columns(),
        sa.Column("redacted", sa.Boolean(), nullable=False),
        sa.Column("redaction_reason", sa.String(length=255), nullable=False),
        sa.Column("alert_level", sa.Integer(), nullable=False),
        sa.Column("alert_message", sa.Text(), nullable=False),
    )

    op.create_table(
        "chat",
        *common_columns(),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id"), nullable=False),
        sa.Column("use_case_id", sa.Integer(), sa.ForeignKey("use_case.id"), nullable=True),
        sa.Column("title", sa.String(length=255), default="Default title", nullable=False),
        sa.Column("from_open_chat", sa.Boolean(), nullable=False),
    )

    op.create_table(
        "message",
        *common_columns(),
        sa.Column("chat_id", sa.Integer(), sa.ForeignKey("chat.id"), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False),
        sa.Column("tokens", sa.Integer(), nullable=False),
        sa.Column(
            "auth_session_id",
            sa.Integer(),
            sa.ForeignKey("auth_session.id"),
            nullable=False,
        ),
        sa.Column(
            "parent_message_id",
            sa.Integer(),
            sa.ForeignKey("message.id"),
            nullable=True,
        ),
        sa.Column("redaction_id", sa.Integer(), sa.ForeignKey("redaction.id")),
        sa.Column("interrupted", sa.Boolean(), nullable=False),
        sa.Column("completion_cost", sa.Numeric(), nullable=True),
        sa.Column("llm_id", sa.Integer(), sa.ForeignKey("llm.id"), nullable=True),
    )

    op.create_table(
        "action_type",
        *common_columns(),
        sa.Column("action_name", sa.String(255), nullable=True),
    )

    op.create_table(
        "user_action",
        *common_columns(),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id"), nullable=False),
        sa.Column(
            "auth_session_id",
            sa.Integer(),
            sa.ForeignKey("auth_session.id"),
            nullable=False,
        ),
        sa.Column(
            "action_type_id",
            sa.Integer(),
            sa.ForeignKey("action_type.id"),
            nullable=False,
        ),
        sa.Column("action_properties", sa.JSON(), nullable=True),
    )

    op.create_table(
        "message_user_group_mapping",
        *common_columns(),
        sa.Column("message_id", sa.Integer(), sa.ForeignKey("message.id"), primary_key=True),
        sa.Column(
            "user_group_id",
            sa.Integer(),
            sa.ForeignKey("user_group.id"),
            primary_key=True,
        ),
    )

    op.create_table(
        "feedback",
        *common_columns(),
        sa.Column("message_id", sa.Integer(), sa.ForeignKey("message.id"), nullable=False),
        sa.Column("feedback_score_id", sa.Integer(), sa.ForeignKey("feedback_score.id")),
        sa.Column("feedback_label_id", sa.Integer(), sa.ForeignKey("feedback_label.id")),
        sa.Column("freetext", sa.Text(), nullable=True),
    )

    # Create indexes
    op.create_index("ix_user_uuid", "user", ["uuid"], unique=False)
    op.create_index("ix_chat_uuid", "chat", ["uuid"], unique=False)
    op.create_index("ix_message_uuid", "message", ["uuid"], unique=False)
    op.create_index("ix_auth_session_uuid", "auth_session", ["uuid"], unique=False)
    op.create_index("ix_feedback_message_id", "feedback", ["message_id"], unique=False)


def downgrade():
    # Drop indexes first
    op.drop_index("ix_user_uuid", table_name="user")
    op.drop_index("ix_chat_uuid", table_name="chat")
    op.drop_index("ix_message_uuid", table_name="message")
    op.drop_index("ix_auth_session_uuid", table_name="auth_session")
    op.drop_index("ix_feedback_message_id", table_name="feedback")

    # Drop tables in reverse order of creation
    op.drop_table("message_user_group_mapping")
    op.drop_table("user_action")
    op.drop_table("action_type")
    op.drop_table("message")
    op.drop_table("chat")
    op.drop_table("redaction")
    op.drop_table("use_case")
    op.drop_table("auth_session")
    op.drop_table("user_group")
    op.drop_table("user")
    op.drop_table("theme")
    op.drop_table("feedback_label")
    op.drop_table("feedback_score")
    op.drop_table("feedback")
    op.drop_table("llm")
