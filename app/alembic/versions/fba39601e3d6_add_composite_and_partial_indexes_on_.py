"""add composite and partial indexes on message table

Revision ID: fba39601e3d6
Revises: 2a0b079dc0b2
Create Date: 2025-11-21 16:36:13.460455

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "fba39601e3d6"
down_revision: Union[str, None] = "2a0b079dc0b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create composite index on (chat_id, deleted_at, created_at)
    # Optimizes queries filtering by chat_id and deleted_at, ordered by created_at
    op.create_index(
        "idx_message_chat_deleted_created", "message", ["chat_id", "deleted_at", "created_at"], unique=False
    )

    # Create partial index on deleted_at WHERE deleted_at IS NULL
    # Optimizes queries that filter out soft-deleted messages
    op.execute("CREATE INDEX idx_message_deleted_at ON message(deleted_at) WHERE deleted_at IS NULL")


def downgrade() -> None:
    # Drop the partial index
    op.drop_index("idx_message_deleted_at", table_name="message")

    # Drop the composite index
    op.drop_index("idx_message_chat_deleted_created", table_name="message")
