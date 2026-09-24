"""rename chat_classification_mapping topic to task_type

Revision ID: 2c48faa99745
Revises: 8834bc4acb6f
Create Date: 2026-09-24 09:46:48.306529

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2c48faa99745"
down_revision: Union[str, None] = "8834bc4acb6f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column("chat_classification_mapping", "topic", new_column_name="task_type")


def downgrade() -> None:
    op.alter_column("chat_classification_mapping", "task_type", new_column_name="topic")
