"""update_sonnet_4_6_max_tokens

Revision ID: 05aa6e9c51b7
Revises: 87807b4451a2
Create Date: 2026-02-17 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import text

revision: str = "05aa6e9c51b7"
down_revision: Union[str, None] = "87807b4451a2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

MODEL = "anthropic.claude-sonnet-4-6"


def upgrade() -> None:
    op.execute(text("UPDATE llm SET max_tokens = 12000 WHERE model = :model").bindparams(model=MODEL))


def downgrade() -> None:
    op.execute(text("UPDATE llm SET max_tokens = 8000 WHERE model = :model").bindparams(model=MODEL))
