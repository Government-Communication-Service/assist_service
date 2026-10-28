"""Add Claude 4.1 Opus

Revision ID: b463d4cca67d
Revises: 94d82c2235cd
Create Date: 2025-08-14 19:38:12.361654

"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import MetaData, Table

# revision identifiers, used by Alembic.
revision: str = "b463d4cca67d"
down_revision: Union[str, None] = "94d82c2235cd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


models = [
    {
        "model": "anthropic.claude-opus-4-1-20250805-v1:0",
        "provider": "bedrock",
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
        "max_tokens": 8192,
    },
]


def upgrade() -> None:
    meta = MetaData()
    meta.reflect(bind=op.get_bind())
    llm_table = Table("llm", meta)
    op.bulk_insert(llm_table, models)


def downgrade() -> None:
    meta = MetaData()
    meta.reflect(bind=op.get_bind())
    llm_table = Table("llm", meta)
    d = llm_table.delete().where(
        llm_table.c.model == "anthropic.claude-opus-4-1-20250805-v1:0" and llm_table.c.provider == "bedrock",
    )
    op.execute(d)
