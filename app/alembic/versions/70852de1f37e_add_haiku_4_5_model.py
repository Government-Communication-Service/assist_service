"""add_haiku_4_5_model

Revision ID: 70852de1f37e
Revises: 05aa6e9c51b7
Create Date: 2026-02-16 14:22:41.782699

"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import MetaData, Table

# revision identifiers, used by Alembic.
revision: str = "70852de1f37e"
down_revision: Union[str, None] = "05aa6e9c51b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

models = [
    {
        "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "provider": "bedrock",
        "input_cost_per_token": 1e-06,
        "output_cost_per_token": 5e-06,
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
    model_names = [m["model"] for m in models]
    d = llm_table.delete().where(llm_table.c.model.in_(model_names))
    op.execute(d)
