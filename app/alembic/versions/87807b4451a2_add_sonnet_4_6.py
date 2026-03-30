"""add_sonnet_4_6

Revision ID: 87807b4451a2
Revises: a56ea001260d
Create Date: 2026-02-16 14:22:41.782699

"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import MetaData, Table

# revision identifiers, used by Alembic.
revision: str = "87807b4451a2"
down_revision: Union[str, None] = "a56ea001260d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

models = [
    {
        "model": "anthropic.claude-sonnet-4-6",
        "provider": "bedrock",
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
        "max_tokens": 8000,
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
