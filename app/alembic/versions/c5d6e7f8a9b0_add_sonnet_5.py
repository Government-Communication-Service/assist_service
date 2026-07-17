"""add_sonnet_5

Revision ID: c5d6e7f8a9b0
Revises: 64391557a482
Create Date: 2026-07-08 14:58:03.090051

"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import MetaData, Table

# revision identifiers, used by Alembic.
revision: str = "c5d6e7f8a9b0"
down_revision: Union[str, None] = "64391557a482"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

models = [
    {
        "model": "anthropic.claude-sonnet-5",
        "provider": "bedrock",
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
        "max_tokens": 12000,
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
