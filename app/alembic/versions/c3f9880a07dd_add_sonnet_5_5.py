"""add_sonnet_5_5

Revision ID: c3f9880a07dd
Revises: 02b768a5cf35
Create Date: 2026-10-23 10:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import MetaData, Table

# revision identifiers, used by Alembic.
revision: str = "c3f9880a07dd"
down_revision: Union[str, None] = "02b768a5cf35"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# $2 / $10 per MTok input / output, like Sonnet 5, but cache reads cost $0.10 per MTok where Sonnet 5's cost $0.20.
# The cache read multiplier is relative to the input price: 0.10 / 2.00 = 0.05. Cache writes use the global multiplier.
models = [
    {
        "model": "anthropic.claude-sonnet-5-5",
        "provider": "bedrock",
        "input_cost_per_token": 2e-06,
        "output_cost_per_token": 1e-05,
        "cache_read_cost_multiplier": 0.05,
        "max_tokens": 12000,
    },
]


def upgrade() -> None:
    # Per-model override of the global cache read multiplier; NULL means use the global one.
    op.add_column("llm", sa.Column("cache_read_cost_multiplier", sa.Float(), nullable=True))
    meta = MetaData()
    meta.reflect(bind=op.get_bind())
    llm_table = Table("llm", meta)
    op.bulk_insert(llm_table, models)


def downgrade() -> None:
    meta = MetaData()
    meta.reflect(bind=op.get_bind())
    llm_table = Table("llm", meta)
    message_table = Table("message", meta)
    model_names = [m["model"] for m in models]
    rows = op.get_bind().execute(llm_table.select().where(llm_table.c.model.in_(model_names))).fetchall()
    if rows:
        ids = [row._mapping["id"] for row in rows]
        op.execute(message_table.update().where(message_table.c.llm_id.in_(ids)).values(llm_id=None))
    op.execute(llm_table.delete().where(llm_table.c.model.in_(model_names)))
    op.drop_column("llm", "cache_read_cost_multiplier")
