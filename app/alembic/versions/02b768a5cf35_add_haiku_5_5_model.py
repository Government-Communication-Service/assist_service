"""add_haiku_5_5_model

Revision ID: 02b768a5cf35
Revises: 68402c9f7d44
Create Date: 2026-10-23 10:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import MetaData, Table

# revision identifiers, used by Alembic.
revision: str = "02b768a5cf35"
down_revision: Union[str, None] = "68402c9f7d44"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Prices are the <=100K-token prompt rate ($0.10 / $0.50 per MTok); prompts over 100K are billed at $0.50 / $2.50,
# which the single-rate llm table can't express. max_tokens is above Haiku 4.5's 8192 because the same text is
# ~30% more tokens on Haiku 5.5's tokenizer.
models = [
    {
        "model": "anthropic.claude-haiku-5-5",
        "provider": "bedrock",
        "input_cost_per_token": 1e-07,
        "output_cost_per_token": 5e-07,
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
    message_table = Table("message", meta)
    model_names = [m["model"] for m in models]
    rows = op.get_bind().execute(llm_table.select().where(llm_table.c.model.in_(model_names))).fetchall()
    if rows:
        ids = [row._mapping["id"] for row in rows]
        op.execute(message_table.update().where(message_table.c.llm_id.in_(ids)).values(llm_id=None))
    op.execute(llm_table.delete().where(llm_table.c.model.in_(model_names)))
