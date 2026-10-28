"""Add opus 4.5

Revision ID: afc2607e4aaf
Revises: 5aacca7fcdeb
Create Date: 2025-12-18 12:28:35.116137

"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import MetaData, Table

# revision identifiers, used by Alembic.
revision: str = "afc2607e4aaf"
down_revision: Union[str, None] = "5aacca7fcdeb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

models = [
    {
        "model": "anthropic.claude-opus-4-5-20251101-v1:0",
        "provider": "bedrock",
        "input_cost_per_token": 5e-06,
        "output_cost_per_token": 2.5e-05,
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
