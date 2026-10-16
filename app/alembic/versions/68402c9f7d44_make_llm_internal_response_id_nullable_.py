"""make llm_internal_response_id nullable on audit tables

Revision ID: 68402c9f7d44
Revises: 8c5e1a1ae81e
Create Date: 2026-09-29 14:21:06.294670

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "68402c9f7d44"
down_revision: Union[str, None] = "8c5e1a1ae81e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


TABLES = ["rewritten_query", "gov_uk_search_query", "gov_uk_search_result", "use_gov_uk_search_decision"]


def upgrade() -> None:
    for table in TABLES:
        op.alter_column(table, "llm_internal_response_id", existing_type=sa.Integer(), nullable=True)


def downgrade() -> None:
    for table in TABLES:
        op.alter_column(table, "llm_internal_response_id", existing_type=sa.Integer(), nullable=False)
