"""add_cache_read_write_tokens_to_message_and_llm_internal_response

Revision ID: 4fd83184f219
Revises: c7e2a91d5b38
Create Date: 2026-09-02 10:47:49.637705

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4fd83184f219"
down_revision: Union[str, None] = "c7e2a91d5b38"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("message", sa.Column("cache_read_tokens", sa.Integer(), nullable=True))
    op.add_column("message", sa.Column("cache_write_tokens", sa.Integer(), nullable=True))
    op.add_column("llm_internal_response", sa.Column("cache_read_tokens", sa.Integer(), nullable=True))
    op.add_column("llm_internal_response", sa.Column("cache_write_tokens", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("llm_internal_response", "cache_write_tokens")
    op.drop_column("llm_internal_response", "cache_read_tokens")
    op.drop_column("message", "cache_write_tokens")
    op.drop_column("message", "cache_read_tokens")
