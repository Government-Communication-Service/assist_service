"""add extra llm table attributes
REFERENCE: https://aws.amazon.com/bedrock/pricing/
Revision ID: 8a2f8810f8f8
Revises: c2bd89d922af
Create Date: 2024-08-01 17:36:13.822278

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8a2f8810f8f8"
down_revision: Union[str, None] = "c2bd89d922af"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add input_cost_per_token, output_cost_per_token, and max_tokens columns
    op.add_column("llm", sa.Column("input_cost_per_token", sa.Float(), nullable=True))
    op.add_column("llm", sa.Column("output_cost_per_token", sa.Float(), nullable=True))
    op.add_column("llm", sa.Column("max_tokens", sa.Integer(), nullable=True))

    # Update input_cost_per_token, output_cost_per_token, and max_tokens for Anthropic models
    op.execute(
        """
        UPDATE llm
        SET
            input_cost_per_token = CASE
                WHEN model LIKE 'anthropic.claude-3-haiku%' THEN 0.00025 / 1000
                WHEN model LIKE 'anthropic.claude-3-sonnet%' THEN 0.003 / 1000
                WHEN model LIKE 'anthropic.claude-3-5-sonnet%' THEN 0.003 / 1000
                WHEN model LIKE 'anthropic.claude-3-opus%' THEN 0.015 / 1000
                WHEN model LIKE 'anthropic.claude-v2%' THEN 0.008 / 1000
                WHEN model LIKE 'anthropic.claude-instant%' THEN 0.0008 / 1000
                ELSE NULL
            END,
            output_cost_per_token = CASE
                WHEN model LIKE 'anthropic.claude-3-haiku%' THEN 0.00125 / 1000
                WHEN model LIKE 'anthropic.claude-3-sonnet%' THEN 0.015 / 1000
                WHEN model LIKE 'anthropic.claude-3-5-sonnet%' THEN 0.015 / 1000
                WHEN model LIKE 'anthropic.claude-3-opus%' THEN 0.075 / 1000
                WHEN model LIKE 'anthropic.claude-v2%' THEN 0.024 / 1000
                WHEN model LIKE 'anthropic.claude-instant%' THEN 0.0024 / 1000
                ELSE NULL
            END,
            max_tokens = CASE
                WHEN model LIKE 'anthropic.claude-3-haiku%' THEN 4096
                WHEN model LIKE 'anthropic.claude-3-sonnet%' THEN 4096
                WHEN model LIKE 'anthropic.claude-3-5-sonnet%' THEN 8192
                WHEN model LIKE 'anthropic.claude-3-opus%' THEN 4096
                WHEN model LIKE 'anthropic.claude-v2%' THEN 4096
                WHEN model LIKE 'anthropic.claude-instant%' THEN 4096
                ELSE NULL
            END
        WHERE provider = 'bedrock' AND model LIKE 'anthropic.%'
    """
    )


def downgrade() -> None:
    op.drop_column("llm", "input_cost_per_token")
    op.drop_column("llm", "output_cost_per_token")
    op.drop_column("llm", "max_tokens")
