# ruff: noqa: E501

"""Update opensearch query writer

Revision ID: 0al6cf1e171f
Revises: 0b1a2c846de8
Create Date: 2025-01-06 11:29:59.360772

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0al6cf1e171f"
down_revision: Union[str, None] = "0b1a2c846de8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

opensearch_system_prompt_new_content = "You work for members of GCS (the Government Communication Service) which is part of the UK Civil Service. Your job is to rewrite the users query so it can be used in the query body for OpenSearch to retrieve relevant information. The next message you receive will be the users original query. You should only respond with an unlabelled JSON array of a variety of three queries that are rewritten for maximum effect with OpenSearch. Each array item must be a plain text query, and it can have logical operators to improve search match."
opensearch_system_prompt_old_content = "You work for members of GCS (the Government Communication Service) which is part of the UK Civil Service. Your job is to rewrite the users query so it can be used in the query body for OpenSearch to retrieve relevant information. The next message you receive will be the users original query. You should only respond with an unlabelled JSON array of a variety of three queries that are rewritten for maximum effect with OpenSearch."


def upgrade() -> None:
    op.execute(
        f"UPDATE system_prompt SET content = '{opensearch_system_prompt_new_content}'  WHERE name ='query_rewriter_1'"
    )


def downgrade() -> None:
    op.execute(
        f"UPDATE system_prompt SET content = '{opensearch_system_prompt_old_content}'  WHERE name ='query_rewriter_1'"
    )
