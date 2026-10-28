"""add index on document_chunk document_id

Revision ID: b8d3e4f1a927
Revises: 1818b315958c
Create Date: 2026-09-17 10:24:18.402517

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b8d3e4f1a927"
down_revision: Union[str, None] = "1818b315958c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEX_NAME = "idx_document_chunk_document_id_active"

CREATE_INDEX = (
    f"CREATE INDEX CONCURRENTLY {INDEX_NAME} "
    "ON document_chunk (document_id) INCLUDE (id_opensearch) "
    "WHERE deleted_at IS NULL"
)

# indisvalid is false when a CONCURRENTLY build failed part way through. Such an index is
# ignored by the planner but still occupies the name, so it must be told apart from a
# finished one rather than matched on name alone.

INDEX_STATE = sa.text(
    "SELECT i.indisvalid FROM pg_class c "
    "JOIN pg_index i ON i.indexrelid = c.oid "
    "WHERE c.relname = :name AND c.relkind = 'i' AND pg_table_is_visible(c.oid)"
)


def upgrade() -> None:
    # document_chunk.document_id had no index - Postgres does not create one for a foreign key -
    # so deleting a single document sequentially scanned the whole table twice, once to collect
    # the OpenSearch ids and once to mark the chunks deleted. The cost scaled with the size of
    # the whole corpus rather than the document being deleted.
    with op.get_context().autocommit_block():
        is_valid = op.get_bind().execute(INDEX_STATE, {"name": INDEX_NAME}).scalar()

        if is_valid is False:
            # An earlier build was interrupted and left an invalid index behind. CREATE INDEX
            # ... IF NOT EXISTS would match it by name and skip, so this revision would be
            # recorded as applied while the deletion queries still had no usable index. Drop
            # it and build again.
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
            is_valid = None

        if is_valid is None:
            op.execute(CREATE_INDEX)


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
