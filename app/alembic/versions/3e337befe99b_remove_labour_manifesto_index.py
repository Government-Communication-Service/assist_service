"""Remove Labour Manifesto index

Revision ID: 3e337befe99b
Revises: 711bc1565eb4
Create Date: 2025-05-14 14:38:04.236622

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3e337befe99b"
down_revision: str | None = "711bc1565eb4"
branch_labels: str | list[str] | None = None
depends_on: str | list[str] | None = None

labour_manifesto_index_name = "labour_manifesto_2024"

# Define table structures for use in the migration
# We only define the columns we interact with.
search_index_table = sa.Table(
    "search_index",
    sa.MetaData(),
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("name", sa.String),
    sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
)

document_table = sa.Table(
    "document",
    sa.MetaData(),
    sa.Column("id", sa.Integer, primary_key=True),
    # search_index_id is not present in this table as per instruction.
    sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
)

document_chunk_table = sa.Table(
    "document_chunk",
    sa.MetaData(),
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("document_id", sa.Integer),  # Foreign key to document.id
    sa.Column("search_index_id", sa.Integer),  # Foreign key to search_index.id
    sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
)


def upgrade() -> None:
    """
    Marks the Labour Manifesto search index, related documents, and document chunks as deleted.
    """
    now_timestamp = sa.func.now()

    # Mark the SearchIndex as deleted
    op.execute(
        search_index_table.update()
        .where(search_index_table.c.name == labour_manifesto_index_name)
        .values(deleted_at=now_timestamp)
    )

    # Create a subquery to get the ID of the SearchIndex to be "deleted"
    labour_manifesto_index_id_subquery = (
        sa.select(search_index_table.c.id)
        .where(search_index_table.c.name == labour_manifesto_index_name)
        .scalar_subquery()
    )

    # Mark related DocumentChunks as deleted
    op.execute(
        document_chunk_table.update()
        .where(document_chunk_table.c.search_index_id == labour_manifesto_index_id_subquery)
        .values(deleted_at=now_timestamp)
    )

    # Subquery to get document_ids related to the Labour Manifesto search index
    # by looking into document_chunk table for chunks associated with that search index.
    related_document_ids_subquery = (
        sa.select(document_chunk_table.c.document_id)
        .where(document_chunk_table.c.search_index_id == labour_manifesto_index_id_subquery)
        .distinct()
    )

    # Mark related Documents as deleted
    op.execute(
        document_table.update()
        .where(document_table.c.id.in_(related_document_ids_subquery))
        .values(deleted_at=now_timestamp)
    )


def downgrade() -> None:
    """
    Reverts the "deletion" of the Labour Manifesto search index, related documents, and document chunks
    by setting their deleted_at fields to NULL.
    """
    # Unmark the SearchIndex as deleted
    op.execute(
        search_index_table.update()
        .where(search_index_table.c.name == labour_manifesto_index_name)
        .values(deleted_at=None)
    )

    # Subquery to get the ID of the SearchIndex
    labour_manifesto_index_id_subquery = (
        sa.select(search_index_table.c.id)
        .where(search_index_table.c.name == labour_manifesto_index_name)
        .scalar_subquery()
    )

    # Unmark related DocumentChunks as deleted
    op.execute(
        document_chunk_table.update()
        .where(document_chunk_table.c.search_index_id == labour_manifesto_index_id_subquery)
        .values(deleted_at=None)
    )

    # Subquery to get document_ids related to the Labour Manifesto search index
    related_document_ids_subquery = (
        sa.select(document_chunk_table.c.document_id)
        .where(document_chunk_table.c.search_index_id == labour_manifesto_index_id_subquery)
        .distinct()
    )

    # Unmark related Documents as deleted
    op.execute(
        document_table.update().where(document_table.c.id.in_(related_document_ids_subquery)).values(deleted_at=None)
    )
