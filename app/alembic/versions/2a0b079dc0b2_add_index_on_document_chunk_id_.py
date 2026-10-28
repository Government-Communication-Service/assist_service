"""Add index on document_chunk.id_opensearch for performance

Revision ID: 2a0b079dc0b2
Revises: 22b1a192325b
Create Date: 2025-11-21 12:42:58.545901

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2a0b079dc0b2"
down_revision: Union[str, None] = "22b1a192325b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create index on document_chunk.id_opensearch for performance optimization
    op.create_index("idx_document_chunk_id_opensearch", "document_chunk", ["id_opensearch"])


def downgrade() -> None:
    # Drop index on document_chunk.id_opensearch
    op.drop_index("idx_document_chunk_id_opensearch", table_name="document_chunk")
