from dataclasses import dataclass
from logging import getLogger

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.sql.selectable import Subquery

from app.database.models import (
    Document,
    DocumentChunk,
    Message,
    MessageDocumentChunkMapping,
)
from app.document_upload.personal_document_rag import ScoredChunk

logger = getLogger(__name__)


@dataclass(frozen=True)
class PositionedChunk:
    """A stored chunk with its place in its document.

    Args:
        document_chunk: The chunk row.
        document: The document it belongs to.
        position: 1-based place of the chunk among the document's active chunks, in reading order.
        total_chunks: Number of active chunks in the document.
    """

    document_chunk: DocumentChunk
    document: Document
    position: int
    total_chunks: int


def chunk_positions(document_uuids: list[str]) -> Subquery:
    """Subquery giving each active chunk of the documents its position in reading order.

    Chunk ids are assigned in the order the parser produced the chunks, so position is the rank of
    the id within its document.

    Args:
        document_uuids: Documents to include.

    Returns:
        A subquery with columns chunk_id, position and total_chunks.
    """
    return (
        select(
            DocumentChunk.id.label("chunk_id"),
            func.row_number().over(partition_by=DocumentChunk.document_id, order_by=DocumentChunk.id).label("position"),
            func.count().over(partition_by=DocumentChunk.document_id).label("total_chunks"),
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            Document.uuid.in_(document_uuids),
            Document.deleted_at.is_(None),
            DocumentChunk.deleted_at.is_(None),
        )
        .subquery()
    )


async def get_document_chunk_mappings(
    chunks: list[ScoredChunk],
    document_uuids: list[str],
    message_id: int,
    session: AsyncSession,
    use_chunk: bool | None = None,
) -> list[PositionedChunk]:
    """Look up the stored rows for search hits and record them against the message.

    Args:
        chunks: Search hits.
        document_uuids: Documents the hits must belong to.
        message_id: The user message the hits were returned for.
        session: Session to query and record in.
        use_chunk: Value for the mapping's use_document_chunk, if it is to be set.

    Returns:
        The hits found in the database, with their positions, in the order of `chunks`.
    """
    if not chunks:
        return []
    positions = chunk_positions(document_uuids)
    rows = await session.execute(
        select(DocumentChunk, Document, positions.c.position, positions.c.total_chunks)
        .join(positions, positions.c.chunk_id == DocumentChunk.id)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.id_opensearch.in_([chunk.id_opensearch for chunk in chunks]))
    )
    by_opensearch_id = {row.DocumentChunk.id_opensearch: row for row in rows}

    results = []
    for chunk in chunks:
        row = by_opensearch_id.get(chunk.id_opensearch)
        if row is None:
            logger.warning("DocumentChunk not found for id_opensearch: %s", chunk.id_opensearch)
            continue
        mapping = MessageDocumentChunkMapping(
            message_id=message_id, document_chunk_id=row.DocumentChunk.id, opensearch_score=chunk.score
        )
        if use_chunk is not None:
            mapping.use_document_chunk = use_chunk
        session.add(mapping)
        results.append(PositionedChunk(row.DocumentChunk, row.Document, row.position, row.total_chunks))
    return results


async def get_previously_returned_opensearch_ids(db_session: AsyncSession, message_id: int) -> set[str]:
    """OpenSearch ids of chunks returned for this message's chat so far, this turn included."""
    chat_id = select(Message.chat_id).where(Message.id == message_id).scalar_subquery()
    stmt = (
        select(DocumentChunk.id_opensearch)
        .join(MessageDocumentChunkMapping, MessageDocumentChunkMapping.document_chunk_id == DocumentChunk.id)
        .join(Message, Message.id == MessageDocumentChunkMapping.message_id)
        .where(Message.chat_id == chat_id, MessageDocumentChunkMapping.use_document_chunk.is_(True))
        .distinct()
    )
    result = await db_session.execute(stmt)
    return set(result.scalars().all())
