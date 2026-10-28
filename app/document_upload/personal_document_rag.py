import logging
from dataclasses import dataclass

from app.config import UPLOADED_DOCUMENTS_MAX_RESULT_CHARS
from app.document_upload.constants import PERSONAL_DOCUMENTS_INDEX_NAME

from ..opensearch import AsyncOpenSearchOperations

logger = logging.getLogger(__name__)


MAX_HITS_PER_QUERY = int(UPLOADED_DOCUMENTS_MAX_RESULT_CHARS / 750)
"""
Hits requested from OpenSearch per query: the result character limit over a low-ball estimate of
characters per chunk, so the limit rather than the hit count is what stops a search.
"""

MULTI_QUERY_BOOST = 0.2
"""Fraction added to a chunk's summed score for each further query it matched."""


@dataclass(frozen=True)
class ScoredChunk:
    """A search hit: everything the rest of the app needs to know about it.

    Args:
        id_opensearch: The chunk's OpenSearch id.
        score: Combined relevance score across the queries.
        char_count: Length of the chunk's content.
    """

    id_opensearch: str
    score: float
    char_count: int


async def select_chunks_for_tool(
    queries: list[str], document_uuids: list[str], exclude_ids: set[str]
) -> tuple[list[ScoredChunk], int]:
    """Choose uploaded-document chunks for a search_uploaded_documents tool call.

    Each query is searched; a chunk's scores add up across the queries that found it, boosted for
    each further query. The highest-scoring chunks are returned up to
    UPLOADED_DOCUMENTS_MAX_RESULT_CHARS.

    Args:
        queries: Keyword queries from the model.
        document_uuids: Documents to search.
        exclude_ids: OpenSearch ids of chunks already returned earlier in the chat.

    Returns:
        The chosen chunks, best first, and how many matching chunks were skipped as already returned.
    """
    totals: dict[str, float] = {}
    query_counts: dict[str, int] = {}
    char_counts: dict[str, int] = {}
    skipped_ids = set()
    for query in queries:
        hits = await AsyncOpenSearchOperations.search_multiple_document_chunks(
            document_uuids, query, PERSONAL_DOCUMENTS_INDEX_NAME, max_size=MAX_HITS_PER_QUERY
        )
        for hit in hits:
            chunk_id = hit["_id"]
            if chunk_id in exclude_ids:
                skipped_ids.add(chunk_id)
                continue
            totals[chunk_id] = totals.get(chunk_id, 0.0) + hit.get("_score", 0.0)
            query_counts[chunk_id] = query_counts.get(chunk_id, 0) + 1
            char_counts[chunk_id] = len(hit.get("_source", {}).get("chunk_content", ""))

    scored = [
        ScoredChunk(chunk_id, total * (1.0 + (query_counts[chunk_id] - 1) * MULTI_QUERY_BOOST), char_counts[chunk_id])
        for chunk_id, total in totals.items()
    ]
    scored.sort(key=lambda chunk: chunk.score, reverse=True)

    selected = []
    chars_used = 0
    for chunk in scored:
        if chars_used + chunk.char_count <= UPLOADED_DOCUMENTS_MAX_RESULT_CHARS:
            selected.append(chunk)
            chars_used += chunk.char_count
    return selected, len(skipped_ids)
