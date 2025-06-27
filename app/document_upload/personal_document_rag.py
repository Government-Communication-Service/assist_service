import logging
from typing import Dict, List

from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.central_guidance.schemas import RagRequest, RetrievalResult
from app.database.models import (
    Message,
    MessageSearchIndexMapping,
    SearchIndex,
)
from app.database.table import async_db_session
from app.document_upload.constants import PERSONAL_DOCUMENTS_INDEX_NAME
from app.document_upload.utils import get_document_chunk_mappings, rewrite_user_query

from ..opensearch import AsyncOpenSearchOperations

logger = logging.getLogger(__name__)


GLOBAL_CHARACTER_LIMIT = 55000
"""
The maximum number of characters that can be included in the context across all documents
"""

LOW_PERCENTILE_ESTIMATE_CHARACTERS_PER_CHUNK = 750
"""
A low-ball estimate of characters per chunk to help us set a max size argument for OpenSearch.
"""

OPENSEARCH_CHUNK_LIMIT = int(GLOBAL_CHARACTER_LIMIT / LOW_PERCENTILE_ESTIMATE_CHARACTERS_PER_CHUNK)
"""
Gives an upper bound of the number of chunks to retrieve from OpenSearch based on
the character limit and a lower-than-median estimate of the number of characters per chunk.
"""


async def _get_rewritten_queries(
    query: str,
    search_index: SearchIndex,
    message: Message,
    db_session: AsyncSession,
) -> List[str]:
    """
    Generates rewritten queries for improved search results based on the user's chat query.
    Args:
        query (str): The original chat query
        search_index (SearchIndex): The search index instance used for saving generated queries against in the database
        message (Message): A `Message` instance representing the current chat message used for saving generated
        queries against in the database
        llm_query_rewriter (LLM): The LLM instance holding LLM model and LLM provider details.
        system_prompt_query_rewriter (SystemPrompt): The system prompt holding instruction for LLM for
        how to generate opensearch queries

    Returns:
        List[str]: A list of rewritten queries
    """

    # Rewrite the user's query to get better results from the search index
    rewritten_queries = await rewrite_user_query(query, search_index, message.id, db_session)
    return rewritten_queries


async def _apply_fair_document_distribution(
    chunk_scores: Dict[str, Dict],
    document_uuids: List[str],
    character_limit: int,
) -> List[Dict]:
    """
    Applies fair document distribution to ensure smaller documents get representation
    while still allowing larger, more relevant documents to be over-represented.

    Strategy:
    1. Guarantee minimum allocation per document (10% of limit / num_docs, min 2000 chars)
    2. Distribute remaining capacity proportionally based on document size
    3. Reallocate unused capacity based on relevance scores

    Args:
        chunk_scores: Dictionary mapping chunk_id to chunk info with scores
        document_uuids: List of document UUIDs being processed
        character_limit: Maximum characters allowed

    Returns:
        List of selected chunks respecting fair distribution
    """
    if not chunk_scores or not document_uuids:
        return []

    # Group chunks by document
    doc_chunks = {}  # document_uuid -> list of chunk_info
    doc_total_chars = {}  # document_uuid -> total available characters

    for chunk_info in chunk_scores.values():
        doc_uuid = chunk_info["chunk_data"].get("_source", {}).get("document_uuid")
        if doc_uuid in document_uuids:
            if doc_uuid not in doc_chunks:
                doc_chunks[doc_uuid] = []
                doc_total_chars[doc_uuid] = 0
            doc_chunks[doc_uuid].append(chunk_info)
            doc_total_chars[doc_uuid] += chunk_info["character_count"]

    # Sort chunks within each document by relevance score
    for doc_uuid in doc_chunks:
        doc_chunks[doc_uuid].sort(key=lambda x: x["final_score"], reverse=True)

    # Calculate fair allocation
    num_docs = len(doc_chunks)
    if num_docs == 0:
        return []

    # Minimum allocation per document: 10% of total limit divided by number of docs, but at least 2000 chars
    min_allocation_per_doc = max(2000, int(character_limit * 0.1 / num_docs))
    total_min_allocation = min_allocation_per_doc * num_docs

    # If minimum allocations exceed limit, scale them down proportionally
    if total_min_allocation > character_limit:
        min_allocation_per_doc = character_limit // num_docs
        total_min_allocation = min_allocation_per_doc * num_docs

    remaining_capacity = character_limit - total_min_allocation

    logger.info(
        "Fair distribution: %d docs, %d chars limit, %d min per doc, %d remaining",
        num_docs,
        character_limit,
        min_allocation_per_doc,
        remaining_capacity,
    )

    # Phase 1: Allocate minimum characters to each document
    selected_chunks = []
    doc_allocations = {}  # document_uuid -> allocated characters
    doc_used_chars = {}  # document_uuid -> actually used characters

    for doc_uuid in doc_chunks:
        doc_allocations[doc_uuid] = min_allocation_per_doc
        doc_used_chars[doc_uuid] = 0

        # Add chunks up to minimum allocation
        for chunk_info in doc_chunks[doc_uuid]:
            if doc_used_chars[doc_uuid] + chunk_info["character_count"] <= min_allocation_per_doc:
                selected_chunks.append(chunk_info["chunk_data"])
                doc_used_chars[doc_uuid] += chunk_info["character_count"]
            else:
                break

    # Phase 2: Distribute remaining capacity proportionally based on document size and relevance
    if remaining_capacity > 0:
        # Calculate proportional allocations based on total available characters in each doc
        total_available_chars = sum(doc_total_chars.values())

        for doc_uuid in doc_chunks:
            if total_available_chars > 0:
                # Proportional allocation based on document size
                proportion = doc_total_chars[doc_uuid] / total_available_chars
                additional_allocation = int(remaining_capacity * proportion)
                doc_allocations[doc_uuid] += additional_allocation

        # Add more chunks up to the new allocations
        for doc_uuid in doc_chunks:
            target_chars = doc_allocations[doc_uuid]

            for chunk_info in doc_chunks[doc_uuid]:
                # Skip chunks already selected
                if chunk_info["chunk_data"] in selected_chunks:
                    continue

                if doc_used_chars[doc_uuid] + chunk_info["character_count"] <= target_chars:
                    selected_chunks.append(chunk_info["chunk_data"])
                    doc_used_chars[doc_uuid] += chunk_info["character_count"]

    # Phase 3: Fill remaining space with highest scoring chunks across all documents
    total_used = sum(doc_used_chars.values())
    if total_used < character_limit:
        # Get all remaining chunks sorted by score
        remaining_chunks = []
        for doc_uuid in doc_chunks:
            for chunk_info in doc_chunks[doc_uuid]:
                if chunk_info["chunk_data"] not in selected_chunks:
                    remaining_chunks.append(chunk_info)

        remaining_chunks.sort(key=lambda x: x["final_score"], reverse=True)

        # Add highest scoring chunks until we hit the limit
        for chunk_info in remaining_chunks:
            if total_used + chunk_info["character_count"] <= character_limit:
                selected_chunks.append(chunk_info["chunk_data"])
                total_used += chunk_info["character_count"]
                doc_used_chars[chunk_info["chunk_data"].get("_source", {}).get("document_uuid")] += chunk_info[
                    "character_count"
                ]

    # Log distribution results
    for doc_uuid in doc_chunks:
        num_chunks = sum(1 for chunk in selected_chunks if chunk.get("_source", {}).get("document_uuid") == doc_uuid)
        logger.info(
            "Document %s: %d chunks, %d characters (target: %d)",
            doc_uuid[:8],
            num_chunks,
            doc_used_chars[doc_uuid],
            doc_allocations[doc_uuid],
        )

    final_char_count = sum(len(chunk.get("_source", {}).get("chunk_content", "")) for chunk in selected_chunks)
    logger.info("Fair distribution complete: %d chunks, %d characters", len(selected_chunks), final_char_count)

    return selected_chunks


async def _retrieve_relevant_chunks(
    rag_request: RagRequest,
    search_index: SearchIndex,
    message: Message,
    db_session: AsyncSession,
) -> List[Dict]:
    """
    Retrieves relevant document chunks based on the user's query and request context.
    This function now implements a global character limit of 55,000 across all documents.
    If the total character count across all documents is <= 55,000, all chunks are included.
    If > 55,000, the top chunks by relevance are retrieved via search until the character limit is reached.

    Args:
        rag_request (RagRequest): The request object containing the user query and the UUIDs
            of the target documents.
        search_index (SearchIndex): The search index db record.
        message (Message): A `Message` instance representing the current chat message
        db_session: The database session

    Returns:
        List[Dict]: A list of unique opensearch documents relevant to the user's query, represented as
        dictionaries.

    """
    logger.info("starting to retrieve pdu relevant chunks with global character limit of %s", GLOBAL_CHARACTER_LIMIT)

    # First, get all chunks across all documents to determine total character count
    # Use a higher max_size since we're now limited by characters, not chunk count
    all_chunks = await AsyncOpenSearchOperations.get_multiple_document_chunks(
        PERSONAL_DOCUMENTS_INDEX_NAME,
        rag_request.document_uuids,
        max_size=OPENSEARCH_CHUNK_LIMIT,
    )

    # Calculate total character count across all chunks
    total_character_count = sum(len(chunk.get("_source", {}).get("chunk_content", "")) for chunk in all_chunks)

    logger.info(
        "Retrieved %s chunks (%s characters) across %s documents (limited by max_size=%s for initial check)",
        len(all_chunks),
        total_character_count,
        len(rag_request.document_uuids),
        OPENSEARCH_CHUNK_LIMIT,
    )

    if total_character_count <= GLOBAL_CHARACTER_LIMIT:
        # Use all chunks without search since we're under the character limit
        logger.info(
            "Using all %s chunks (%s characters) since total is within the global limit of %s characters",
            len(all_chunks),
            total_character_count,
            GLOBAL_CHARACTER_LIMIT,
        )
        unique_chunks = all_chunks
    else:
        # Need to search for the most relevant chunks since we exceed the character limit
        logger.info(
            "Total characters (%s) exceed global limit (%s), performing search for top chunks within character limit",
            total_character_count,
            GLOBAL_CHARACTER_LIMIT,
        )

        # Generate rewritten queries for better search results
        rewritten_queries = await _get_rewritten_queries(rag_request.query, search_index, message, db_session)

        # Search across all documents for the most relevant chunks and consolidate scores
        chunk_scores = {}  # chunk_id -> {chunk_data, total_score, query_count, final_score, character_count}

        for query in rewritten_queries:
            # Request more chunks since we'll filter by characters
            chunks = await AsyncOpenSearchOperations.search_multiple_document_chunks(
                rag_request.document_uuids, query, PERSONAL_DOCUMENTS_INDEX_NAME, max_size=OPENSEARCH_CHUNK_LIMIT
            )

            for chunk in chunks:
                chunk_id = chunk["_id"]
                opensearch_score = chunk.get("_score", 0.0)
                chunk_content = chunk.get("_source", {}).get("chunk_content", "")
                character_count = len(chunk_content)

                if chunk_id in chunk_scores:
                    # Chunk appeared in multiple queries - boost its score
                    chunk_scores[chunk_id]["total_score"] += opensearch_score
                    chunk_scores[chunk_id]["query_count"] += 1
                    # Apply multi-query boost: chunks appearing in multiple queries get a bonus
                    multi_query_boost = 1.0 + (chunk_scores[chunk_id]["query_count"] - 1) * 0.2
                    chunk_scores[chunk_id]["final_score"] = chunk_scores[chunk_id]["total_score"] * multi_query_boost
                else:
                    # First time seeing this chunk
                    chunk_scores[chunk_id] = {
                        "chunk_data": chunk,
                        "total_score": opensearch_score,
                        "query_count": 1,
                        "final_score": opensearch_score,
                        "character_count": character_count,
                    }

        # Apply fair document distribution to ensure smaller documents get representation
        unique_chunks = await _apply_fair_document_distribution(
            chunk_scores, rag_request.document_uuids, GLOBAL_CHARACTER_LIMIT
        )

        final_character_count = sum(len(chunk.get("_source", {}).get("chunk_content", "")) for chunk in unique_chunks)
        logger.info(
            "Consolidated %s chunks from %s queries into %s chunks (%s characters). "
            "Chunks appearing in multiple queries: %s",
            len(chunk_scores),
            len(rewritten_queries),
            len(unique_chunks),
            final_character_count,
            sum(1 for item in chunk_scores.values() if item["query_count"] > 1),
        )

    total_character_count_final = sum(len(chunk.get("_source", {}).get("chunk_content", "")) for chunk in unique_chunks)
    logger.info(
        "Retrieved %s chunks with %s total characters for user query", len(unique_chunks), total_character_count_final
    )
    return unique_chunks


async def _message_search_index_mapping_record(message: Message) -> tuple[MessageSearchIndexMapping, SearchIndex]:
    """
    Creates a search index mapping record for the given message and retrieves the associated search index.
    Logs the mapping of the message to personal document upload index.

    Args:
        message (Message): The chat message for which the search index mapping record is created.

    Returns:
        tuple: containing:
            - `MessageSearchIndexMapping`: The record linking the message to the search index.
            - `SearchIndex`: The associated search index object.

    """
    async with async_db_session() as session:
        execute = await session.execute(select(SearchIndex).where(SearchIndex.name == PERSONAL_DOCUMENTS_INDEX_NAME))
        pdu_search_index = execute.scalar()

        # create a search index mapping record to log personal index used for the message
        stmt = (
            insert(MessageSearchIndexMapping)
            .values(
                search_index_id=pdu_search_index.id,
                message_id=message.id,
                llm_internal_response_id=None,
                use_index=True,
            )
            .returning(MessageSearchIndexMapping)
        )
        result = await session.execute(stmt)
        pdu_message_search_index_mapping = result.scalars().first()
        logger.info("Added index mapping %s for message %s", pdu_search_index.id, message.id)
    return pdu_message_search_index_mapping, pdu_search_index


async def _convert_to_retrieval_results(
    document_chunks: List[Dict], search_index: SearchIndex, message: Message
) -> List[RetrievalResult]:
    # associate document chunks with the user message and save in the database
    async with async_db_session() as db_session:
        document_chunks_retrieved = await get_document_chunk_mappings(
            document_chunks, search_index, message.id, db_session, use_chunk=True
        )

    return document_chunks_retrieved


async def personal_document_rag(
    rag_request: RagRequest,
    message: Message,
    db_session: AsyncSession,
) -> Dict:
    # create record for personal index usage
    message_search_index_mapping, pdu_search_index = await _message_search_index_mapping_record(message)

    relevant_chunks = await _retrieve_relevant_chunks(rag_request, pdu_search_index, message, db_session)

    retrieval_results = await _convert_to_retrieval_results(relevant_chunks, pdu_search_index, message)
    personal_document_rag_result = {
        "retrieval_results": retrieval_results,
        "message_search_index_mappings": [message_search_index_mapping],
    }
    return personal_document_rag_result
