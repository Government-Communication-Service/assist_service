import logging
from typing import Dict, List, Optional

from anthropic.types import ToolUseBlock
from sqlalchemy import insert, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.bedrock import BedrockHandler, RunMode
from app.central_guidance.constants import (
    CENTRAL_RAG_INDEX_NAME,
    SYSTEM_PROMPT_CHUNK_RELEVANCE_EVALUATOR,
    SYSTEM_PROMPT_INDEX_RELEVANCE_EVALUATOR,
    SYSTEM_PROMPT_OPENSEARCH_QUERY_GENERATOR,
    TOOL_CHUNK_RELEVANCE_EVALUATOR,
    TOOL_INDEX_RELEVANCE_EVALUATOR,
    TOOL_NAME_CHUNK_RELEVANCE_EVALUATOR,
    TOOL_NAME_INDEX_RELEVANCE_EVALUATOR,
    TOOL_NAME_OPENSEARCH_QUERY_GENERATOR,
    TOOL_OPENSEARCH_QUERY_GENERATOR,
)
from app.central_guidance.schemas import RetrievalResult
from app.chat.utils import prepare_recent_turns_for_decision
from app.config import (
    LLM_CHUNK_REVIEWER,
    LLM_INDEX_ROUTER,
    LLM_OPENSEARCH_QUERY_GENERATOR,
    MAX_CENTRAL_GUIDANCE_CHUNK_CHARS,
    MAX_CENTRAL_GUIDANCE_RESULTS,
)
from app.database.models import (
    LLM,
    Document,
    DocumentChunk,
    Message,
    MessageDocumentChunkMapping,
    MessageSearchIndexMapping,
    RewrittenQuery,
    SearchIndex,
)
from app.opensearch.service import AsyncOpenSearchOperations

logger = logging.getLogger(__name__)

# How many recent messages to give the index router, query rewriter, and chunk evaluator, so
# they can resolve follow-up references (pronouns, "that", "the third one") in the query.
RECENT_TURNS_FOR_DECISION = 6


def _build_recent_context(messages: Optional[List[Message]]) -> str:
    """Format recent raw conversation turns as an XML block for decision-making prompts."""
    if not messages:
        return ""
    recent = prepare_recent_turns_for_decision(messages, num_turns=RECENT_TURNS_FOR_DECISION)
    if not recent:
        return ""
    turns = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)
    return f"<recent-conversation>\n{turns}\n</recent-conversation>\n\n"


# =============================================================================
# MAIN ENTRY POINT - Must maintain exact interface for compatibility
# =============================================================================


async def search_central_guidance(
    query: str, message_id: int, db_session: AsyncSession, messages: Optional[List[Message]] = None
):
    """
    Runs RAG process to search information related to the user's query and
    inject that information to the user query sent to the LLM model.

    This is the main entry point called by chat_create_message.

    messages: prior messages in the chat (not including this turn's), used to resolve
    follow-up references in the query for the index router, query rewriter, and chunk
    evaluator - not passed on to the final compiled prompt segment.
    """
    logger.debug("starting to run rag with query: %s", query)
    try:
        recent_context = _build_recent_context(messages)

        # Step 1: Get index and check relevance
        index = await get_central_guidance_index(db_session)
        index_mapping = await check_index_relevance(query, index, message_id, db_session, recent_context)

        # Step 2: Search and filter (if relevant)
        if index_mapping and index_mapping.use_index:
            retrieval_results = await search_and_filter_chunks(query, index, message_id, db_session, recent_context)
        else:
            retrieval_results = []

        # Step 3: Compile results
        mappings = [index_mapping] if index_mapping else []
        prompt_segment, citations = await compile_results(retrieval_results, mappings, db_session)

        logger.info(f"Central guidance citation message: {citations}")
        return (prompt_segment, citations)

    except Exception:
        logger.exception("Rag process got error - returning empty results for graceful degradation")
        return ("", [])


# =============================================================================
# CORE OPERATIONS
# =============================================================================


async def get_central_guidance_index(db_session: AsyncSession) -> SearchIndex:
    """Get the central guidance search index from database."""
    search_index_query = await db_session.execute(
        select(SearchIndex).filter(SearchIndex.name == CENTRAL_RAG_INDEX_NAME, SearchIndex.deleted_at.is_(None))
    )
    return search_index_query.scalar_one()


async def check_index_relevance(
    query: str, index: SearchIndex, message_id: int, db_session: AsyncSession, recent_context: str = ""
) -> Optional[MessageSearchIndexMapping]:
    """Use LLM to determine if the central guidance index is relevant to the user's query."""
    try:
        logger.info("Checking if query requires rag from central guidance index...")

        # Get LLM for index routing
        execute = await db_session.execute(select(LLM).filter(LLM.model == LLM_INDEX_ROUTER))
        llm = execute.scalar_one()

        # Use modern tool-based approach for evaluation
        bedrock_handler = BedrockHandler(llm=llm, mode=RunMode.ASYNC)

        # Prepare the evaluation message with structured format
        evaluation_message = (
            f"{recent_context}"
            f"<User-Query>{query}</User-Query>\n\n"
            f"<Search-Index>\n"
            f"<Index-Name>{index.name}</Index-Name>\n"
            f"<Index-Description>{index.description}</Index-Description>\n"
            f"</Search-Index>"
        )

        response = await bedrock_handler.invoke_async(
            db_session=db_session,
            max_tokens=llm.max_tokens,
            system=SYSTEM_PROMPT_INDEX_RELEVANCE_EVALUATOR,
            messages=[{"role": "user", "content": evaluation_message}],
            tools=[TOOL_INDEX_RELEVANCE_EVALUATOR],
            tool_choice={"type": "tool", "name": TOOL_NAME_INDEX_RELEVANCE_EVALUATOR},
        )

        # Extract tool response
        requires_index = False
        reasoning = "Error parsing response"

        for block in response.content:
            if isinstance(block, ToolUseBlock):
                tool_input = block.input
                requires_index = tool_input.get("requires_index", False)
                reasoning = tool_input.get("reasoning", "No reasoning provided")
                break

        logger.info(f"Index relevance decision: requires_index={requires_index}, reasoning={reasoning}")

        # Create and save index mapping
        stmt = (
            insert(MessageSearchIndexMapping)
            .values(
                search_index_id=index.id,
                message_id=message_id,
                llm_internal_response_id=response.llm_internal_response_id,
                use_index=requires_index,
            )
            .returning(MessageSearchIndexMapping)
        )

        result = await db_session.execute(stmt)
        return result.scalar_one()

    except Exception:
        logger.exception("Error checking index relevance")
        return None


async def search_and_filter_chunks(
    query: str, index: SearchIndex, message_id: int, db_session: AsyncSession, recent_context: str = ""
) -> List[RetrievalResult]:
    """Search the index with rewritten queries and filter results using LLM evaluation."""
    logger.info("Retrieving relevant chunks from central guidance index...")

    # Step 1: Generate rewritten queries using LLM
    rewritten_queries = await generate_rewritten_queries(query, index, message_id, db_session, recent_context)

    # Step 2: Search with each rewritten query, deduplicating hits by OpenSearch ID up front
    # so the same chunk isn't evaluated twice. Keep the highest score seen across queries.
    unique_hits_by_id: Dict[str, dict] = {}
    for rewritten_query in rewritten_queries:
        chunks = await AsyncOpenSearchOperations.search_for_chunks(rewritten_query, index.name)
        for hit in chunks:
            existing = unique_hits_by_id.get(hit["_id"])
            if existing is None or hit["_score"] > existing["_score"]:
                unique_hits_by_id[hit["_id"]] = hit

    if not unique_hits_by_id:
        return []

    # Step 3: Create mappings only for chunks not already cited earlier in this chat - avoids
    # an insert immediately followed by an update for chunks we already know to discard.
    previously_cited_chunk_ids = await get_previously_cited_chunk_ids(db_session, message_id)
    candidates = await create_chunk_mappings(
        list(unique_hits_by_id.values()), index, message_id, db_session, previously_cited_chunk_ids
    )
    if not candidates:
        return []

    logger.debug(f"Filtering {len(candidates)} unique, not-yet-cited retrieval results")

    # Step 4: Evaluate all remaining candidate chunks' relevance in a single LLM call.
    try:
        evaluated_results = await evaluate_chunks_relevance(candidates, query, db_session, recent_context)
    except Exception:
        logger.exception("Error evaluating chunk relevance")
        return []

    final_results = [result for result in evaluated_results if result.message_document_chunk_mapping.use_document_chunk]
    logger.info(f"Processed results from central guidance index: {len(final_results)} chunks")
    return final_results


async def compile_results(
    retrieval_results: List[RetrievalResult],
    message_search_index_mappings: List[MessageSearchIndexMapping],
    db_session: AsyncSession,
) -> tuple[str, list]:
    """Compile prompt segments and citations from retrieval results."""
    if retrieval_results:
        return compile_results_with_citations(retrieval_results)

    # Handle case where no results found but index was searched
    return await compile_no_results_message(message_search_index_mappings, db_session)


# =============================================================================
# LLM OPERATIONS (kept separate for maintainability)
# =============================================================================


async def generate_rewritten_queries(
    query: str, index: SearchIndex, message_id: int, db_session: AsyncSession, recent_context: str = ""
) -> List[str]:
    """Generate rewritten queries using LLM for better search results."""
    logger.info("Generating OpenSearch queries...")

    # Get LLM for query generation
    execute = await db_session.execute(select(LLM).filter(LLM.model == LLM_OPENSEARCH_QUERY_GENERATOR))
    llm = execute.scalar_one()

    # Use LLM to generate rewritten queries
    bedrock_handler = BedrockHandler(llm=llm, mode=RunMode.ASYNC)
    response = await bedrock_handler.invoke_async(
        db_session=db_session,
        max_tokens=llm.max_tokens,
        system=SYSTEM_PROMPT_OPENSEARCH_QUERY_GENERATOR,
        messages=[{"role": "user", "content": f"{recent_context}{query}"}],
        tools=[TOOL_OPENSEARCH_QUERY_GENERATOR],
        tool_choice={"type": "tool", "name": TOOL_NAME_OPENSEARCH_QUERY_GENERATOR},
    )

    # Extract queries from tool response
    opensearch_queries = []
    for block in response.content:
        if isinstance(block, ToolUseBlock):
            opensearch_queries = block.input["keyword_queries"]
            break

    logger.info(f"OpenSearch keyword queries generated by LLM: {opensearch_queries}")

    # Save rewritten queries for analytics
    query_models = [
        {
            "search_index_id": index.id,
            "message_id": message_id,
            "llm_internal_response_id": response.llm_internal_response_id,
            "content": rewritten_query,
        }
        for rewritten_query in opensearch_queries
    ]
    await db_session.execute(insert(RewrittenQuery), query_models)

    return opensearch_queries


async def evaluate_chunks_relevance(
    retrieval_results: List[RetrievalResult], user_query: str, db_session: AsyncSession, recent_context: str = ""
) -> List[RetrievalResult]:
    """Evaluate every candidate chunk's relevance to the query in a single LLM call."""
    # Get LLM for chunk evaluation
    execute = await db_session.execute(select(LLM).filter(LLM.model == LLM_CHUNK_REVIEWER))
    llm = execute.scalar_one()

    # Use modern tool-based approach for evaluation
    bedrock_handler = BedrockHandler(llm=llm, mode=RunMode.ASYNC)

    chunk_blocks = [
        f"<chunk-{i}>\n"
        f"<document-title>{result.document.name}</document-title>\n"
        f"<section-title>{result.document_chunk.name}</section-title>\n"
        f"<content>{result.document_chunk.content[:MAX_CENTRAL_GUIDANCE_CHUNK_CHARS]}</content>\n"
        f"</chunk-{i}>"
        for i, result in enumerate(retrieval_results)
    ]
    evaluation_message = (
        f"{recent_context}<User-Query>{user_query}</User-Query>\n\n<Document-Chunks>\n"
        + "\n".join(chunk_blocks)
        + "\n</Document-Chunks>"
    )

    try:
        response = await bedrock_handler.invoke_async(
            db_session=db_session,
            max_tokens=llm.max_tokens,
            system=SYSTEM_PROMPT_CHUNK_RELEVANCE_EVALUATOR,
            messages=[{"role": "user", "content": evaluation_message}],
            tools=[TOOL_CHUNK_RELEVANCE_EVALUATOR],
            tool_choice={"type": "tool", "name": TOOL_NAME_CHUNK_RELEVANCE_EVALUATOR},
        )
    except Exception as e:
        logger.exception(f"Error invoking LLM for chunk evaluation: {e}")
        raise

    # A chunk missing from the response defaults to not relevant. chunk_index is coerced to
    # int, since the model doesn't strictly honor the declared integer type.
    evaluations_by_index: Dict[int, dict] = {}
    for block in response.content:
        if isinstance(block, ToolUseBlock):
            for evaluation in block.input.get("evaluations", []):
                chunk_index = evaluation.get("chunk_index")
                if chunk_index is not None:
                    try:
                        evaluations_by_index[int(chunk_index)] = evaluation
                    except (TypeError, ValueError):
                        logger.warning(f"Chunk evaluation had non-integer chunk_index: {chunk_index!r}")
            break

    updated_results = []
    for i, result in enumerate(retrieval_results):
        evaluation = evaluations_by_index.get(i, {})
        is_relevant_raw = evaluation.get("is_relevant", False)
        # If the LLM returns a texty 'True'/'False' value here we need to coerce to boolean
        use_chunk = is_relevant_raw is True or str(is_relevant_raw).lower() == "true"
        reasoning = evaluation.get("reasoning", "No reasoning provided")

        logger.debug(f"Chunk {i} evaluation result: use_chunk={use_chunk}, reasoning={reasoning}")

        updated_mapping = await update_chunk_mapping(
            db_session, result.message_document_chunk_mapping.id, response.llm_internal_response_id, use_chunk
        )
        updated_results.append(
            RetrievalResult(
                search_index=result.search_index,
                document_chunk=result.document_chunk,
                document=result.document,
                message_document_chunk_mapping=updated_mapping,
            )
        )

    return updated_results


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================


async def create_chunk_mappings(
    chunks: List[dict],
    index: SearchIndex,
    message_id: int,
    db_session: AsyncSession,
    skip_chunk_ids: Optional[set] = None,
) -> List[RetrievalResult]:
    """Create database mappings for document chunks and return RetrievalResult objects.

    Chunks whose DocumentChunk.id is in skip_chunk_ids are resolved (to check membership)
    but no mapping row is created for them - used to avoid mapping rows for chunks already
    known to be discarded (e.g. already cited earlier in this chat).
    """
    retrieval_results = []
    for hit in chunks:
        id_opensearch = hit["_id"]
        logger.debug("create_chunk_mappings-id_opensearch %s", id_opensearch)

        execute = await db_session.execute(select(DocumentChunk).filter(DocumentChunk.id_opensearch == id_opensearch))
        doc_chunk = execute.scalar_one_or_none()
        if not doc_chunk:
            logger.warning(f"DocumentChunk not found for id_opensearch: {id_opensearch}")
            continue

        if skip_chunk_ids and doc_chunk.id in skip_chunk_ids:
            logger.debug(f"Skipping chunk {doc_chunk.id} - already cited earlier in this chat")
            continue

        # Create mapping for analytics
        message_document_chunk_mapping = MessageDocumentChunkMapping(
            message_id=message_id, document_chunk_id=doc_chunk.id, opensearch_score=hit["_score"]
        )
        db_session.add(message_document_chunk_mapping)

        # Get associated document
        execute = await db_session.execute(select(Document).filter(Document.id == doc_chunk.document_id))
        document = execute.scalar_one()

        retrieval_result = RetrievalResult(
            search_index=index,
            document_chunk=doc_chunk,
            document=document,
            message_document_chunk_mapping=message_document_chunk_mapping,
        )

        retrieval_results.append(retrieval_result)

    return retrieval_results


async def get_previously_cited_chunk_ids(db_session: AsyncSession, message_id: int) -> set:
    """Document chunk IDs already cited (and judged relevant) earlier in this message's chat."""
    stmt = (
        select(MessageDocumentChunkMapping.document_chunk_id)
        .join(Message, Message.id == MessageDocumentChunkMapping.message_id)
        .where(
            Message.chat_id == select(Message.chat_id).where(Message.id == message_id).scalar_subquery(),
            Message.id != message_id,
            MessageDocumentChunkMapping.use_document_chunk.is_(True),
        )
        .distinct()
    )
    result = await db_session.execute(stmt)
    return set(result.scalars().all())


async def update_chunk_mapping(
    db_session: AsyncSession, mapping_id: int, llm_response_id: Optional[int], use_chunk: bool
) -> MessageDocumentChunkMapping:
    """Update document chunk mapping with LLM decision for analytics."""
    stmt = (
        update(MessageDocumentChunkMapping)
        .where(MessageDocumentChunkMapping.id == mapping_id)
        .values(llm_internal_response_id=llm_response_id, use_document_chunk=use_chunk)
        .returning(MessageDocumentChunkMapping)
    )

    result = await db_session.execute(stmt)
    return result.scalar_one()


# =============================================================================
# RESULT COMPILATION
# =============================================================================


def compile_results_with_citations(retrieval_results: List[RetrievalResult]) -> tuple[str, list]:
    """Compile prompt segments when we have retrieval results, keeping the highest-scoring chunks."""
    ranked_results = sorted(
        retrieval_results, key=lambda result: result.message_document_chunk_mapping.opensearch_score, reverse=True
    )[:MAX_CENTRAL_GUIDANCE_RESULTS]

    citations = {}
    prompt_parts = ["<government-comms-central-guidance-search-results>"]

    for i, result in enumerate(ranked_results):
        document = result.document
        doc_chunk = result.document_chunk

        # Build citation
        citations[str(document.uuid)] = {"docname": document.name, "docurl": document.url}

        content = doc_chunk.content
        if len(content) > MAX_CENTRAL_GUIDANCE_CHUNK_CHARS:
            content = content[:MAX_CENTRAL_GUIDANCE_CHUNK_CHARS] + "... [content truncated]"

        # Build content reference
        content_ref = (
            f"<document-title>{document.name}</document-title>\n"
            f"<section-title>{doc_chunk.name}</section-title>\n"
            f"<content>{content}</content>"
        )
        prompt_parts.append(f"\n<result-{i}>\n{content_ref}\n</result-{i}>")

    prompt_parts.append("\n</government-comms-central-guidance-search-results>")

    return ("\n".join(prompt_parts), list(citations.values()))


async def compile_no_results_message(
    message_search_index_mappings: List[MessageSearchIndexMapping], db_session: AsyncSession
) -> tuple[str, list]:
    """Compile message when no results found but the central guidance index was searched."""
    searched_index_ids = [mapping.search_index_id for mapping in message_search_index_mappings if mapping.use_index]

    if not searched_index_ids:
        return ("", [])

    # Get documents that were searched but yielded no results
    prompt_parts = [
        "<government-comms-central-guidance-search-results>",
        "The following document(s) were searched but no relevant material was found:",
    ]

    for index_id in searched_index_ids:
        documents = await get_documents_for_index(db_session, index_id)
        for document in documents:
            prompt_parts.append(f"\n<document-title>{document.name}</document-title>")

    prompt_parts.append("\n</government-comms-central-guidance-search-results>")

    return ("\n".join(prompt_parts), [])


async def get_documents_for_index(db_session: AsyncSession, index_id: int) -> List[Document]:
    """Get all documents associated with the central guidance search index."""
    stmt = (
        select(Document)
        .distinct(Document.id)
        .join(DocumentChunk, DocumentChunk.document_id == Document.id)
        .join(SearchIndex, SearchIndex.id == DocumentChunk.search_index_id)
        .where(SearchIndex.id == index_id)
    )
    result = await db_session.execute(stmt)
    return result.scalars().all()
