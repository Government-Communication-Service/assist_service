import asyncio
import logging
from typing import List, Optional

from anthropic.types import ToolUseBlock
from sqlalchemy import insert, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.bedrock import BedrockHandler, RunMode
from app.bedrock.service import calculate_completion_cost
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
from app.config import LLM_CHUNK_REVIEWER, LLM_INDEX_ROUTER, LLM_OPENSEARCH_QUERY_GENERATOR
from app.database.models import (
    LLM,
    Document,
    DocumentChunk,
    LlmInternalResponse,
    MessageDocumentChunkMapping,
    MessageSearchIndexMapping,
    RewrittenQuery,
    SearchIndex,
)
from app.opensearch.service import AsyncOpenSearchOperations

logger = logging.getLogger(__name__)


# =============================================================================
# MAIN ENTRY POINT - Must maintain exact interface for compatibility
# =============================================================================


async def search_central_guidance(query: str, message_id: int, db_session: AsyncSession):
    """
    Runs RAG process to search information related to the user's query and
    inject that information to the user query sent to the LLM model.

    This is the main entry point called by chat_create_message.
    """
    logger.debug("starting to run rag with query: %s", query)
    try:
        # Step 1: Get index and check relevance
        index = await get_central_guidance_index(db_session)
        index_mapping = await check_index_relevance(query, index, message_id, db_session)

        # Step 2: Search and filter (if relevant)
        if index_mapping and index_mapping.use_index:
            retrieval_results = await search_and_filter_chunks(query, index, message_id, db_session)
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
    query: str, index: SearchIndex, message_id: int, db_session: AsyncSession
) -> Optional[MessageSearchIndexMapping]:
    """Use LLM to determine if the central guidance index is relevant to the user's query."""
    try:
        logger.info("Checking if query requires rag from central guidance index...")

        # Get LLM for index routing
        execute = await db_session.execute(select(LLM).filter(LLM.model == LLM_INDEX_ROUTER))
        llm = execute.scalar_one()

        # Use modern tool-based approach for evaluation
        bedrock_handler = BedrockHandler(llm, mode=RunMode.ASYNC)

        # Prepare the evaluation message with structured format
        evaluation_message = (
            f"<User-Query>{query}</User-Query>\n\n"
            f"<Search-Index>\n"
            f"<Index-Name>{index.name}</Index-Name>\n"
            f"<Index-Description>{index.description}</Index-Description>\n"
            f"</Search-Index>"
        )

        response = await bedrock_handler.invoke_async(
            max_tokens=llm.max_tokens,
            model=f"us.{llm.model}",
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

        # Save LLM response with usage tracking
        llm_response = await save_llm_response(
            db_session, llm, str(response.content), response.usage.input_tokens, response.usage.output_tokens
        )

        # Create and save index mapping
        stmt = (
            insert(MessageSearchIndexMapping)
            .values(
                search_index_id=index.id,
                message_id=message_id,
                llm_internal_response_id=llm_response.id,
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
    query: str, index: SearchIndex, message_id: int, db_session: AsyncSession
) -> List[RetrievalResult]:
    """Search the index with rewritten queries and filter results using LLM evaluation."""
    logger.info("Retrieving relevant chunks from central guidance index...")

    # Step 1: Generate rewritten queries using LLM
    rewritten_queries = await generate_rewritten_queries(query, index, message_id, db_session)

    # Step 2: Search with each rewritten query
    all_raw_results = []
    for rewritten_query in rewritten_queries:
        chunks = await AsyncOpenSearchOperations.search_for_chunks(rewritten_query, index.name)
        results = await create_chunk_mappings(chunks, index, message_id, db_session)
        all_raw_results.extend(results)

    # Step 3: Filter results using LLM evaluation
    if not all_raw_results:
        return []

    logger.debug(f"Filtering {len(all_raw_results)} retrieval results")

    # Process all results concurrently with LLM evaluation
    tasks = [evaluate_chunk_relevance(result, query, db_session) for result in all_raw_results]
    processed_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and log errors
    valid_results = []
    for result in processed_results:
        if isinstance(result, Exception):
            logger.error(f"Error processing retrieval result: {result}")
        else:
            valid_results.append(result)

    # Step 4: Filter and deduplicate results
    # Filter out results marked as not useful
    relevant_results = [result for result in valid_results if result.message_document_chunk_mapping.use_document_chunk]

    # Deduplicate by document chunk ID
    unique_results = {}
    for result in relevant_results:
        chunk_id = result.document_chunk.id
        if chunk_id not in unique_results:
            unique_results[chunk_id] = result

    final_results = list(unique_results.values())
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
    query: str, index: SearchIndex, message_id: int, db_session: AsyncSession
) -> List[str]:
    """Generate rewritten queries using LLM for better search results."""
    logger.info("Generating OpenSearch queries...")

    # Get LLM for query generation
    execute = await db_session.execute(select(LLM).filter(LLM.model == LLM_OPENSEARCH_QUERY_GENERATOR))
    llm = execute.scalar_one()

    # Use LLM to generate rewritten queries
    bedrock_handler = BedrockHandler(llm, mode=RunMode.ASYNC)
    response = await bedrock_handler.invoke_async(
        max_tokens=llm.max_tokens,
        model=f"us.{llm.model}",
        system=SYSTEM_PROMPT_OPENSEARCH_QUERY_GENERATOR,
        messages=[{"role": "user", "content": query}],
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

    # Save LLM response for analytics
    llm_response = await save_llm_response(
        db_session, llm, str(response.content), response.usage.input_tokens, response.usage.output_tokens
    )

    # Save rewritten queries for analytics
    query_models = [
        {
            "search_index_id": index.id,
            "message_id": message_id,
            "llm_internal_response_id": llm_response.id,
            "content": rewritten_query,
        }
        for rewritten_query in opensearch_queries
    ]
    await db_session.execute(insert(RewrittenQuery), query_models)

    return opensearch_queries


async def evaluate_chunk_relevance(
    retrieval_result: RetrievalResult, user_query: str, db_session: AsyncSession
) -> RetrievalResult:
    """Evaluate a single chunk's relevance using LLM."""
    doc_chunk = retrieval_result.document_chunk
    document = retrieval_result.document
    mapping = retrieval_result.message_document_chunk_mapping

    # Get LLM for chunk evaluation
    execute = await db_session.execute(select(LLM).filter(LLM.model == LLM_CHUNK_REVIEWER))
    llm = execute.scalar_one()

    # Use modern tool-based approach for evaluation
    bedrock_handler = BedrockHandler(llm, mode=RunMode.ASYNC)

    # Prepare the evaluation message
    evaluation_message = (
        f"<User-Query>{user_query}</User-Query>\n\n"
        f"<Document>\n"
        f"<Document-Title>{document.name}</Document-Title>\n"
        f"<Section-Title>{doc_chunk.name}</Section-Title>\n"
        f"<Content>{doc_chunk.content}</Content>"
        f"</Document>\n"
    )

    try:
        response = await bedrock_handler.invoke_async(
            max_tokens=llm.max_tokens,
            model=f"us.{llm.model}",
            system=SYSTEM_PROMPT_CHUNK_RELEVANCE_EVALUATOR,
            messages=[{"role": "user", "content": evaluation_message}],
            tools=[TOOL_CHUNK_RELEVANCE_EVALUATOR],
            tool_choice={"type": "tool", "name": TOOL_NAME_CHUNK_RELEVANCE_EVALUATOR},
        )
    except Exception as e:
        logger.exception(f"Error invoking LLM for chunk evaluation: {e}")
        raise

    # Extract tool response
    use_chunk = False
    reasoning = "Error parsing response"

    for block in response.content:
        if isinstance(block, ToolUseBlock):
            tool_input = block.input
            use_chunk = tool_input.get("is_relevant", False)
            reasoning = tool_input.get("reasoning", "No reasoning provided")
            break

    logger.debug(f"Chunk evaluation result: use_chunk={use_chunk}, reasoning={reasoning}")

    # Save LLM response to database for analytics
    llm_response = await save_llm_response(
        db_session, llm, str(response.content), response.usage.input_tokens, response.usage.output_tokens
    )

    # Update chunk mapping with LLM decision
    updated_mapping = await update_chunk_mapping(db_session, mapping.id, llm_response.id, use_chunk)

    return RetrievalResult(
        search_index=retrieval_result.search_index,
        document_chunk=doc_chunk,
        document=document,
        message_document_chunk_mapping=updated_mapping,
    )


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================


async def create_chunk_mappings(
    chunks: List[dict], index: SearchIndex, message_id: int, db_session: AsyncSession
) -> List[RetrievalResult]:
    """Create database mappings for document chunks and return RetrievalResult objects."""
    retrieval_results = []
    for hit in chunks:
        id_opensearch = hit["_id"]
        logger.debug("create_chunk_mappings-id_opensearch %s", id_opensearch)

        execute = await db_session.execute(select(DocumentChunk).filter(DocumentChunk.id_opensearch == id_opensearch))
        doc_chunk = execute.scalar_one_or_none()
        if not doc_chunk:
            logger.warning(f"DocumentChunk not found for id_opensearch: {id_opensearch}")
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


async def save_llm_response(
    db_session: AsyncSession, llm: LLM, content: str, input_tokens: int, output_tokens: int
) -> LlmInternalResponse:
    """Save LLM response and usage to database for analytics."""
    stmt = (
        insert(LlmInternalResponse)
        .values(
            llm_id=llm.id,
            content=content,
            tokens_in=input_tokens,
            tokens_out=output_tokens,
            completion_cost=calculate_completion_cost(llm, input_tokens, output_tokens),
        )
        .returning(LlmInternalResponse)
    )

    result = await db_session.execute(stmt)
    return result.scalar_one()


async def update_chunk_mapping(
    db_session: AsyncSession, mapping_id: int, llm_response_id: int, use_chunk: bool
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
    """Compile prompt segments when we have retrieval results."""
    citations = {}
    prompt_parts = ["<government-comms-central-guidance-search-results>"]

    for i, result in enumerate(retrieval_results):
        document = result.document
        doc_chunk = result.document_chunk

        # Build citation
        citations[str(document.uuid)] = {"docname": document.name, "docurl": document.url}

        # Build content reference
        content_ref = (
            f"<document-title>{document.name}</document-title>\n"
            f"<section-title>{doc_chunk.name}</section-title>\n"
            f"<content>{doc_chunk.content}</content>"
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
