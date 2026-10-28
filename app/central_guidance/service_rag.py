import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import ValidationError
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.bedrock import BedrockHandler, RunMode
from app.bedrock.call_log import LlmCallContext
from app.bedrock.tool_schema import ToolResponseError
from app.central_guidance.constants import CENTRAL_RAG_INDEX_NAME
from app.central_guidance.prompts import (
    SYSTEM_PROMPT_CHUNK_RELEVANCE_EVALUATOR,
    TOOL_RESULT_NO_QUERIES,
    TOOL_RESULT_NO_RELEVANT_PASSAGES,
    TOOL_RESULT_RELEVANCE_CHECK_FAILED,
    already_shown_note,
    unchecked_note,
    wrap_tool_result,
)
from app.central_guidance.schemas import RetrievalResult
from app.central_guidance.tools import CHUNK_RELEVANCE
from app.chat.utils import prepare_recent_turns_for_decision
from app.config import (
    CENTRAL_GUIDANCE_EVALUATOR_RECENT_TURNS,
    CENTRAL_GUIDANCE_MAX_TOOL_QUERIES,
    LLM_CHUNK_REVIEWER,
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
from app.database.table import async_db_session
from app.opensearch.service import AsyncOpenSearchOperations

logger = logging.getLogger(__name__)


def _build_recent_context(messages: Optional[List[Message]]) -> str:
    """Format recent raw conversation turns as an XML block for decision-making prompts."""
    if not messages:
        return ""
    recent = prepare_recent_turns_for_decision(messages, num_turns=CENTRAL_GUIDANCE_EVALUATOR_RECENT_TURNS)
    if not recent:
        return ""
    turns = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)
    return f"<recent-conversation>\n{turns}\n</recent-conversation>\n\n"


# =============================================================================
# MAIN ENTRY POINT - search_central_guidance tool
# =============================================================================


async def search_central_guidance_for_tool(
    queries: list[str],
    message_id: int,
    user_query: str,
    llm_internal_response_id: int | None,
    db_session: AsyncSession,
    recent_messages: Optional[List[Message]] = None,
    remove_duplicates: bool = False,
) -> tuple[str, list[dict], bool]:
    """Search central guidance with the chat model's own keyword queries.

    Holds no DB connection during the OpenSearch and relevance-check calls: the lookups run first
    in their own short session, and db_session is only used at the end, to record the search
    (message_search_index_mapping), the queries (rewritten_query) and each chunk's relevance
    decision (message_document_chunk_mapping).

    Args:
        queries: Keyword queries from the model; blanks and duplicates dropped, first
            CENTRAL_GUIDANCE_MAX_TOOL_QUERIES used.
        message_id: This turn's user message id.
        user_query: This turn's user message text, for the chunk evaluator.
        llm_internal_response_id: The logged model call that requested the search.
        db_session: Session to record analytics in. Not touched until the network calls are done.
        recent_messages: Earlier messages in the chat, for the chunk evaluator.
        remove_duplicates: Skip chunks already cited earlier in the chat, this turn included.

    Returns:
        The tool result text, the citations for the passages returned, and whether the search
        failed (the relevance check gave no decision for any candidate).
    """
    unique_queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))[:CENTRAL_GUIDANCE_MAX_TOOL_QUERIES]
    if not unique_queries:
        return wrap_tool_result(TOOL_RESULT_NO_QUERIES), [], False

    unique_hits_by_id: Dict[str, dict] = {}
    for query in unique_queries:
        for hit in await AsyncOpenSearchOperations.search_for_chunks(query, CENTRAL_RAG_INDEX_NAME):
            existing = unique_hits_by_id.get(hit["_id"])
            if existing is None or hit["_score"] > existing["_score"]:
                unique_hits_by_id[hit["_id"]] = hit

    async with async_db_session() as read_session:
        index = await get_central_guidance_index(read_session)
        previously_cited_chunk_ids = (
            await get_previously_cited_chunk_ids(read_session, message_id, include_message=True)
            if remove_duplicates
            else set()
        )
        candidates = await resolve_chunks(
            read_session, list(unique_hits_by_id.values()), index, message_id, previously_cited_chunk_ids
        )
        llm = (await read_session.execute(select(LLM).filter(LLM.model == LLM_CHUNK_REVIEWER))).scalar_one()
    already_shown = len(unique_hits_by_id) - len(candidates)

    decisions: list[bool | None] = []
    if candidates:
        relevance = await evaluate_chunks_relevance(
            candidates,
            user_query,
            llm,
            _build_recent_context(recent_messages),
            LlmCallContext(purpose="central_guidance_relevance", message_id=message_id),
        )
        decisions = relevance.decisions
        for candidate, decision in zip(candidates, decisions, strict=True):
            candidate.message_document_chunk_mapping.llm_internal_response_id = relevance.llm_internal_response_id
            candidate.message_document_chunk_mapping.use_document_chunk = bool(decision)

    db_session.add(
        MessageSearchIndexMapping(
            search_index_id=index.id,
            message_id=message_id,
            llm_internal_response_id=llm_internal_response_id,
            use_index=True,
        )
    )
    await db_session.execute(
        insert(RewrittenQuery),
        [
            {
                "search_index_id": index.id,
                "message_id": message_id,
                "llm_internal_response_id": llm_internal_response_id,
                "content": query,
            }
            for query in unique_queries
        ],
    )
    db_session.add_all(c.message_document_chunk_mapping for c in candidates)

    relevant_results = [c for c, decision in zip(candidates, decisions, strict=True) if decision]
    unchecked = sum(decision is None for decision in decisions)
    logger.info(
        f"Central guidance tool search: {len(unique_hits_by_id)} hits, {already_shown} already cited, "
        f"{len(relevant_results)} relevant, {unchecked} unchecked"
    )

    if candidates and unchecked == len(candidates):
        return wrap_tool_result(TOOL_RESULT_RELEVANCE_CHECK_FAILED), [], True

    notes = []
    if unchecked:
        notes.append(unchecked_note(unchecked))
    if relevant_results:
        content, citations = compile_results_with_citations(relevant_results, notes)
        return content, citations, False

    if already_shown:
        notes.append(already_shown_note(already_shown))
    return wrap_tool_result(" ".join([TOOL_RESULT_NO_RELEVANT_PASSAGES, *notes])), [], False


# =============================================================================
# CORE OPERATIONS
# =============================================================================


async def get_central_guidance_index(db_session: AsyncSession) -> SearchIndex:
    """Get the central guidance search index from database."""
    search_index_query = await db_session.execute(
        select(SearchIndex).filter(SearchIndex.name == CENTRAL_RAG_INDEX_NAME, SearchIndex.deleted_at.is_(None))
    )
    return search_index_query.scalar_one()


# =============================================================================
# LLM OPERATIONS (kept separate for maintainability)
# =============================================================================


@dataclass
class ChunkRelevance:
    """The relevance check's verdict on each candidate chunk.

    Args:
        decisions: One per candidate, in order: True if relevant, False if not, None if the
            response gave no valid decision for it.
        llm_internal_response_id: The logged evaluator call.
    """

    decisions: list[bool | None]
    llm_internal_response_id: int | None


async def evaluate_chunks_relevance(
    retrieval_results: List[RetrievalResult],
    user_query: str,
    llm: LLM,
    recent_context: str = "",
    context: LlmCallContext | None = None,
) -> ChunkRelevance:
    """Evaluate every candidate chunk's relevance to the query in a single LLM call. Uses no DB session."""
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
            context=context,
            max_tokens=llm.max_tokens,
            system=SYSTEM_PROMPT_CHUNK_RELEVANCE_EVALUATOR,
            messages=[{"role": "user", "content": evaluation_message}],
            tools=[CHUNK_RELEVANCE.spec],
            tool_choice=CHUNK_RELEVANCE.forced_choice,
        )
    except Exception as e:
        logger.exception(f"Error invoking LLM for chunk evaluation: {e}")
        raise

    try:
        evaluations_by_index = {e.chunk_index: e for e in CHUNK_RELEVANCE.parse_response(response).evaluations}
    except (ToolResponseError, ValidationError):
        logger.error(
            f"Chunk evaluation response didn't validate (stop_reason={getattr(response, 'stop_reason', None)!r}); "
            f"none of the {len(retrieval_results)} chunks could be checked",
            exc_info=True,
        )
        evaluations_by_index = {}

    decisions: list[bool | None] = []
    for i in range(len(retrieval_results)):
        evaluation = evaluations_by_index.get(i)
        decisions.append(evaluation.is_relevant if evaluation else None)
        logger.debug(
            f"Chunk {i} evaluation result: use_chunk={decisions[-1]}, "
            f"reasoning={evaluation.reasoning if evaluation else 'No evaluation returned'}"
        )

    missing = sum(d is None for d in decisions)
    if evaluations_by_index and missing:
        logger.warning(f"Chunk evaluation gave no decision for {missing} of {len(retrieval_results)} chunks")

    return ChunkRelevance(decisions=decisions, llm_internal_response_id=response.llm_internal_response_id)


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================


async def resolve_chunks(
    db_session: AsyncSession,
    hits: List[dict],
    index: SearchIndex,
    message_id: int,
    skip_chunk_ids: Optional[set] = None,
) -> List[RetrievalResult]:
    """Look up the chunk and document for each OpenSearch hit, in hit order.

    Each result carries a new, unsaved MessageDocumentChunkMapping for the caller to fill in and
    add. Hits with no DocumentChunk, or whose chunk id is in skip_chunk_ids (already cited earlier
    in this chat), are left out and get no mapping.
    """
    if not hits:
        return []
    rows = await db_session.execute(
        select(DocumentChunk, Document)
        .join(Document, Document.id == DocumentChunk.document_id)
        .filter(DocumentChunk.id_opensearch.in_([hit["_id"] for hit in hits]))
    )
    by_opensearch_id = {chunk.id_opensearch: (chunk, document) for chunk, document in rows.all()}

    retrieval_results = []
    for hit in hits:
        found = by_opensearch_id.get(hit["_id"])
        if found is None:
            logger.warning(f"DocumentChunk not found for id_opensearch: {hit['_id']}")
            continue
        doc_chunk, document = found
        if skip_chunk_ids and doc_chunk.id in skip_chunk_ids:
            logger.debug(f"Skipping chunk {doc_chunk.id} - already cited earlier in this chat")
            continue
        retrieval_results.append(
            RetrievalResult(
                search_index=index,
                document_chunk=doc_chunk,
                document=document,
                message_document_chunk_mapping=MessageDocumentChunkMapping(
                    message_id=message_id, document_chunk_id=doc_chunk.id, opensearch_score=hit["_score"]
                ),
            )
        )

    return retrieval_results


async def get_previously_cited_chunk_ids(
    db_session: AsyncSession, message_id: int, include_message: bool = False
) -> set:
    """Document chunk IDs already cited (and judged relevant) earlier in this message's chat.

    Args:
        db_session: Session to query.
        message_id: The current user message.
        include_message: Also count chunks cited for message_id itself (earlier tool calls this turn).
    """
    conditions = [
        Message.chat_id == select(Message.chat_id).where(Message.id == message_id).scalar_subquery(),
        MessageDocumentChunkMapping.use_document_chunk.is_(True),
    ]
    if not include_message:
        conditions.append(Message.id != message_id)
    stmt = (
        select(MessageDocumentChunkMapping.document_chunk_id)
        .join(Message, Message.id == MessageDocumentChunkMapping.message_id)
        .where(*conditions)
        .distinct()
    )
    result = await db_session.execute(stmt)
    return set(result.scalars().all())


# =============================================================================
# RESULT COMPILATION
# =============================================================================


def compile_results_with_citations(
    retrieval_results: List[RetrievalResult], notes: Optional[List[str]] = None
) -> tuple[str, list]:
    """Compile prompt segments when we have retrieval results, keeping the highest-scoring chunks.

    Args:
        retrieval_results: The relevant chunks.
        notes: Lines for the model, added after the results inside the results tag.
    """
    ranked_results = sorted(
        retrieval_results, key=lambda result: result.message_document_chunk_mapping.opensearch_score, reverse=True
    )[:MAX_CENTRAL_GUIDANCE_RESULTS]

    citations = {}
    prompt_parts = []

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
        prompt_parts.append(f"<result-{i}>\n{content_ref}\n</result-{i}>")

    prompt_parts.extend(notes or [])

    return wrap_tool_result("\n\n".join(prompt_parts)), list(citations.values())
