from logging import getLogger
from typing import List, Optional

from anthropic.types import ToolUseBlock
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.bedrock import BedrockHandler, RunMode
from app.bedrock.service import calculate_completion_cost
from app.central_guidance.constants import (
    SYSTEM_PROMPT_OPENSEARCH_QUERY_GENERATOR,
    TOOL_NAME_OPENSEARCH_QUERY_GENERATOR,
    TOOL_OPENSEARCH_QUERY_GENERATOR,
)
from app.central_guidance.schemas import RetrievalResult
from app.config import LLM_OPENSEARCH_QUERY_GENERATOR
from app.database.models import (
    LLM,
    Document,
    DocumentChunk,
    LlmInternalResponse,
    MessageDocumentChunkMapping,
    RewrittenQuery,
    SearchIndex,
)

logger = getLogger(__name__)


async def rewrite_user_query(
    user_message: str,
    index: SearchIndex,
    message_id: int,
    db_session: AsyncSession,
) -> List[str]:
    logger.info("Generating OpenSearch queries...")
    execute = await db_session.execute(select(LLM).filter(LLM.model == LLM_OPENSEARCH_QUERY_GENERATOR))
    llm = execute.scalar_one()
    bedrock_handler = BedrockHandler(llm, mode=RunMode.ASYNC)
    response = await bedrock_handler.invoke_async(
        max_tokens=llm.max_tokens,
        model=f"us.{llm.model}",
        system=SYSTEM_PROMPT_OPENSEARCH_QUERY_GENERATOR,
        messages=[{"role": "user", "content": user_message}],
        tools=[TOOL_OPENSEARCH_QUERY_GENERATOR],
        tool_choice={"type": "tool", "name": TOOL_NAME_OPENSEARCH_QUERY_GENERATOR},
    )
    logger.info(f"Raw LLM response when rewriting queries: {response}")
    for block in response.content:
        if isinstance(block, ToolUseBlock):
            opensearch_queries = block.input["keyword_queries"]
    logger.info(f"OpenSearch keyword queries generated by LLM: {opensearch_queries}")

    # Record the llm transaction
    stmt = (
        insert(LlmInternalResponse)
        .values(
            llm_id=llm.id,
            content=str(response.content),
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens,
            completion_cost=calculate_completion_cost(llm, response.usage.input_tokens, response.usage.output_tokens),
        )
        .returning(LlmInternalResponse)  # Return the  inserted row
    )
    result = await db_session.execute(stmt)
    inserted_response = result.scalar_one()

    # Record the rewritten queries
    query_models = [
        {
            "search_index_id": index.id,
            "message_id": message_id,
            "llm_internal_response_id": inserted_response.id,
            "content": rewritten_query,
        }
        for rewritten_query in opensearch_queries
    ]
    await db_session.execute(insert(RewrittenQuery), query_models)

    return opensearch_queries


async def get_document_chunk_mappings(
    chunks: List[dict], index: SearchIndex, message_id: int, session: AsyncSession, use_chunk: Optional[bool] = None
) -> List[RetrievalResult]:
    """
    Processes a list of document chunks, creates mappings between messages and document chunks,
    and returns a list of RetrievalResult objects.
    """
    retrieval_results = []
    for hit in chunks:
        id_opensearch = hit["_id"]
        logger.debug("get_document_chunk_mappings-id_opensearch %s", id_opensearch)

        execute = await session.execute(select(DocumentChunk).filter(DocumentChunk.id_opensearch == id_opensearch))
        doc_chunk = execute.scalar_one_or_none()
        if not doc_chunk:
            logger.warning(f"DocumentChunk not found for id_opensearch: {id_opensearch}")
            continue

        message_document_chunk_mapping = MessageDocumentChunkMapping(
            message_id=message_id, document_chunk_id=doc_chunk.id, opensearch_score=hit["_score"]
        )
        if use_chunk is not None:
            message_document_chunk_mapping.use_document_chunk = use_chunk

        session.add(message_document_chunk_mapping)

        execute = await session.execute(select(Document).filter(Document.id == doc_chunk.document_id))
        document = execute.scalar_one()

        retrieval_result = RetrievalResult(
            search_index=index,
            document_chunk=doc_chunk,
            document=document,
            message_document_chunk_mapping=message_document_chunk_mapping,
        )

        retrieval_results.append(retrieval_result)

    return retrieval_results
