from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types import ToolUseBlock
from sqlalchemy.future import select

from app.central_guidance.prompts import TOOL_RESULT_RELEVANCE_CHECK_FAILED
from app.central_guidance.service_rag import ChunkRelevance, search_central_guidance_for_tool
from app.central_guidance.tools import CHUNK_RELEVANCE
from app.config import LLM_CHAT_RESPONSE_MODEL
from app.database.models import (
    AuthSession,
    Chat,
    LlmInternalResponse,
    Message,
    MessageDocumentChunkMapping,
    MessageSearchIndexMapping,
    RewrittenQuery,
    User,
)
from app.database.table import LLMTable


async def create_test_message(db_session, query_content: str) -> int:
    """Helper function to create a complete Message with all required foreign key relationships."""
    user = User()
    db_session.add(user)
    await db_session.flush()

    auth_session = AuthSession(user_id=user.id)
    db_session.add(auth_session)
    await db_session.flush()

    chat = Chat(user_id=user.id, title="Test Chat", from_open_chat=True)
    db_session.add(chat)
    await db_session.flush()

    message = Message(
        chat_id=chat.id,
        content=query_content,
        role="user",
        tokens=0,
        auth_session_id=auth_session.id,
        interrupted=False,
    )
    db_session.add(message)
    await db_session.flush()

    return message.id


async def create_loop_call(db_session) -> int:
    """A logged chat-model call, standing in for the loop call that requested the search."""
    llm = LLMTable().get_by_model(LLM_CHAT_RESPONSE_MODEL)
    row = LlmInternalResponse(llm_id=llm.id, content="tool_use", tokens_in=1, tokens_out=1, completion_cost=0)
    db_session.add(row)
    await db_session.flush()
    return row.id


@pytest.fixture(autouse=True)
def lookups_use_test_session(db_session):
    """The search reads in its own short session; point it at the test's session to see its unflushed rows."""

    @asynccontextmanager
    async def same_session():
        yield db_session

    with patch("app.central_guidance.service_rag.async_db_session", same_session):
        yield


@pytest.fixture
async def fake_chunk_evaluator(db_session):
    """Replace the evaluator's Haiku call with one that marks every chunk relevant.

    The response carries a real llm_internal_response id, as chunk mappings reference it.
    """
    llm_internal_response_id = await create_loop_call(db_session)
    response = MagicMock()
    response.content = [
        ToolUseBlock(
            id="toolu_eval",
            type="tool_use",
            name=CHUNK_RELEVANCE.name,
            input={"evaluations": [{"chunk_index": i, "reasoning": "fake", "is_relevant": True} for i in range(50)]},
        )
    ]
    response.llm_internal_response_id = llm_internal_response_id
    with patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock:
        mock_bedrock.return_value.invoke_async = AsyncMock(return_value=response)
        yield mock_bedrock.return_value.invoke_async


async def _search(
    db_session,
    message_id,
    queries,
    llm_internal_response_id=None,
    user_query="What is MCOM?",
    remove_duplicates=False,
):
    return await search_central_guidance_for_tool(
        queries=queries,
        message_id=message_id,
        user_query=user_query,
        llm_internal_response_id=llm_internal_response_id,
        db_session=db_session,
        remove_duplicates=remove_duplicates,
    )


@pytest.mark.asyncio
async def test_search_with_mcom_query_returns_citations(db_session, sync_central_rag_index, fake_chunk_evaluator):
    message_id = await create_test_message(db_session, "What is MCOM?")

    content, citations, failed = await _search(
        db_session, message_id, ["MCOM", "Modern Communications Operating Model"]
    )

    assert failed is False
    assert len(citations) >= 1
    assert all(isinstance(c["docname"], str) and isinstance(c["docurl"], str) for c in citations)
    assert content.startswith("<government-comms-central-guidance-search-results>")


@pytest.mark.asyncio
async def test_database_tracking_for_analytics(db_session, sync_central_rag_index, fake_chunk_evaluator):
    """The search, the model's queries and every chunk decision are recorded."""
    message_id = await create_test_message(db_session, "What is MCOM?")
    loop_call_id = await create_loop_call(db_session)

    _, citations, _ = await _search(db_session, message_id, ["MCOM", "operating model"], loop_call_id)

    mappings = (
        (await db_session.execute(select(MessageSearchIndexMapping).filter_by(message_id=message_id))).scalars().all()
    )
    assert [(m.use_index, m.llm_internal_response_id) for m in mappings] == [(True, loop_call_id)]

    queries = (await db_session.execute(select(RewrittenQuery).filter_by(message_id=message_id))).scalars().all()
    assert sorted(q.content for q in queries) == ["MCOM", "operating model"]
    assert {q.llm_internal_response_id for q in queries} == {loop_call_id}

    chunk_mappings = (
        (await db_session.execute(select(MessageDocumentChunkMapping).filter_by(message_id=message_id))).scalars().all()
    )
    assert citations
    assert chunk_mappings
    assert all(m.llm_internal_response_id is not None for m in chunk_mappings)


@pytest.mark.asyncio
async def test_chunks_cited_earlier_are_not_repeated_when_removing_duplicates(
    db_session, sync_central_rag_index, fake_chunk_evaluator
):
    message_id = await create_test_message(db_session, "What is MCOM?")

    _, first_citations, _ = await _search(db_session, message_id, ["MCOM"])
    second_content, second_citations, _ = await _search(db_session, message_id, ["MCOM"], remove_duplicates=True)

    assert first_citations
    assert "already provided earlier in the conversation" in second_content
    assert second_citations == []


@pytest.mark.asyncio
async def test_chunks_cited_earlier_are_returned_again_by_default(
    db_session, sync_central_rag_index, fake_chunk_evaluator
):
    message_id = await create_test_message(db_session, "What is MCOM?")

    _, first_citations, _ = await _search(db_session, message_id, ["MCOM"])
    second_content, second_citations, _ = await _search(db_session, message_id, ["MCOM"])

    assert first_citations
    assert second_citations == first_citations
    assert "already provided" not in second_content


@pytest.mark.asyncio
async def test_no_chunks_found(db_session):
    message_id = await create_test_message(db_session, "nonexistent topic")

    with patch("app.central_guidance.service_rag.AsyncOpenSearchOperations.search_for_chunks", return_value=[]):
        content, citations, failed = await _search(db_session, message_id, ["nonexistent topic"])

    assert failed is False
    assert citations == []
    assert "No new relevant passages found" in content


@pytest.mark.asyncio
async def test_all_chunks_filtered_out(db_session, sync_central_rag_index):
    message_id = await create_test_message(db_session, "What is MCOM?")

    async def nothing_relevant(retrieval_results, user_query, llm, recent_context="", context=None):
        return ChunkRelevance(decisions=[False] * len(retrieval_results), llm_internal_response_id=None)

    with patch("app.central_guidance.service_rag.evaluate_chunks_relevance", side_effect=nothing_relevant):
        content, citations, failed = await _search(db_session, message_id, ["MCOM"])

    assert failed is False
    assert citations == []
    assert "No new relevant passages found" in content


@pytest.mark.asyncio
async def test_blank_queries_are_not_searched(db_session):
    message_id = await create_test_message(db_session, "hi")

    with patch("app.central_guidance.service_rag.AsyncOpenSearchOperations.search_for_chunks") as search:
        content, citations, _ = await _search(db_session, message_id, ["  ", ""])

    search.assert_not_called()
    assert citations == []
    assert "No search queries were given" in content


@pytest.mark.asyncio
async def test_evaluator_failure_is_raised_for_the_loop_to_handle(db_session):
    """Failures propagate; the agent loop turns them into a generic is_error tool_result."""
    message_id = await create_test_message(db_session, "What is MCOM?")
    candidate = MagicMock()
    candidate.document_chunk.content = "MCOM content"

    with (
        patch(
            "app.central_guidance.service_rag.AsyncOpenSearchOperations.search_for_chunks",
            return_value=[{"_id": "os-1", "_score": 1.0}],
        ),
        patch("app.central_guidance.service_rag.resolve_chunks", return_value=[candidate]),
        patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock,
    ):
        mock_bedrock.return_value.invoke_async.side_effect = Exception("Bedrock API failure")

        with pytest.raises(Exception, match="Bedrock API failure"):
            await _search(db_session, message_id, ["MCOM"])


@pytest.mark.asyncio
async def test_relevance_check_that_does_not_validate_is_an_error(db_session, sync_central_rag_index):
    """A broken relevance response is reported as a failure, not as "nothing found" (M17)."""
    message_id = await create_test_message(db_session, "What is MCOM?")
    loop_call_id = await create_loop_call(db_session)
    response = MagicMock()
    response.content = []
    response.stop_reason = "max_tokens"
    response.llm_internal_response_id = loop_call_id

    with patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock:
        mock_bedrock.return_value.invoke_async = AsyncMock(return_value=response)
        content, citations, failed = await _search(db_session, message_id, ["MCOM"])

    assert failed is True
    assert citations == []
    assert TOOL_RESULT_RELEVANCE_CHECK_FAILED in content
    chunk_mappings = (
        (await db_session.execute(select(MessageDocumentChunkMapping).filter_by(message_id=message_id))).scalars().all()
    )
    assert chunk_mappings
    assert not any(m.use_document_chunk for m in chunk_mappings)
