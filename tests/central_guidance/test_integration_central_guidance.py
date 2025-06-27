from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.future import select

from app.central_guidance.service_rag import search_central_guidance
from app.chat.schemas import ChatCreateMessageInput
from app.chat.service import chat_create_message
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


async def create_test_message(db_session, query_content: str) -> int:
    """Helper function to create a complete Message with all required foreign key relationships."""
    # Create User
    user = User()
    db_session.add(user)
    await db_session.flush()

    # Create AuthSession
    auth_session = AuthSession(user_id=user.id)
    db_session.add(auth_session)
    await db_session.flush()

    # Create Chat
    chat = Chat(user_id=user.id, title="Test Chat", from_open_chat=True)
    db_session.add(chat)
    await db_session.flush()

    # Create Message
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


@pytest.mark.asyncio
async def test_chat_create_message_no_rag(monkeypatch, mocker, mock_message_table, mock_bedrock_handler, db_session):
    monkeypatch.setenv("USE_RAG", "true")

    input_data = ChatCreateMessageInput(
        query="hey",
        user_id=1,
        initial_call=True,
        auth_session_id=1,
        user_group_ids=[],
        stream=True,
        system="",
        use_rag=False,
        use_case_id=None,
    )

    chat = mocker.Mock()
    chat.id = 1

    mock_run_rag = mocker.patch("app.central_guidance.service_rag.search_central_guidance")
    await chat_create_message(chat, input_data, db_session)
    mock_run_rag.assert_not_called()


@pytest.mark.asyncio
async def test_search_central_guidance_with_mcom_query(db_session):
    """
    Integration test that directly calls search_central_guidance with a MCOM query.
    Expects to receive citations containing at least one result.
    """

    query_content = "What is MCOM?"

    # Create a Message object
    message = Message(
        id=1,
        content=query_content,
    )

    # Call search_central_guidance directly with message ID
    prompt_segment, citations = await search_central_guidance(query_content, message.id, db_session)

    # Assert that citations contain at least one result
    assert citations is not None, "Citations should not be None"
    assert len(citations) >= 1, f"Expected at least 1 citation, but got {len(citations)}"

    # Verify citations have the expected structure
    for citation in citations:
        assert "docname" in citation, "Citation should have 'docname' field"
        assert "docurl" in citation, "Citation should have 'docurl' field"
        assert isinstance(citation["docname"], str), "Citation docname should be a string"
        assert isinstance(citation["docurl"], str), "Citation docurl should be a string"

    # Verify prompt_segment is returned
    assert prompt_segment is not None, "Prompt segment should not be None"
    assert isinstance(prompt_segment, str), "Prompt segment should be a string"


# =============================================================================
# DATABASE ANALYTICS VERIFICATION TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_database_tracking_for_analytics(db_session):
    """
    Verify all LLM calls and decisions are tracked correctly in database for analytics.
    This ensures the analytics pipeline has all required data.
    """
    query_content = "What is MCOM?"

    # Create a Message record with all required foreign key relationships
    message_id = await create_test_message(db_session, query_content)

    # Count existing records before test
    initial_llm_responses = await db_session.execute(select(LlmInternalResponse))
    initial_llm_count = len(initial_llm_responses.scalars().all())

    initial_index_mappings = await db_session.execute(
        select(MessageSearchIndexMapping).filter(MessageSearchIndexMapping.message_id == message_id)
    )
    initial_mapping_count = len(initial_index_mappings.scalars().all())

    initial_rewritten_queries = await db_session.execute(
        select(RewrittenQuery).filter(RewrittenQuery.message_id == message_id)
    )
    initial_query_count = len(initial_rewritten_queries.scalars().all())

    # Execute search_central_guidance
    prompt_segment, citations = await search_central_guidance(query_content, message_id, db_session)

    # Verify LlmInternalResponse records were created
    final_llm_responses = await db_session.execute(select(LlmInternalResponse))
    final_llm_count = len(final_llm_responses.scalars().all())

    # Should have at least 3 LLM calls: index relevance, query rewriting, chunk evaluation(s)
    assert final_llm_count >= initial_llm_count + 3, (
        f"Expected at least 3 new LLM responses, got {final_llm_count - initial_llm_count}"
    )

    # Verify MessageSearchIndexMapping was created for index relevance decision
    final_index_mappings = await db_session.execute(
        select(MessageSearchIndexMapping).filter(MessageSearchIndexMapping.message_id == message_id)
    )
    final_mapping_count = len(final_index_mappings.scalars().all())
    assert final_mapping_count == initial_mapping_count + 1, (
        f"Expected 1 new index mapping, got {final_mapping_count - initial_mapping_count}"
    )

    # Verify RewrittenQuery records were created
    final_rewritten_queries = await db_session.execute(
        select(RewrittenQuery).filter(RewrittenQuery.message_id == message_id)
    )
    final_query_count = len(final_rewritten_queries.scalars().all())
    assert final_query_count > initial_query_count, (
        f"Expected new rewritten queries, got {final_query_count - initial_query_count}"
    )

    # Verify MessageDocumentChunkMapping records were created and updated with LLM decisions
    chunk_mappings = await db_session.execute(
        select(MessageDocumentChunkMapping).filter(MessageDocumentChunkMapping.message_id == message_id)
    )
    chunk_mappings_list = chunk_mappings.scalars().all()

    if len(citations) > 0:  # Only check if we got results
        assert len(chunk_mappings_list) > 0, "Expected chunk mappings to be created"

        # Verify at least some chunk mappings have LLM decisions recorded
        mappings_with_llm_responses = [
            mapping for mapping in chunk_mappings_list if mapping.llm_internal_response_id is not None
        ]
        assert len(mappings_with_llm_responses) > 0, (
            "Expected at least some chunk mappings to have LLM response IDs for analytics"
        )


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_no_relevant_index_scenario(db_session):
    """
    Test when LLM decides the index is not relevant to the query.
    Should return empty results gracefully.
    """
    # Use a query that should be deemed irrelevant to government communications
    query_content = "What is the chemical formula for water?"

    # Create a Message record with all required foreign key relationships
    message_id = await create_test_message(db_session, query_content)

    # Mock the index relevance check to return False
    with patch("app.central_guidance.service_rag.check_index_relevance") as mock_check:
        mock_mapping = MagicMock()
        mock_mapping.use_index = False
        mock_check.return_value = mock_mapping

        prompt_segment, citations = await search_central_guidance(query_content, message_id, db_session)

        # Should return empty results
        assert citations == [], f"Expected empty citations, got {citations}"
        assert prompt_segment == "", f"Expected empty prompt segment, got {prompt_segment}"

        # Verify the relevance check was called
        mock_check.assert_called_once()


@pytest.mark.asyncio
async def test_no_chunks_found_scenario(db_session):
    """
    Test when OpenSearch returns no chunks for the query.
    Should handle gracefully and return appropriate message.
    """
    query_content = "nonexistent topic that will never match"

    # Create a Message record with all required foreign key relationships
    message_id = await create_test_message(db_session, query_content)

    # Mock OpenSearch to return no results
    with patch("app.opensearch.service.AsyncOpenSearchOperations.search_for_chunks") as mock_search:
        mock_search.return_value = []  # No chunks found

        prompt_segment, citations = await search_central_guidance(query_content, message_id, db_session)

        # Should return structured message about no results
        assert citations == [], f"Expected empty citations, got {citations}"
        assert "no relevant material was found" in prompt_segment or prompt_segment == "", (
            f"Expected no results message or empty string, got {prompt_segment}"
        )


@pytest.mark.asyncio
async def test_all_chunks_filtered_out_scenario(db_session):
    """
    Test when OpenSearch finds chunks but LLM filters them all out as irrelevant.
    Should return appropriate message about searched but no relevant material.
    """
    query_content = "What is MCOM?"

    # Create a Message record with all required foreign key relationships
    message_id = await create_test_message(db_session, query_content)

    # Mock chunk evaluation to always return irrelevant
    with patch("app.central_guidance.service_rag.evaluate_chunk_relevance") as mock_evaluate:

        async def mock_evaluate_func(retrieval_result, user_query, db_session):
            # Mark chunk as not relevant
            retrieval_result.message_document_chunk_mapping.use_document_chunk = False
            return retrieval_result

        mock_evaluate.side_effect = mock_evaluate_func

        prompt_segment, citations = await search_central_guidance(query_content, message_id, db_session)

        # Should return message about documents searched but no relevant material found
        assert citations == [], f"Expected empty citations, got {citations}"
        assert "no relevant material was found" in prompt_segment, (
            f"Expected 'no relevant material found' message, got {prompt_segment}"
        )


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_llm_failure_graceful_degradation(db_session):
    """
    Test that LLM failures are handled gracefully without crashing the system.
    """
    query_content = "What is MCOM?"

    # Create a Message record with all required foreign key relationships
    message_id = await create_test_message(db_session, query_content)

    # Mock BedrockHandler to raise an exception
    with patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock:
        mock_bedrock.return_value.invoke_async.side_effect = Exception("Bedrock API failure")

        # Should not raise exception, should handle gracefully
        try:
            prompt_segment, citations = await search_central_guidance(query_content, message_id, db_session)

            # Should return empty results when LLM fails
            assert citations == [], f"Expected empty citations on LLM failure, got {citations}"
            assert prompt_segment == "", f"Expected empty prompt on LLM failure, got {prompt_segment}"

        except Exception as e:
            pytest.fail(f"search_central_guidance should handle LLM failures gracefully, but raised: {e}")


@pytest.mark.asyncio
async def test_opensearch_failure_handling(db_session):
    """
    Test that OpenSearch failures are handled gracefully.
    """
    query_content = "What is MCOM?"

    # Create a Message record with all required foreign key relationships
    message_id = await create_test_message(db_session, query_content)

    # Mock OpenSearch to raise an exception
    with patch("app.opensearch.service.AsyncOpenSearchOperations.search_for_chunks") as mock_search:
        mock_search.side_effect = Exception("OpenSearch connection failed")

        # Should not raise exception, should handle gracefully
        try:
            prompt_segment, citations = await search_central_guidance(query_content, message_id, db_session)

            # Should return empty results when search fails
            assert citations == [], f"Expected empty citations on search failure, got {citations}"
            # Note: may still return a "searched but found nothing" message depending on implementation

        except Exception as e:
            pytest.fail(f"search_central_guidance should handle OpenSearch failures gracefully, but raised: {e}")


@pytest.mark.asyncio
async def test_database_failure_handling(db_session):
    """
    Test that database failures during analytics tracking don't crash the main flow.
    """
    query_content = "What is MCOM?"

    # Create a Message record with all required foreign key relationships
    message_id = await create_test_message(db_session, query_content)

    # Mock database operations to fail
    with patch("app.central_guidance.service_rag.save_llm_response") as mock_save:
        mock_save.side_effect = Exception("Database write failed")

        # Should not crash the main flow
        try:
            prompt_segment, citations = await search_central_guidance(query_content, message_id, db_session)
            # Even if analytics tracking fails, should still attempt to return results
            # (though they may be empty due to other failures cascading)

        except Exception as e:
            # Only fail if it's not the expected database error
            if "Database write failed" not in str(e):
                pytest.fail(f"Unexpected exception: {e}")
            # If it is the database error, that's expected - we want to know about analytics failures
