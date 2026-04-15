from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types import ToolUseBlock

from app.central_guidance.schemas import RetrievalResult
from app.central_guidance.service_rag import evaluate_chunk_relevance


def make_tool_use_block(is_relevant, reasoning="some reasoning"):
    block = MagicMock(spec=ToolUseBlock)
    block.input = {"is_relevant": is_relevant, "reasoning": reasoning}
    return block


def make_llm_response(is_relevant, reasoning="some reasoning"):
    response = MagicMock()
    response.content = [make_tool_use_block(is_relevant, reasoning)]
    response.usage = MagicMock(input_tokens=10, output_tokens=5)
    return response


def make_retrieval_result():
    doc_chunk = MagicMock()
    doc_chunk.name = "Section 1"
    doc_chunk.content = "Some content about comms strategy."

    document = MagicMock()
    document.name = "GCS Guidance"

    mapping = MagicMock()
    mapping.id = 42

    search_index = MagicMock()

    return RetrievalResult(
        search_index=search_index,
        document_chunk=doc_chunk,
        document=document,
        message_document_chunk_mapping=mapping,
    )


@pytest.fixture
def mock_db_session():
    session = MagicMock()
    llm = MagicMock()
    llm.model = "claude-3-5-haiku-20241022-v1:0"
    llm.max_tokens = 512
    execute_result = MagicMock()
    execute_result.scalar_one.return_value = llm
    session.execute = AsyncMock(return_value=execute_result)
    return session


class TestEvaluateChunkRelevance:
    @pytest.mark.asyncio
    async def test_boolean_true_marks_chunk_as_relevant(self, mock_db_session):
        """LLM returns JSON boolean true -> use_chunk is True."""
        retrieval_result = make_retrieval_result()
        llm_response = make_llm_response(is_relevant=True)
        saved_llm_response = MagicMock(id=99)
        updated_mapping = MagicMock()

        with (
            patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock,
            patch("app.central_guidance.service_rag.save_llm_response", new_callable=AsyncMock) as mock_save,
            patch("app.central_guidance.service_rag.update_chunk_mapping", new_callable=AsyncMock) as mock_update,
        ):
            mock_bedrock.return_value.invoke_async = AsyncMock(return_value=llm_response)
            mock_save.return_value = saved_llm_response
            mock_update.return_value = updated_mapping

            result = await evaluate_chunk_relevance(retrieval_result, "what is comms strategy?", mock_db_session)

        mock_update.assert_called_once_with(mock_db_session, 42, 99, True)
        assert result.message_document_chunk_mapping is updated_mapping

    @pytest.mark.asyncio
    async def test_boolean_false_marks_chunk_as_not_relevant(self, mock_db_session):
        """LLM returns JSON boolean false -> use_chunk is False."""
        retrieval_result = make_retrieval_result()
        llm_response = make_llm_response(is_relevant=False)
        saved_llm_response = MagicMock(id=99)
        updated_mapping = MagicMock()

        with (
            patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock,
            patch("app.central_guidance.service_rag.save_llm_response", new_callable=AsyncMock) as mock_save,
            patch("app.central_guidance.service_rag.update_chunk_mapping", new_callable=AsyncMock) as mock_update,
        ):
            mock_bedrock.return_value.invoke_async = AsyncMock(return_value=llm_response)
            mock_save.return_value = saved_llm_response
            mock_update.return_value = updated_mapping

            await evaluate_chunk_relevance(retrieval_result, "what is comms strategy?", mock_db_session)

        mock_update.assert_called_once_with(mock_db_session, 42, 99, False)

    @pytest.mark.asyncio
    async def test_string_true_is_coerced_to_boolean_true(self, mock_db_session):
        """LLM returns string 'True' instead of JSON boolean -> coerced to True."""
        retrieval_result = make_retrieval_result()
        llm_response = make_llm_response(is_relevant="True")
        saved_llm_response = MagicMock(id=99)
        updated_mapping = MagicMock()

        with (
            patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock,
            patch("app.central_guidance.service_rag.save_llm_response", new_callable=AsyncMock) as mock_save,
            patch("app.central_guidance.service_rag.update_chunk_mapping", new_callable=AsyncMock) as mock_update,
        ):
            mock_bedrock.return_value.invoke_async = AsyncMock(return_value=llm_response)
            mock_save.return_value = saved_llm_response
            mock_update.return_value = updated_mapping

            await evaluate_chunk_relevance(retrieval_result, "what is comms strategy?", mock_db_session)

        mock_update.assert_called_once_with(mock_db_session, 42, 99, True)

    @pytest.mark.asyncio
    async def test_string_false_is_coerced_to_boolean_false(self, mock_db_session):
        """LLM returns string 'False' (the original bug) -> coerced to False, not truthy string."""
        retrieval_result = make_retrieval_result()
        llm_response = make_llm_response(is_relevant="False")
        saved_llm_response = MagicMock(id=99)
        updated_mapping = MagicMock()

        with (
            patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock,
            patch("app.central_guidance.service_rag.save_llm_response", new_callable=AsyncMock) as mock_save,
            patch("app.central_guidance.service_rag.update_chunk_mapping", new_callable=AsyncMock) as mock_update,
        ):
            mock_bedrock.return_value.invoke_async = AsyncMock(return_value=llm_response)
            mock_save.return_value = saved_llm_response
            mock_update.return_value = updated_mapping

            await evaluate_chunk_relevance(retrieval_result, "what is comms strategy?", mock_db_session)

        mock_update.assert_called_once_with(mock_db_session, 42, 99, False)

    @pytest.mark.asyncio
    async def test_missing_is_relevant_defaults_to_false(self, mock_db_session):
        """Tool response omits is_relevant entirely -> defaults to False."""
        retrieval_result = make_retrieval_result()

        response = MagicMock()
        block = MagicMock(spec=ToolUseBlock)
        block.input = {"reasoning": "no decision given"}
        response.content = [block]
        response.usage = MagicMock(input_tokens=10, output_tokens=5)

        saved_llm_response = MagicMock(id=99)
        updated_mapping = MagicMock()

        with (
            patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock,
            patch("app.central_guidance.service_rag.save_llm_response", new_callable=AsyncMock) as mock_save,
            patch("app.central_guidance.service_rag.update_chunk_mapping", new_callable=AsyncMock) as mock_update,
        ):
            mock_bedrock.return_value.invoke_async = AsyncMock(return_value=response)
            mock_save.return_value = saved_llm_response
            mock_update.return_value = updated_mapping

            await evaluate_chunk_relevance(retrieval_result, "what is comms strategy?", mock_db_session)

        mock_update.assert_called_once_with(mock_db_session, 42, 99, False)

    @pytest.mark.asyncio
    async def test_no_tool_use_block_defaults_to_false(self, mock_db_session):
        """LLM response contains no ToolUseBlock at all -> use_chunk defaults to False."""
        retrieval_result = make_retrieval_result()

        response = MagicMock()
        response.content = [MagicMock()]  # not a ToolUseBlock instance
        response.usage = MagicMock(input_tokens=10, output_tokens=5)

        saved_llm_response = MagicMock(id=99)
        updated_mapping = MagicMock()

        with (
            patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock,
            patch("app.central_guidance.service_rag.save_llm_response", new_callable=AsyncMock) as mock_save,
            patch("app.central_guidance.service_rag.update_chunk_mapping", new_callable=AsyncMock) as mock_update,
        ):
            mock_bedrock.return_value.invoke_async = AsyncMock(return_value=response)
            mock_save.return_value = saved_llm_response
            mock_update.return_value = updated_mapping

            await evaluate_chunk_relevance(retrieval_result, "what is comms strategy?", mock_db_session)

        mock_update.assert_called_once_with(mock_db_session, 42, 99, False)

    @pytest.mark.asyncio
    async def test_bedrock_exception_propagates(self, mock_db_session):
        """If BedrockHandler raises, evaluate_chunk_relevance re-raises (no silent swallow)."""
        retrieval_result = make_retrieval_result()

        with (
            patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock,
            patch("app.central_guidance.service_rag.save_llm_response", new_callable=AsyncMock),
            patch("app.central_guidance.service_rag.update_chunk_mapping", new_callable=AsyncMock),
        ):
            mock_bedrock.return_value.invoke_async = AsyncMock(side_effect=RuntimeError("Bedrock unavailable"))

            with pytest.raises(RuntimeError, match="Bedrock unavailable"):
                await evaluate_chunk_relevance(retrieval_result, "query", mock_db_session)
