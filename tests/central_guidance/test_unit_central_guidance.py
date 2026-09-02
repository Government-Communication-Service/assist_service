from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types import ToolUseBlock

from app.central_guidance.schemas import RetrievalResult
from app.central_guidance.service_rag import create_chunk_mappings, evaluate_chunks_relevance, search_and_filter_chunks


def make_tool_use_block(evaluations):
    block = MagicMock(spec=ToolUseBlock)
    block.input = {"evaluations": evaluations}
    return block


def make_llm_response(evaluations):
    response = MagicMock()
    response.content = [make_tool_use_block(evaluations)]
    response.usage = MagicMock(input_tokens=10, output_tokens=5)
    response.llm_internal_response_id = 99
    return response


def make_retrieval_result(mapping_id=42):
    doc_chunk = MagicMock()
    doc_chunk.name = "Section 1"
    doc_chunk.content = "Some content about comms strategy."

    document = MagicMock()
    document.name = "GCS Guidance"

    mapping = MagicMock()
    mapping.id = mapping_id

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


class TestEvaluateChunksRelevance:
    @pytest.mark.asyncio
    async def test_boolean_true_marks_chunk_as_relevant(self, mock_db_session):
        """LLM returns JSON boolean true -> use_chunk is True."""
        retrieval_result = make_retrieval_result()
        llm_response = make_llm_response([{"chunk_index": 0, "is_relevant": True, "reasoning": "relevant"}])
        updated_mapping = MagicMock()

        with (
            patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock,
            patch("app.central_guidance.service_rag.update_chunk_mapping", new_callable=AsyncMock) as mock_update,
        ):
            mock_bedrock.return_value.invoke_async = AsyncMock(return_value=llm_response)
            mock_update.return_value = updated_mapping

            results = await evaluate_chunks_relevance([retrieval_result], "what is comms strategy?", mock_db_session)

        mock_update.assert_called_once_with(mock_db_session, 42, 99, True)
        assert results[0].message_document_chunk_mapping is updated_mapping

    @pytest.mark.asyncio
    async def test_boolean_false_marks_chunk_as_not_relevant(self, mock_db_session):
        """LLM returns JSON boolean false -> use_chunk is False."""
        retrieval_result = make_retrieval_result()
        llm_response = make_llm_response([{"chunk_index": 0, "is_relevant": False, "reasoning": "not relevant"}])
        updated_mapping = MagicMock()

        with (
            patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock,
            patch("app.central_guidance.service_rag.update_chunk_mapping", new_callable=AsyncMock) as mock_update,
        ):
            mock_bedrock.return_value.invoke_async = AsyncMock(return_value=llm_response)
            mock_update.return_value = updated_mapping

            await evaluate_chunks_relevance([retrieval_result], "what is comms strategy?", mock_db_session)

        mock_update.assert_called_once_with(mock_db_session, 42, 99, False)

    @pytest.mark.asyncio
    async def test_string_true_is_coerced_to_boolean_true(self, mock_db_session):
        """LLM returns string 'True' instead of JSON boolean -> coerced to True."""
        retrieval_result = make_retrieval_result()
        llm_response = make_llm_response([{"chunk_index": 0, "is_relevant": "True", "reasoning": "relevant"}])
        updated_mapping = MagicMock()

        with (
            patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock,
            patch("app.central_guidance.service_rag.update_chunk_mapping", new_callable=AsyncMock) as mock_update,
        ):
            mock_bedrock.return_value.invoke_async = AsyncMock(return_value=llm_response)
            mock_update.return_value = updated_mapping

            await evaluate_chunks_relevance([retrieval_result], "what is comms strategy?", mock_db_session)

        mock_update.assert_called_once_with(mock_db_session, 42, 99, True)

    @pytest.mark.asyncio
    async def test_string_false_is_coerced_to_boolean_false(self, mock_db_session):
        """LLM returns string 'False' (the original bug) -> coerced to False, not truthy string."""
        retrieval_result = make_retrieval_result()
        llm_response = make_llm_response([{"chunk_index": 0, "is_relevant": "False", "reasoning": "not relevant"}])
        updated_mapping = MagicMock()

        with (
            patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock,
            patch("app.central_guidance.service_rag.update_chunk_mapping", new_callable=AsyncMock) as mock_update,
        ):
            mock_bedrock.return_value.invoke_async = AsyncMock(return_value=llm_response)
            mock_update.return_value = updated_mapping

            await evaluate_chunks_relevance([retrieval_result], "what is comms strategy?", mock_db_session)

        mock_update.assert_called_once_with(mock_db_session, 42, 99, False)

    @pytest.mark.asyncio
    async def test_missing_chunk_evaluation_defaults_to_false(self, mock_db_session):
        """Tool response omits an evaluation for a chunk entirely -> defaults to False."""
        retrieval_result = make_retrieval_result()
        llm_response = make_llm_response([])
        updated_mapping = MagicMock()

        with (
            patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock,
            patch("app.central_guidance.service_rag.update_chunk_mapping", new_callable=AsyncMock) as mock_update,
        ):
            mock_bedrock.return_value.invoke_async = AsyncMock(return_value=llm_response)
            mock_update.return_value = updated_mapping

            await evaluate_chunks_relevance([retrieval_result], "what is comms strategy?", mock_db_session)

        mock_update.assert_called_once_with(mock_db_session, 42, 99, False)

    @pytest.mark.asyncio
    async def test_no_tool_use_block_defaults_to_false(self, mock_db_session):
        """LLM response contains no ToolUseBlock at all -> use_chunk defaults to False."""
        retrieval_result = make_retrieval_result()

        response = MagicMock()
        response.content = [MagicMock()]  # not a ToolUseBlock instance
        response.usage = MagicMock(input_tokens=10, output_tokens=5)
        response.llm_internal_response_id = 99
        updated_mapping = MagicMock()

        with (
            patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock,
            patch("app.central_guidance.service_rag.update_chunk_mapping", new_callable=AsyncMock) as mock_update,
        ):
            mock_bedrock.return_value.invoke_async = AsyncMock(return_value=response)
            mock_update.return_value = updated_mapping

            await evaluate_chunks_relevance([retrieval_result], "what is comms strategy?", mock_db_session)

        mock_update.assert_called_once_with(mock_db_session, 42, 99, False)

    @pytest.mark.asyncio
    async def test_multiple_chunks_evaluated_in_a_single_llm_call(self, mock_db_session):
        """Several candidate chunks are evaluated with exactly one LLM invocation."""
        retrieval_results = [make_retrieval_result(mapping_id=1), make_retrieval_result(mapping_id=2)]
        llm_response = make_llm_response(
            [
                {"chunk_index": 0, "is_relevant": True, "reasoning": "relevant"},
                {"chunk_index": 1, "is_relevant": False, "reasoning": "not relevant"},
            ]
        )

        with (
            patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock,
            patch("app.central_guidance.service_rag.update_chunk_mapping", new_callable=AsyncMock) as mock_update,
        ):
            mock_invoke = AsyncMock(return_value=llm_response)
            mock_bedrock.return_value.invoke_async = mock_invoke
            mock_update.side_effect = [MagicMock(), MagicMock()]

            await evaluate_chunks_relevance(retrieval_results, "what is comms strategy?", mock_db_session)

        mock_invoke.assert_called_once()
        assert mock_update.call_count == 2
        mock_update.assert_any_call(mock_db_session, 1, 99, True)
        mock_update.assert_any_call(mock_db_session, 2, 99, False)

    @pytest.mark.asyncio
    async def test_string_chunk_index_is_coerced_to_int(self, mock_db_session):
        """LLM returns chunk_index as a string ('0') instead of an int -> still matched."""
        retrieval_result = make_retrieval_result()
        llm_response = make_llm_response([{"chunk_index": "0", "is_relevant": True, "reasoning": "relevant"}])
        updated_mapping = MagicMock()

        with (
            patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock,
            patch("app.central_guidance.service_rag.update_chunk_mapping", new_callable=AsyncMock) as mock_update,
        ):
            mock_bedrock.return_value.invoke_async = AsyncMock(return_value=llm_response)
            mock_update.return_value = updated_mapping

            await evaluate_chunks_relevance([retrieval_result], "what is comms strategy?", mock_db_session)

        mock_update.assert_called_once_with(mock_db_session, 42, 99, True)

    @pytest.mark.asyncio
    async def test_unparseable_chunk_index_is_ignored_not_fatal(self, mock_db_session):
        """LLM returns a chunk_index that can't be coerced to int -> logged, doesn't crash."""
        retrieval_result = make_retrieval_result()
        llm_response = make_llm_response([{"chunk_index": "not-a-number", "is_relevant": True, "reasoning": "x"}])
        updated_mapping = MagicMock()

        with (
            patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock,
            patch("app.central_guidance.service_rag.update_chunk_mapping", new_callable=AsyncMock) as mock_update,
        ):
            mock_bedrock.return_value.invoke_async = AsyncMock(return_value=llm_response)
            mock_update.return_value = updated_mapping

            await evaluate_chunks_relevance([retrieval_result], "what is comms strategy?", mock_db_session)

        # The chunk gets no matching evaluation, so it defaults to not relevant.
        mock_update.assert_called_once_with(mock_db_session, 42, 99, False)

    @pytest.mark.asyncio
    async def test_chunk_content_is_truncated_before_sending_to_llm(self, mock_db_session):
        """Regression: chunk content must be capped before evaluation, not just at compile time."""
        retrieval_result = make_retrieval_result()
        retrieval_result.document_chunk.content = "x" * 100_000
        llm_response = make_llm_response([{"chunk_index": 0, "is_relevant": True, "reasoning": "relevant"}])

        with (
            patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock,
            patch("app.central_guidance.service_rag.update_chunk_mapping", new_callable=AsyncMock),
            patch("app.central_guidance.service_rag.MAX_CENTRAL_GUIDANCE_CHUNK_CHARS", 100),
        ):
            mock_invoke = AsyncMock(return_value=llm_response)
            mock_bedrock.return_value.invoke_async = mock_invoke

            await evaluate_chunks_relevance([retrieval_result], "query", mock_db_session)

        sent_content = mock_invoke.call_args.kwargs["messages"][0]["content"]
        assert "x" * 100 in sent_content
        assert "x" * 101 not in sent_content

    @pytest.mark.asyncio
    async def test_bedrock_exception_propagates(self, mock_db_session):
        """If BedrockHandler raises, evaluate_chunks_relevance re-raises (no silent swallow)."""
        retrieval_result = make_retrieval_result()

        with (
            patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock,
            patch("app.central_guidance.service_rag.update_chunk_mapping", new_callable=AsyncMock),
        ):
            mock_bedrock.return_value.invoke_async = AsyncMock(side_effect=RuntimeError("Bedrock unavailable"))

            with pytest.raises(RuntimeError, match="Bedrock unavailable"):
                await evaluate_chunks_relevance([retrieval_result], "query", mock_db_session)


class TestCreateChunkMappings:
    @pytest.mark.asyncio
    async def test_skip_chunk_ids_excludes_matching_chunks_without_creating_mapping(self):
        """Regression: no insert (and later update) for chunks already known to be discarded."""
        doc_chunk_a = MagicMock(id=1, document_id=100)
        doc_chunk_b = MagicMock(id=2, document_id=200)
        document_b = MagicMock()

        result_a = MagicMock()
        result_a.scalar_one_or_none.return_value = doc_chunk_a
        result_b = MagicMock()
        result_b.scalar_one_or_none.return_value = doc_chunk_b
        result_document_b = MagicMock()
        result_document_b.scalar_one.return_value = document_b

        db_session = MagicMock()
        db_session.execute = AsyncMock(side_effect=[result_a, result_b, result_document_b])
        db_session.add = MagicMock()

        hits = [{"_id": "os-a", "_score": 1.0}, {"_id": "os-b", "_score": 2.0}]

        results = await create_chunk_mappings(
            hits, index=MagicMock(), message_id=1, db_session=db_session, skip_chunk_ids={1}
        )

        assert [r.document_chunk.id for r in results] == [2]
        db_session.add.assert_called_once()


class TestSearchAndFilterChunksDedup:
    @pytest.mark.asyncio
    async def test_dedup_keeps_highest_score_across_rewritten_queries(self):
        """Regression: a chunk retrieved by multiple rewritten queries keeps its highest score."""
        with (
            patch("app.central_guidance.service_rag.generate_rewritten_queries", new_callable=AsyncMock) as mock_gen,
            patch(
                "app.central_guidance.service_rag.AsyncOpenSearchOperations.search_for_chunks",
                new_callable=AsyncMock,
            ) as mock_search,
            patch("app.central_guidance.service_rag.create_chunk_mappings", new_callable=AsyncMock) as mock_create,
            patch(
                "app.central_guidance.service_rag.get_previously_cited_chunk_ids", new_callable=AsyncMock
            ) as mock_cited,
        ):
            mock_gen.return_value = ["query one", "query two"]
            mock_search.side_effect = [
                [{"_id": "chunk-x", "_score": 0.4}],
                [{"_id": "chunk-x", "_score": 0.9}],
            ]
            mock_cited.return_value = set()
            mock_create.return_value = []

            await search_and_filter_chunks("query", MagicMock(), message_id=1, db_session=MagicMock())

        passed_hits = mock_create.call_args.args[0]
        assert passed_hits == [{"_id": "chunk-x", "_score": 0.9}]
