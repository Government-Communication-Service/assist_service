from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types import ToolUseBlock

from app.central_guidance.prompts import TOOL_RESULT_RELEVANCE_CHECK_FAILED
from app.central_guidance.schemas import RetrievalResult
from app.central_guidance.service_rag import (
    ChunkRelevance,
    evaluate_chunks_relevance,
    resolve_chunks,
    search_central_guidance_for_tool,
)
from app.central_guidance.tools import CHUNK_RELEVANCE


def make_tool_use_block(evaluations):
    return ToolUseBlock(id="toolu_1", type="tool_use", name=CHUNK_RELEVANCE.name, input={"evaluations": evaluations})


def make_llm_response(evaluations):
    response = MagicMock()
    response.content = [make_tool_use_block(evaluations)]
    response.usage = MagicMock(input_tokens=10, output_tokens=5)
    response.llm_internal_response_id = 99
    response.stop_reason = "tool_use"
    return response


def make_retrieval_result(score=1.0):
    doc_chunk = MagicMock()
    doc_chunk.name = "Section 1"
    doc_chunk.content = "Some content about comms strategy."

    document = MagicMock()
    document.name = "GCS Guidance"

    mapping = MagicMock()
    mapping.opensearch_score = score

    return RetrievalResult(
        search_index=MagicMock(),
        document_chunk=doc_chunk,
        document=document,
        message_document_chunk_mapping=mapping,
    )


@pytest.fixture
def llm():
    llm = MagicMock()
    llm.model = "claude-3-5-haiku-20241022-v1:0"
    llm.max_tokens = 512
    return llm


async def _evaluate(llm, response, retrieval_results=None):
    with patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock:
        mock_invoke = AsyncMock(return_value=response)
        mock_bedrock.return_value.invoke_async = mock_invoke
        relevance = await evaluate_chunks_relevance(
            retrieval_results or [make_retrieval_result()], "what is comms strategy?", llm
        )
    return relevance, mock_invoke


class TestEvaluateChunksRelevance:
    @pytest.mark.asyncio
    async def test_boolean_true_marks_chunk_as_relevant(self, llm):
        relevance, _ = await _evaluate(
            llm, make_llm_response([{"chunk_index": 0, "is_relevant": True, "reasoning": "relevant"}])
        )
        assert relevance == ChunkRelevance(decisions=[True], llm_internal_response_id=99)

    @pytest.mark.asyncio
    async def test_boolean_false_marks_chunk_as_not_relevant(self, llm):
        relevance, _ = await _evaluate(
            llm, make_llm_response([{"chunk_index": 0, "is_relevant": False, "reasoning": "not relevant"}])
        )
        assert relevance.decisions == [False]

    @pytest.mark.asyncio
    async def test_string_true_is_coerced_to_boolean_true(self, llm):
        relevance, _ = await _evaluate(
            llm, make_llm_response([{"chunk_index": 0, "is_relevant": "True", "reasoning": "relevant"}])
        )
        assert relevance.decisions == [True]

    @pytest.mark.asyncio
    async def test_string_false_is_coerced_to_boolean_false(self, llm):
        """LLM returns string 'False' (the original bug) -> coerced to False, not truthy string."""
        relevance, _ = await _evaluate(
            llm, make_llm_response([{"chunk_index": 0, "is_relevant": "False", "reasoning": "not relevant"}])
        )
        assert relevance.decisions == [False]

    @pytest.mark.asyncio
    async def test_missing_chunk_evaluation_is_unchecked(self, llm):
        """A chunk with no evaluation gets no decision (None), not 'not relevant'."""
        relevance, _ = await _evaluate(
            llm,
            make_llm_response([{"chunk_index": 0, "is_relevant": True, "reasoning": "relevant"}]),
            [make_retrieval_result(), make_retrieval_result()],
        )
        assert relevance.decisions == [True, None]

    @pytest.mark.asyncio
    async def test_no_tool_use_block_leaves_every_chunk_unchecked(self, llm, caplog):
        response = MagicMock()
        response.content = [MagicMock()]  # not a ToolUseBlock instance
        response.llm_internal_response_id = 99
        response.stop_reason = "max_tokens"

        relevance, _ = await _evaluate(llm, response)

        assert relevance.decisions == [None]
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert errors and "max_tokens" in errors[0].getMessage()

    @pytest.mark.asyncio
    async def test_multiple_chunks_evaluated_in_a_single_llm_call(self, llm):
        relevance, mock_invoke = await _evaluate(
            llm,
            make_llm_response(
                [
                    {"chunk_index": 0, "is_relevant": True, "reasoning": "relevant"},
                    {"chunk_index": 1, "is_relevant": False, "reasoning": "not relevant"},
                ]
            ),
            [make_retrieval_result(), make_retrieval_result()],
        )
        mock_invoke.assert_called_once()
        assert relevance.decisions == [True, False]

    @pytest.mark.asyncio
    async def test_string_chunk_index_is_coerced_to_int(self, llm):
        relevance, _ = await _evaluate(
            llm, make_llm_response([{"chunk_index": "0", "is_relevant": True, "reasoning": "relevant"}])
        )
        assert relevance.decisions == [True]

    @pytest.mark.asyncio
    async def test_unparseable_chunk_index_leaves_every_chunk_unchecked(self, llm):
        """The response doesn't validate, so no chunk gets a decision."""
        relevance, _ = await _evaluate(
            llm, make_llm_response([{"chunk_index": "not-a-number", "is_relevant": True, "reasoning": "x"}])
        )
        assert relevance.decisions == [None]

    @pytest.mark.asyncio
    async def test_stringified_evaluations_list_is_parsed(self, llm):
        """Bugsnag regression: evaluations sent as a JSON-encoded string, not an array."""
        relevance, _ = await _evaluate(
            llm, make_llm_response('[{"chunk_index": 0, "is_relevant": true, "reasoning": "relevant"}]')
        )
        assert relevance.decisions == [True]

    @pytest.mark.asyncio
    async def test_request_uses_strict_forced_tool(self, llm):
        _, mock_invoke = await _evaluate(llm, make_llm_response([]))

        kwargs = mock_invoke.call_args.kwargs
        assert kwargs["tools"] == [CHUNK_RELEVANCE.spec]
        assert kwargs["tools"][0]["strict"] is True
        assert kwargs["tool_choice"] == CHUNK_RELEVANCE.forced_choice

    @pytest.mark.asyncio
    async def test_chunk_content_is_truncated_before_sending_to_llm(self, llm):
        """Regression: chunk content must be capped before evaluation, not just at compile time."""
        retrieval_result = make_retrieval_result()
        retrieval_result.document_chunk.content = "x" * 100_000
        response = make_llm_response([{"chunk_index": 0, "is_relevant": True, "reasoning": "relevant"}])

        with patch("app.central_guidance.service_rag.MAX_CENTRAL_GUIDANCE_CHUNK_CHARS", 100):
            _, mock_invoke = await _evaluate(llm, response, [retrieval_result])

        sent_content = mock_invoke.call_args.kwargs["messages"][0]["content"]
        assert "x" * 100 in sent_content
        assert "x" * 101 not in sent_content

    @pytest.mark.asyncio
    async def test_bedrock_exception_propagates(self, llm):
        """If BedrockHandler raises, evaluate_chunks_relevance re-raises (no silent swallow)."""
        with patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock:
            mock_bedrock.return_value.invoke_async = AsyncMock(side_effect=RuntimeError("Bedrock unavailable"))

            with pytest.raises(RuntimeError, match="Bedrock unavailable"):
                await evaluate_chunks_relevance([make_retrieval_result()], "query", llm)


class TestResolveChunks:
    @pytest.mark.asyncio
    async def test_skips_listed_and_unknown_chunks_and_keeps_hit_order(self):
        """No mapping for chunks already known to be discarded, or with no DocumentChunk row."""
        chunk_a = MagicMock(id=1, id_opensearch="os-a")
        chunk_b = MagicMock(id=2, id_opensearch="os-b")
        chunk_c = MagicMock(id=3, id_opensearch="os-c")
        rows = MagicMock()
        rows.all.return_value = [(chunk_c, MagicMock()), (chunk_a, MagicMock()), (chunk_b, MagicMock())]
        db_session = MagicMock()
        db_session.execute = AsyncMock(return_value=rows)

        hits = [
            {"_id": "os-a", "_score": 1.0},
            {"_id": "os-b", "_score": 2.0},
            {"_id": "os-missing", "_score": 3.0},
            {"_id": "os-c", "_score": 4.0},
        ]
        results = await resolve_chunks(db_session, hits, index=MagicMock(), message_id=7, skip_chunk_ids={1})

        assert [r.document_chunk.id for r in results] == [2, 3]
        assert [r.message_document_chunk_mapping.opensearch_score for r in results] == [2.0, 4.0]
        assert all(r.message_document_chunk_mapping.message_id == 7 for r in results)
        db_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_hits_runs_no_query(self):
        db_session = MagicMock()
        db_session.execute = AsyncMock()
        assert await resolve_chunks(db_session, [], index=MagicMock(), message_id=1) == []
        db_session.execute.assert_not_called()


@pytest.fixture
def search_patches():
    """Patch out every external call search_central_guidance_for_tool makes."""
    read_session = MagicMock()
    read_session.execute = AsyncMock(return_value=MagicMock())

    @asynccontextmanager
    async def fake_session():
        yield read_session

    with (
        patch("app.central_guidance.service_rag.async_db_session", fake_session),
        patch("app.central_guidance.service_rag.get_central_guidance_index", new_callable=AsyncMock),
        patch(
            "app.central_guidance.service_rag.AsyncOpenSearchOperations.search_for_chunks", new_callable=AsyncMock
        ) as search,
        patch("app.central_guidance.service_rag.resolve_chunks", new_callable=AsyncMock) as resolve,
        patch("app.central_guidance.service_rag.get_previously_cited_chunk_ids", new_callable=AsyncMock) as cited,
        patch("app.central_guidance.service_rag.evaluate_chunks_relevance", new_callable=AsyncMock) as evaluate,
    ):
        search.return_value = [{"_id": "chunk-x", "_score": 1.0}]
        cited.return_value = set()
        resolve.return_value = []
        yield MagicMock(search=search, resolve=resolve, evaluate=evaluate)


async def _search(queries=("query",)):
    write_session = MagicMock()
    write_session.add = MagicMock()
    write_session.add_all = MagicMock()
    write_session.execute = AsyncMock()
    result = await search_central_guidance_for_tool(
        queries=list(queries),
        message_id=1,
        user_query="query",
        llm_internal_response_id=None,
        db_session=write_session,
    )
    return result, write_session


class TestSearchCentralGuidanceForTool:
    @pytest.mark.asyncio
    async def test_dedup_keeps_highest_score_across_queries(self, search_patches):
        """A chunk retrieved by several of the model's queries keeps its highest score."""
        search_patches.search.side_effect = [
            [{"_id": "chunk-x", "_score": 0.4}],
            [{"_id": "chunk-x", "_score": 0.9}],
        ]

        await _search(["query one", "query two", "query one"])

        assert search_patches.search.await_count == 2
        passed_hits = search_patches.resolve.call_args.args[1]
        assert passed_hits == [{"_id": "chunk-x", "_score": 0.9}]

    @pytest.mark.asyncio
    async def test_write_session_is_not_used_until_the_relevance_check_is_done(self, search_patches):
        """H1 step 2: no connection is held across the OpenSearch and Haiku calls."""
        search_patches.resolve.return_value = [make_retrieval_result()]
        calls = []
        write_session = MagicMock()
        write_session.add = MagicMock(side_effect=lambda *_: calls.append("write"))
        write_session.add_all = MagicMock(side_effect=lambda *_: calls.append("write"))
        write_session.execute = AsyncMock(side_effect=lambda *_: calls.append("write"))

        async def evaluate(*args, **kwargs):
            calls.append("evaluate")
            return ChunkRelevance(decisions=[True], llm_internal_response_id=5)

        search_patches.evaluate.side_effect = evaluate
        search_patches.search.side_effect = lambda *_: calls.append("search") or []

        await search_central_guidance_for_tool(
            queries=["q"], message_id=1, user_query="q", llm_internal_response_id=None, db_session=write_session
        )

        assert calls[0] == "search"
        assert calls.index("evaluate") < calls.index("write")

    @pytest.mark.asyncio
    async def test_relevance_check_failure_is_an_error_not_no_results(self, search_patches):
        search_patches.resolve.return_value = [make_retrieval_result(), make_retrieval_result()]
        search_patches.evaluate.return_value = ChunkRelevance(decisions=[None, None], llm_internal_response_id=5)

        (content, citations, failed), write_session = await _search()

        assert failed is True
        assert citations == []
        assert TOOL_RESULT_RELEVANCE_CHECK_FAILED in content
        assert "No new relevant passages" not in content
        # The search and its chunk decisions are still recorded.
        write_session.add_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_partial_relevance_failure_returns_checked_passages_with_a_note(self, search_patches):
        relevant = make_retrieval_result()
        relevant.document.uuid = "doc-1"
        relevant.document.url = "https://example.com"
        search_patches.resolve.return_value = [relevant, make_retrieval_result()]
        search_patches.evaluate.return_value = ChunkRelevance(decisions=[True, None], llm_internal_response_id=5)

        (content, citations, failed), _ = await _search()

        assert failed is False
        assert citations == [{"docname": "GCS Guidance", "docurl": "https://example.com"}]
        assert "1 result(s) could not be checked for relevance and were left out." in content
        assert relevant.message_document_chunk_mapping.use_document_chunk is True
        assert relevant.message_document_chunk_mapping.llm_internal_response_id == 5

    @pytest.mark.asyncio
    async def test_nothing_relevant_is_not_an_error(self, search_patches):
        search_patches.resolve.return_value = [make_retrieval_result()]
        search_patches.evaluate.return_value = ChunkRelevance(decisions=[False], llm_internal_response_id=5)

        (content, citations, failed), _ = await _search()

        assert failed is False
        assert citations == []
        assert "No new relevant passages found" in content
