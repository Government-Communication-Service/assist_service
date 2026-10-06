from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic.types import ToolUseBlock

from app.central_guidance.constants import MAX_REWRITTEN_QUERIES
from app.central_guidance.service_rag import generate_rewritten_queries, search_and_filter_chunks
from app.central_guidance.utils import sanitise_keyword_queries
from app.document_upload.utils import rewrite_user_query

STRINGIFIED_QUERIES = '["press release guidance", "news release template", "media statement advice"]'
EXPECTED_QUERIES = ["press release guidance", "news release template", "media statement advice"]


class TestSanitiseKeywordQueries:
    """Regression: the query-rewriter tool call sometimes returns keyword_queries as a single
    string rather than a JSON array. Iterating that string ran one OpenSearch search and saved one
    RewrittenQuery row per character (221 for one message in production)."""

    def test_list_is_returned_unchanged(self):
        assert sanitise_keyword_queries(EXPECTED_QUERIES, fallback_query="q") == EXPECTED_QUERIES

    def test_stringified_json_array_is_parsed_into_list(self):
        assert sanitise_keyword_queries(STRINGIFIED_QUERIES, fallback_query="q") == EXPECTED_QUERIES

    def test_plain_string_is_treated_as_a_single_query(self):
        assert sanitise_keyword_queries("press release guidance", fallback_query="q") == ["press release guidance"]

    def test_json_string_that_is_not_an_array_is_treated_as_a_single_query(self):
        assert sanitise_keyword_queries('"press release"', fallback_query="q") == ['"press release"']

    def test_blank_non_string_and_duplicate_items_are_dropped(self):
        value = ["  press release  ", "", "   ", None, 3, "press release", "media statement"]
        assert sanitise_keyword_queries(value, fallback_query="q") == ["press release", "media statement"]

    @pytest.mark.parametrize("value", [None, {}, 42, [], "", "   ", "[]", ["", None]])
    def test_falls_back_to_user_query_when_nothing_usable(self, value):
        assert sanitise_keyword_queries(value, fallback_query="original question") == ["original question"]

    def test_fallback_query_is_stripped(self):
        assert sanitise_keyword_queries(None, fallback_query="  original question  ") == ["original question"]

    @pytest.mark.parametrize("fallback_query", ["", "   ", "\n\t", None])
    def test_blank_fallback_query_returns_no_queries(self, fallback_query):
        assert sanitise_keyword_queries([], fallback_query=fallback_query) == []

    def test_number_of_queries_is_capped(self):
        value = [f"query {i}" for i in range(MAX_REWRITTEN_QUERIES + 5)]
        assert sanitise_keyword_queries(value, fallback_query="q") == value[:MAX_REWRITTEN_QUERIES]


def make_query_rewriter_db_session():
    """db_session whose execute() answers the LLM lookup first, then the RewrittenQuery insert."""
    llm_result = MagicMock()
    llm_result.scalar_one.return_value = MagicMock(max_tokens=512)
    session = MagicMock()
    session.execute = AsyncMock(side_effect=[llm_result, MagicMock()])
    return session


def make_query_rewriter_response(keyword_queries):
    block = MagicMock(spec=ToolUseBlock)
    block.input = {"keyword_queries": keyword_queries}
    response = MagicMock()
    response.content = [block]
    response.llm_internal_response_id = 99
    return response


def saved_rewritten_query_contents(db_session):
    query_models = db_session.execute.call_args_list[1].args[1]
    return [row["content"] for row in query_models]


class TestRewrittenQueriesFromStringifiedToolOutput:
    @pytest.fixture
    def index(self):
        return MagicMock(id=7)

    @pytest.mark.asyncio
    async def test_central_guidance_generate_rewritten_queries(self, index):
        db_session = make_query_rewriter_db_session()
        response = make_query_rewriter_response(STRINGIFIED_QUERIES)

        with patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock:
            mock_bedrock.return_value.invoke_async = AsyncMock(return_value=response)
            result = await generate_rewritten_queries("query", index, message_id=1, db_session=db_session)

        assert result == EXPECTED_QUERIES
        assert saved_rewritten_query_contents(db_session) == EXPECTED_QUERIES

    @pytest.mark.asyncio
    async def test_personal_documents_rewrite_user_query(self, index):
        db_session = make_query_rewriter_db_session()
        response = make_query_rewriter_response(STRINGIFIED_QUERIES)

        with patch("app.document_upload.utils.BedrockHandler") as mock_bedrock:
            mock_bedrock.return_value.invoke_async = AsyncMock(return_value=response)
            result = await rewrite_user_query("query", index, message_id=1, db_session=db_session)

        assert result == EXPECTED_QUERIES
        assert saved_rewritten_query_contents(db_session) == EXPECTED_QUERIES

    @pytest.mark.asyncio
    async def test_missing_tool_block_falls_back_to_user_query(self, index):
        db_session = make_query_rewriter_db_session()
        response = MagicMock(content=[], llm_internal_response_id=99)

        with patch("app.document_upload.utils.BedrockHandler") as mock_bedrock:
            mock_bedrock.return_value.invoke_async = AsyncMock(return_value=response)
            result = await rewrite_user_query("original question", index, message_id=1, db_session=db_session)

        assert result == ["original question"]
        assert saved_rewritten_query_contents(db_session) == ["original question"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "rewrite, module",
        [
            (generate_rewritten_queries, "app.central_guidance.service_rag"),
            (rewrite_user_query, "app.document_upload.utils"),
        ],
    )
    async def test_blank_query_saves_no_rewritten_queries(self, index, rewrite, module):
        db_session = make_query_rewriter_db_session()
        response = make_query_rewriter_response([])

        with patch(f"{module}.BedrockHandler") as mock_bedrock:
            mock_bedrock.return_value.invoke_async = AsyncMock(return_value=response)
            result = await rewrite("   ", index, message_id=1, db_session=db_session)

        assert result == []
        # Only the LLM lookup ran - no RewrittenQuery insert.
        assert db_session.execute.await_count == 1

    @pytest.mark.asyncio
    async def test_blank_query_runs_no_central_guidance_search(self, index):
        db_session = make_query_rewriter_db_session()
        response = make_query_rewriter_response([])

        with (
            patch("app.central_guidance.service_rag.BedrockHandler") as mock_bedrock,
            patch(
                "app.central_guidance.service_rag.AsyncOpenSearchOperations.search_for_chunks",
                new_callable=AsyncMock,
            ) as mock_search,
        ):
            mock_bedrock.return_value.invoke_async = AsyncMock(return_value=response)
            result = await search_and_filter_chunks("   ", index, message_id=1, db_session=db_session)

        assert result == []
        mock_search.assert_not_awaited()
