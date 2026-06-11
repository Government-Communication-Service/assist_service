from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.gov_uk_search.service import (
    assess_document_relevancy,
    assess_if_next_message_should_use_gov_uk_search,
    get_search_queries,
)

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(msg_id=1, role="user", content="hello", summary=None, content_enhanced_with_rag=None):
    msg = MagicMock()
    msg.id = msg_id
    msg.role = role
    msg.content = content
    msg.summary = summary
    msg.content_enhanced_with_rag = content_enhanced_with_rag
    return msg


def _make_tool_use_response(tool_name, input_data, tokens_in=10, tokens_out=5):
    block = MagicMock()
    block.type = "tool_use"
    block.input = input_data
    response = MagicMock()
    response.content = [block]
    response.usage = MagicMock()
    response.usage.input_tokens = tokens_in
    response.usage.output_tokens = tokens_out
    return response


def _make_text_response(text="I cannot use a tool", tokens_in=10, tokens_out=5):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.usage = MagicMock()
    response.usage.input_tokens = tokens_in
    response.usage.output_tokens = tokens_out
    return response


# ---------------------------------------------------------------------------
# assess_if_next_message_should_use_gov_uk_search
# ---------------------------------------------------------------------------


class TestAssessIfNextMessageShouldUseGovUkSearch:
    @pytest.mark.asyncio
    async def test_returns_true_when_llm_says_true(self):
        db_session = AsyncMock()
        db_session.execute = AsyncMock(return_value=MagicMock(fetchall=MagicMock(return_value=[])))
        db_session.execute.return_value.fetchall.return_value = []

        llm_response = _make_tool_use_response(
            "assess_if_gov_uk_search_should_be_used",
            {"use_gov_uk_search": True},
        )

        with (
            patch("app.gov_uk_search.service.LLMTable") as mock_llm_table,
            patch("app.gov_uk_search.service.BedrockHandler") as mock_bedrock,
            patch(
                "app.gov_uk_search.service.DbOperations.insert_llm_internal_response_id_query",
                new_callable=AsyncMock,
            ) as mock_insert,
            patch("app.gov_uk_search.service.insert"),
        ):
            mock_llm_table.return_value.get_by_model.return_value = MagicMock(
                input_cost_per_token=0.001, output_cost_per_token=0.002
            )
            mock_bedrock_instance = MagicMock()
            mock_bedrock_instance.invoke_async = AsyncMock(return_value=llm_response)
            mock_bedrock_instance.llm = MagicMock(input_cost_per_token=0.001, output_cost_per_token=0.002)
            mock_bedrock.return_value = mock_bedrock_instance

            mock_llm_internal = MagicMock()
            mock_llm_internal.id = 99
            mock_insert.return_value = mock_llm_internal

            messages = [_make_message(msg_id=1)]
            result = await assess_if_next_message_should_use_gov_uk_search(
                messages=messages,
                new_user_message_content="search for this",
                new_user_message_id=42,
                db_session=db_session,
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_llm_says_false(self):
        db_session = AsyncMock()
        db_session.execute = AsyncMock(return_value=MagicMock(fetchall=MagicMock(return_value=[])))

        llm_response = _make_tool_use_response(
            "assess_if_gov_uk_search_should_be_used",
            {"use_gov_uk_search": False},
        )

        with (
            patch("app.gov_uk_search.service.LLMTable") as mock_llm_table,
            patch("app.gov_uk_search.service.BedrockHandler") as mock_bedrock,
            patch(
                "app.gov_uk_search.service.DbOperations.insert_llm_internal_response_id_query", new_callable=AsyncMock
            ) as mock_insert,
        ):
            mock_llm_table.return_value.get_by_model.return_value = MagicMock(
                input_cost_per_token=0.001, output_cost_per_token=0.002
            )
            mock_bedrock_instance = MagicMock()
            mock_bedrock_instance.invoke_async = AsyncMock(return_value=llm_response)
            mock_bedrock_instance.llm = MagicMock(input_cost_per_token=0.001, output_cost_per_token=0.002)
            mock_bedrock.return_value = mock_bedrock_instance

            mock_llm_internal = MagicMock()
            mock_llm_internal.id = 99
            mock_insert.return_value = mock_llm_internal

            messages = [_make_message(msg_id=1)]
            result = await assess_if_next_message_should_use_gov_uk_search(
                messages=messages,
                new_user_message_content="just rewrite this",
                new_user_message_id=42,
                db_session=db_session,
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_llm_error(self):
        db_session = AsyncMock()
        db_session.execute = AsyncMock(return_value=MagicMock(fetchall=MagicMock(return_value=[])))

        with (
            patch("app.gov_uk_search.service.LLMTable") as mock_llm_table,
            patch("app.gov_uk_search.service.BedrockHandler") as mock_bedrock,
        ):
            mock_llm_table.return_value.get_by_model.return_value = MagicMock()
            mock_bedrock_instance = MagicMock()
            mock_bedrock_instance.invoke_async = AsyncMock(side_effect=Exception("Bedrock exploded"))
            mock_bedrock.return_value = mock_bedrock_instance

            messages = [_make_message(msg_id=1)]
            result = await assess_if_next_message_should_use_gov_uk_search(
                messages=messages,
                new_user_message_content="anything",
                new_user_message_id=42,
                db_session=db_session,
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_uses_summary_for_compacted_messages(self):
        """Messages with summaries should be sent as summaries, not raw content."""
        db_session = AsyncMock()
        db_session.execute = AsyncMock(return_value=MagicMock(fetchall=MagicMock(return_value=[])))

        llm_response = _make_tool_use_response(
            "assess_if_gov_uk_search_should_be_used",
            {"use_gov_uk_search": False},
        )

        captured_messages = []

        async def capture_invoke(**kwargs):
            captured_messages.extend(kwargs.get("messages", []))
            return llm_response

        with (
            patch("app.gov_uk_search.service.LLMTable") as mock_llm_table,
            patch("app.gov_uk_search.service.BedrockHandler") as mock_bedrock,
            patch(
                "app.gov_uk_search.service.DbOperations.insert_llm_internal_response_id_query", new_callable=AsyncMock
            ) as mock_insert,
        ):
            mock_llm_table.return_value.get_by_model.return_value = MagicMock(
                input_cost_per_token=0.001, output_cost_per_token=0.002
            )
            mock_bedrock_instance = MagicMock()
            mock_bedrock_instance.invoke_async = capture_invoke
            mock_bedrock_instance.llm = MagicMock(input_cost_per_token=0.001, output_cost_per_token=0.002)
            mock_bedrock.return_value = mock_bedrock_instance
            mock_insert.return_value = MagicMock(id=99)

            messages = [
                _make_message(msg_id=1, role="user", content="original long content", summary="short summary"),
                _make_message(msg_id=2, role="assistant", content="assistant reply", summary=None),
            ]
            await assess_if_next_message_should_use_gov_uk_search(
                messages=messages,
                new_user_message_content="new query",
                new_user_message_id=99,
                db_session=db_session,
            )

        # The compacted message should appear as its summary, not original content
        contents = [m["content"] for m in captured_messages]
        assert any("short summary" in c for c in contents)
        assert not any("original long content" in c for c in contents)

    @pytest.mark.asyncio
    async def test_only_last_20_messages_sent(self):
        """Only the last 20 messages should be included in the context window."""
        db_session = AsyncMock()
        db_session.execute = AsyncMock(return_value=MagicMock(fetchall=MagicMock(return_value=[])))

        llm_response = _make_tool_use_response(
            "assess_if_gov_uk_search_should_be_used",
            {"use_gov_uk_search": False},
        )

        captured_messages = []

        async def capture_invoke(**kwargs):
            captured_messages.extend(kwargs.get("messages", []))
            return llm_response

        with (
            patch("app.gov_uk_search.service.LLMTable") as mock_llm_table,
            patch("app.gov_uk_search.service.BedrockHandler") as mock_bedrock,
            patch(
                "app.gov_uk_search.service.DbOperations.insert_llm_internal_response_id_query", new_callable=AsyncMock
            ) as mock_insert,
        ):
            mock_llm_table.return_value.get_by_model.return_value = MagicMock(
                input_cost_per_token=0.001, output_cost_per_token=0.002
            )
            mock_bedrock_instance = MagicMock()
            mock_bedrock_instance.invoke_async = capture_invoke
            mock_bedrock_instance.llm = MagicMock(input_cost_per_token=0.001, output_cost_per_token=0.002)
            mock_bedrock.return_value = mock_bedrock_instance
            mock_insert.return_value = MagicMock(id=99)

            # 50 alternating user/assistant messages — only last 20 + new message should be sent
            messages = []
            for i in range(50):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append(_make_message(msg_id=i, role=role, content=f"message {i}"))

            await assess_if_next_message_should_use_gov_uk_search(
                messages=messages,
                new_user_message_content="new query",
                new_user_message_id=999,
                db_session=db_session,
            )

        # 20 recent messages (some may be merged by prepare_message_objects_for_llm)
        # plus 1 new user message appended — total must be <= 21
        assert len(captured_messages) <= 21

    @pytest.mark.asyncio
    async def test_previous_urls_included_in_system_prompt(self):
        """Previously retrieved URLs should appear in the system prompt."""
        db_session = AsyncMock()

        url_row = MagicMock()
        url_row.url = "https://www.gov.uk/some-guidance"
        db_session.execute = AsyncMock(return_value=MagicMock(fetchall=MagicMock(return_value=[url_row])))

        llm_response = _make_tool_use_response(
            "assess_if_gov_uk_search_should_be_used",
            {"use_gov_uk_search": False},
        )

        captured_system = []

        async def capture_invoke(system=None, **kwargs):
            if system:
                captured_system.append(system)
            return llm_response

        with (
            patch("app.gov_uk_search.service.LLMTable") as mock_llm_table,
            patch("app.gov_uk_search.service.BedrockHandler") as mock_bedrock,
            patch(
                "app.gov_uk_search.service.DbOperations.insert_llm_internal_response_id_query", new_callable=AsyncMock
            ) as mock_insert,
        ):
            mock_llm_table.return_value.get_by_model.return_value = MagicMock(
                input_cost_per_token=0.001, output_cost_per_token=0.002
            )
            mock_bedrock_instance = MagicMock()
            mock_bedrock_instance.invoke_async = capture_invoke
            mock_bedrock_instance.llm = MagicMock(input_cost_per_token=0.001, output_cost_per_token=0.002)
            mock_bedrock.return_value = mock_bedrock_instance
            mock_insert.return_value = MagicMock(id=99)

            messages = [_make_message(msg_id=1)]
            await assess_if_next_message_should_use_gov_uk_search(
                messages=messages,
                new_user_message_content="tell me more",
                new_user_message_id=42,
                db_session=db_session,
            )

        assert captured_system
        assert "https://www.gov.uk/some-guidance" in captured_system[0]


# ---------------------------------------------------------------------------
# get_search_queries
# ---------------------------------------------------------------------------


class TestGetSearchQueries:
    @pytest.mark.asyncio
    async def test_returns_search_terms_from_tool_call(self):
        db_session = AsyncMock()

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.input = {"search_terms": ["cycling safety", "bike lanes uk"]}
        response = MagicMock()
        response.content = [tool_block]
        response.usage = MagicMock(input_tokens=20, output_tokens=10)
        response.dict.return_value = {
            "content": [{"type": "tool_use", "input": {"search_terms": ["cycling safety", "bike lanes uk"]}}],
            "stop_reason": "tool_use",
        }

        mock_llm_internal = MagicMock()
        mock_llm_internal.id = 7

        with (
            patch("app.gov_uk_search.service.LLMTable") as mock_llm_table,
            patch("app.gov_uk_search.service.BedrockHandler") as mock_bedrock,
            patch(
                "app.gov_uk_search.service.DbOperations.insert_llm_internal_response_id_query", new_callable=AsyncMock
            ) as mock_insert,
        ):
            mock_llm_table.return_value.get_by_model.return_value = MagicMock(
                input_cost_per_token=0.001, output_cost_per_token=0.002
            )
            mock_bedrock_instance = MagicMock()
            mock_bedrock_instance.invoke_async = AsyncMock(return_value=response)
            mock_bedrock_instance.llm = MagicMock(input_cost_per_token=0.001, output_cost_per_token=0.002)
            mock_bedrock.return_value = mock_bedrock_instance
            mock_insert.return_value = mock_llm_internal

            search_terms, llm_id, params, cost = await get_search_queries(
                role="user", query="cycling safety in the UK", db_session=db_session
            )

        assert set(search_terms) == {"cycling safety", "bike lanes uk"}
        assert llm_id == 7
        assert isinstance(cost, Decimal)

    @pytest.mark.asyncio
    async def test_returns_empty_terms_on_no_tool_call(self):
        db_session = AsyncMock()

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "I cannot search right now"
        response = MagicMock()
        response.content = [text_block]
        response.usage = MagicMock(input_tokens=10, output_tokens=5)
        response.dict.return_value = {
            "content": [{"type": "text", "text": "I cannot search right now"}],
            "stop_reason": "end_turn",
        }

        mock_llm_internal = MagicMock()
        mock_llm_internal.id = 1

        with (
            patch("app.gov_uk_search.service.LLMTable") as mock_llm_table,
            patch("app.gov_uk_search.service.BedrockHandler") as mock_bedrock,
            patch(
                "app.gov_uk_search.service.DbOperations.insert_llm_internal_response_id_query", new_callable=AsyncMock
            ) as mock_insert,
        ):
            mock_llm_table.return_value.get_by_model.return_value = MagicMock(
                input_cost_per_token=0.001, output_cost_per_token=0.002
            )
            mock_bedrock_instance = MagicMock()
            mock_bedrock_instance.invoke_async = AsyncMock(return_value=response)
            mock_bedrock_instance.llm = MagicMock(input_cost_per_token=0.001, output_cost_per_token=0.002)
            mock_bedrock.return_value = mock_bedrock_instance
            mock_insert.return_value = mock_llm_internal

            search_terms, _, _, _ = await get_search_queries(role="user", query="something", db_session=db_session)

        assert search_terms == []

    @pytest.mark.asyncio
    async def test_conversation_context_included_when_messages_provided(self):
        """Recent messages should appear in the LLM prompt when provided."""
        db_session = AsyncMock()

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "no search needed"
        response = MagicMock()
        response.content = [text_block]
        response.usage = MagicMock(input_tokens=10, output_tokens=5)
        response.dict.return_value = {
            "content": [{"type": "text", "text": "no search needed"}],
            "stop_reason": "end_turn",
        }

        captured_messages = []

        async def capture_invoke(messages, **kwargs):
            captured_messages.extend(messages)
            return response

        mock_llm_internal = MagicMock()
        mock_llm_internal.id = 1

        with (
            patch("app.gov_uk_search.service.LLMTable") as mock_llm_table,
            patch("app.gov_uk_search.service.BedrockHandler") as mock_bedrock,
            patch(
                "app.gov_uk_search.service.DbOperations.insert_llm_internal_response_id_query", new_callable=AsyncMock
            ) as mock_insert,
        ):
            mock_llm_table.return_value.get_by_model.return_value = MagicMock(
                input_cost_per_token=0.001, output_cost_per_token=0.002
            )
            mock_bedrock_instance = MagicMock()
            mock_bedrock_instance.invoke_async = capture_invoke
            mock_bedrock_instance.llm = MagicMock(input_cost_per_token=0.001, output_cost_per_token=0.002)
            mock_bedrock.return_value = mock_bedrock_instance
            mock_insert.return_value = mock_llm_internal

            prior_messages = [
                _make_message(msg_id=1, role="user", content="tell me about cycling"),
                _make_message(msg_id=2, role="assistant", content="here is info about cycling"),
            ]
            await get_search_queries(
                role="user",
                query="now get the same doc for 2022",
                db_session=db_session,
                messages=prior_messages,
            )

        assert captured_messages
        combined_content = " ".join(m["content"] for m in captured_messages)
        assert "cycling" in combined_content

    @pytest.mark.asyncio
    async def test_no_conversation_context_when_no_messages(self):
        """Prompt should not contain a recent-conversation block if no messages are passed."""
        db_session = AsyncMock()

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "no search"
        response = MagicMock()
        response.content = [text_block]
        response.usage = MagicMock(input_tokens=10, output_tokens=5)
        response.dict.return_value = {
            "content": [{"type": "text", "text": "no search"}],
            "stop_reason": "end_turn",
        }

        captured_messages = []

        async def capture_invoke(messages, **kwargs):
            captured_messages.extend(messages)
            return response

        mock_llm_internal = MagicMock()
        mock_llm_internal.id = 1

        with (
            patch("app.gov_uk_search.service.LLMTable") as mock_llm_table,
            patch("app.gov_uk_search.service.BedrockHandler") as mock_bedrock,
            patch(
                "app.gov_uk_search.service.DbOperations.insert_llm_internal_response_id_query", new_callable=AsyncMock
            ) as mock_insert,
        ):
            mock_llm_table.return_value.get_by_model.return_value = MagicMock(
                input_cost_per_token=0.001, output_cost_per_token=0.002
            )
            mock_bedrock_instance = MagicMock()
            mock_bedrock_instance.invoke_async = capture_invoke
            mock_bedrock_instance.llm = MagicMock(input_cost_per_token=0.001, output_cost_per_token=0.002)
            mock_bedrock.return_value = mock_bedrock_instance
            mock_insert.return_value = mock_llm_internal

            await get_search_queries(role="user", query="anything", db_session=db_session)

        combined_content = " ".join(m["content"] for m in captured_messages)
        assert "<recent-conversation>" not in combined_content


# ---------------------------------------------------------------------------
# assess_document_relevancy
# ---------------------------------------------------------------------------


class TestAssessDocumentRelevancy:
    @pytest.mark.asyncio
    async def test_returns_true_for_relevant_document(self):
        db_session = AsyncMock()

        llm_response = _make_tool_use_response("assess_document_relevance", {"is_relevant": True})
        llm_response.dict.return_value = {"content": [{"type": "tool_use", "input": {"is_relevant": True}}]}

        mock_llm_internal = MagicMock()
        mock_llm_internal.id = 5

        with (
            patch("app.gov_uk_search.service.LLMTable") as mock_llm_table,
            patch("app.gov_uk_search.service.BedrockHandler") as mock_bedrock,
            patch(
                "app.gov_uk_search.service.DbOperations.insert_llm_internal_response_id_query", new_callable=AsyncMock
            ) as mock_insert,
        ):
            mock_llm_table.return_value.get_by_model.return_value = MagicMock(
                input_cost_per_token=0.001, output_cost_per_token=0.002
            )
            mock_bedrock_instance = MagicMock()
            mock_bedrock_instance.invoke_async = AsyncMock(return_value=llm_response)
            mock_bedrock_instance.llm = MagicMock(input_cost_per_token=0.001, output_cost_per_token=0.002)
            mock_bedrock.return_value = mock_bedrock_instance
            mock_insert.return_value = mock_llm_internal

            is_relevant, cost, llm_id = await assess_document_relevancy(
                role="user",
                query="cycling safety",
                title="Cycling Safety Guide",
                description="A guide to cycling safety on UK roads",
                full_content="Full content about cycling on UK roads...",
                db_session=db_session,
            )

        assert is_relevant is True
        assert isinstance(cost, Decimal)
        assert llm_id == 5

    @pytest.mark.asyncio
    async def test_returns_false_for_irrelevant_document(self):
        db_session = AsyncMock()

        llm_response = _make_tool_use_response("assess_document_relevance", {"is_relevant": False})
        llm_response.dict.return_value = {"content": [{"type": "tool_use", "input": {"is_relevant": False}}]}

        mock_llm_internal = MagicMock()
        mock_llm_internal.id = 5

        with (
            patch("app.gov_uk_search.service.LLMTable") as mock_llm_table,
            patch("app.gov_uk_search.service.BedrockHandler") as mock_bedrock,
            patch(
                "app.gov_uk_search.service.DbOperations.insert_llm_internal_response_id_query", new_callable=AsyncMock
            ) as mock_insert,
        ):
            mock_llm_table.return_value.get_by_model.return_value = MagicMock(
                input_cost_per_token=0.001, output_cost_per_token=0.002
            )
            mock_bedrock_instance = MagicMock()
            mock_bedrock_instance.invoke_async = AsyncMock(return_value=llm_response)
            mock_bedrock_instance.llm = MagicMock(input_cost_per_token=0.001, output_cost_per_token=0.002)
            mock_bedrock.return_value = mock_bedrock_instance
            mock_insert.return_value = mock_llm_internal

            is_relevant, cost, _ = await assess_document_relevancy(
                role="user",
                query="cycling safety",
                title="Login Page",
                description="Please log in to continue",
                full_content="Username: Password:",
                db_session=db_session,
            )

        assert is_relevant is False

    @pytest.mark.asyncio
    async def test_tool_choice_is_forced(self):
        """invoke_async must be called with tool_choice forcing the assess_document_relevance tool."""
        db_session = AsyncMock()

        llm_response = _make_tool_use_response("assess_document_relevance", {"is_relevant": True})
        llm_response.dict.return_value = {"content": [{"type": "tool_use", "input": {"is_relevant": True}}]}

        captured_kwargs = {}

        async def capture_invoke(messages, **kwargs):
            captured_kwargs.update(kwargs)
            return llm_response

        with (
            patch("app.gov_uk_search.service.LLMTable") as mock_llm_table,
            patch("app.gov_uk_search.service.BedrockHandler") as mock_bedrock,
            patch(
                "app.gov_uk_search.service.DbOperations.insert_llm_internal_response_id_query", new_callable=AsyncMock
            ) as mock_insert,
        ):
            mock_llm_table.return_value.get_by_model.return_value = MagicMock(
                input_cost_per_token=0.001, output_cost_per_token=0.002
            )
            mock_bedrock_instance = MagicMock()
            mock_bedrock_instance.invoke_async = capture_invoke
            mock_bedrock_instance.llm = MagicMock(input_cost_per_token=0.001, output_cost_per_token=0.002)
            mock_bedrock.return_value = mock_bedrock_instance
            mock_insert.return_value = MagicMock(id=1)

            await assess_document_relevancy(
                role="user",
                query="test query",
                title="Some Page",
                description="Some description",
                full_content="Some content",
                db_session=db_session,
            )

        assert "tool_choice" in captured_kwargs
        assert captured_kwargs["tool_choice"]["type"] == "tool"
        assert captured_kwargs["tool_choice"]["name"] == "assess_document_relevance"

    @pytest.mark.asyncio
    async def test_returns_false_on_llm_error(self):
        db_session = AsyncMock()

        with (
            patch("app.gov_uk_search.service.LLMTable") as mock_llm_table,
            patch("app.gov_uk_search.service.BedrockHandler") as mock_bedrock,
        ):
            mock_llm_table.return_value.get_by_model.return_value = MagicMock()
            mock_bedrock_instance = MagicMock()
            mock_bedrock_instance.invoke_async = AsyncMock(side_effect=Exception("Bedrock timeout"))
            mock_bedrock.return_value = mock_bedrock_instance

            is_relevant, cost, llm_id = await assess_document_relevancy(
                role="user",
                query="test",
                title="Some Page",
                description="description",
                full_content="content",
                db_session=db_session,
            )

        assert is_relevant is False
        assert cost == Decimal(0)
        assert llm_id == 0

    @pytest.mark.asyncio
    async def test_content_is_truncated_to_6000_chars(self):
        """Full content longer than 6000 chars should be truncated before sending."""
        db_session = AsyncMock()

        llm_response = _make_tool_use_response("assess_document_relevance", {"is_relevant": True})
        llm_response.dict.return_value = {"content": [{"type": "tool_use", "input": {"is_relevant": True}}]}

        captured_messages = []

        async def capture_invoke(messages, **kwargs):
            captured_messages.extend(messages)
            return llm_response

        with (
            patch("app.gov_uk_search.service.LLMTable") as mock_llm_table,
            patch("app.gov_uk_search.service.BedrockHandler") as mock_bedrock,
            patch(
                "app.gov_uk_search.service.DbOperations.insert_llm_internal_response_id_query", new_callable=AsyncMock
            ) as mock_insert,
        ):
            mock_llm_table.return_value.get_by_model.return_value = MagicMock(
                input_cost_per_token=0.001, output_cost_per_token=0.002
            )
            mock_bedrock_instance = MagicMock()
            mock_bedrock_instance.invoke_async = capture_invoke
            mock_bedrock_instance.llm = MagicMock(input_cost_per_token=0.001, output_cost_per_token=0.002)
            mock_bedrock.return_value = mock_bedrock_instance
            mock_insert.return_value = MagicMock(id=1)

            long_content = "x" * 20000
            await assess_document_relevancy(
                role="user",
                query="test",
                title="Page",
                description="short description",
                full_content=long_content,
                db_session=db_session,
            )

        combined_content = " ".join(m["content"] for m in captured_messages)
        # 20000 chars of 'x' should not appear in full — truncated to ~6000
        assert "x" * 7000 not in combined_content
        assert "[content truncated]" in combined_content
