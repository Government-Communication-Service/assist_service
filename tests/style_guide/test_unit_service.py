"""
Unit tests for style guide service functions.
Tests the router logic, conversation wrapping, and simple response handler.
"""
import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest
from anthropic.types import Message as AnthropicMessage
from anthropic.types import TextBlock, Usage

from app.database.models import Message
from app.style_guide.service import (
    StyleGuideContentType,
    _get_document_content_for_style_guide,
    _wrap_conversation_context,
    determine_style_guide_content_type,
)

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.unit,
]


def create_mock_message(role: str, content: str, message_id: int = 1) -> Message:
    """Helper to create mock Message objects."""
    message = Mock(spec=Message)
    message.id = message_id
    message.role = role
    message.content = content
    message.content_enhanced_with_rag = content
    message.summary = None  # Explicitly set to None so hasattr check works properly
    return message


def create_mock_llm_response(text: str) -> AnthropicMessage:
    """Helper to create mock Anthropic LLM response."""
    return AnthropicMessage(
        id="msg_test",
        content=[TextBlock(text=text, type="text")],
        model="test-model",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage=Usage(input_tokens=100, output_tokens=50),
    )


class TestConversationContextWrapping:
    """Tests for _wrap_conversation_context function."""

    def test_empty_messages_list(self):
        """Test that empty messages list returns empty string."""
        result = _wrap_conversation_context([])
        assert result == ""

    def test_single_message(self):
        """Test wrapping a single message."""
        messages = [create_mock_message("user", "Check this text")]
        result = _wrap_conversation_context(messages)

        assert "<ConversationContext>" in result
        assert "</ConversationContext>" in result
        assert "<user>" in result
        assert "Check this text" in result
        assert "</user>" in result

    def test_multiple_messages(self):
        """Test wrapping multiple messages in conversation order."""
        messages = [
            create_mock_message("user", "Check this document", 1),
            create_mock_message("assistant", "Here are the violations", 2),
            create_mock_message("user", "Remove the sort code", 3),
        ]
        result = _wrap_conversation_context(messages)

        assert result.count("<user>") == 2
        assert result.count("<assistant>") == 1
        assert "Check this document" in result
        assert "Here are the violations" in result
        assert "Remove the sort code" in result


class TestContentTypeRouter:
    """Tests for determine_style_guide_content_type router function."""

    @pytest.mark.asyncio
    async def test_empty_messages_returns_query_text(self):
        """Test that empty messages returns QUERY_TEXT as default."""
        result = await determine_style_guide_content_type([], has_documents=False)
        assert result == StyleGuideContentType.QUERY_TEXT

    @pytest.mark.asyncio
    @patch("app.style_guide.service.LLMTable")
    @patch("app.style_guide.service.BedrockHandler")
    async def test_router_decides_check_query_text(self, mock_bedrock_handler, mock_llm_table):
        """Test LLM router deciding to check query text on a follow-up message."""
        # Setup mocks
        mock_llm = Mock()
        mock_llm_table.return_value.get_by_model.return_value = mock_llm

        mock_handler_instance = Mock()
        mock_handler_instance.invoke_async = AsyncMock(
            return_value=create_mock_llm_response("CHECK_QUERY_TEXT")
        )
        mock_bedrock_handler.return_value = mock_handler_instance

        # Use a follow-up (multi-message) conversation so the LLM router is invoked.
        # Single-message conversations use a deterministic fast-path that skips the LLM.
        messages = [
            create_mock_message("user", "Check this paragraph: hello world", 1),
            create_mock_message("assistant", "Here are the violations found.", 2),
            create_mock_message("user", "Check this new text: some text here", 3),
        ]

        result = await determine_style_guide_content_type(messages, has_documents=False)

        assert result == StyleGuideContentType.QUERY_TEXT
        mock_handler_instance.invoke_async.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.style_guide.service.LLMTable")
    @patch("app.style_guide.service.BedrockHandler")
    async def test_router_decides_check_documents(self, mock_bedrock_handler, mock_llm_table):
        """Test LLM router deciding to check documents on a follow-up message."""
        # Setup mocks
        mock_llm = Mock()
        mock_llm_table.return_value.get_by_model.return_value = mock_llm

        mock_handler_instance = Mock()
        mock_handler_instance.invoke_async = AsyncMock(
            return_value=create_mock_llm_response("CHECK_DOCUMENTS")
        )
        mock_bedrock_handler.return_value = mock_handler_instance

        # Use a follow-up (multi-message) conversation so the LLM router is invoked.
        messages = [
            create_mock_message("user", "Check my uploaded document", 1),
            create_mock_message("assistant", "Found 3 violations.", 2),
            create_mock_message("user", "Now check the second document too", 3),
        ]

        result = await determine_style_guide_content_type(messages, has_documents=True)

        assert result == StyleGuideContentType.DOCUMENTS
        mock_handler_instance.invoke_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_initial_message_without_documents_skips_llm(self):
        """Test that an initial (single) message without documents uses the deterministic
        fast-path and returns QUERY_TEXT without calling the LLM."""
        messages = [create_mock_message("user", "Check this text: hello world")]
        result = await determine_style_guide_content_type(messages, has_documents=False)
        assert result == StyleGuideContentType.QUERY_TEXT

    @pytest.mark.asyncio
    async def test_initial_message_with_documents_skips_llm(self):
        """Test that an initial (single) message with documents uses the deterministic
        fast-path and returns DOCUMENTS without calling the LLM."""
        messages = [create_mock_message("user", "Check my document")]
        result = await determine_style_guide_content_type(messages, has_documents=True)
        assert result == StyleGuideContentType.DOCUMENTS

    @pytest.mark.asyncio
    @patch("app.style_guide.service.LLMTable")
    @patch("app.style_guide.service.BedrockHandler")
    async def test_router_fallback_when_no_documents(self, mock_bedrock_handler, mock_llm_table):
        """Test router falling back to QUERY_TEXT when documents requested but none available."""
        # Setup mocks
        mock_llm = Mock()
        mock_llm_table.return_value.get_by_model.return_value = mock_llm

        mock_handler_instance = Mock()
        mock_handler_instance.invoke_async = AsyncMock(
            return_value=create_mock_llm_response("CHECK_DOCUMENTS")
        )
        mock_bedrock_handler.return_value = mock_handler_instance

        messages = [create_mock_message("user", "Check my document")]

        # No documents available
        result = await determine_style_guide_content_type(messages, has_documents=False)

        # Should fall back to QUERY_TEXT
        assert result == StyleGuideContentType.QUERY_TEXT

    @pytest.mark.asyncio
    @patch("app.style_guide.service.LLMTable")
    @patch("app.style_guide.service.BedrockHandler")
    async def test_router_decides_simple_response(self, mock_bedrock_handler, mock_llm_table):
        """Test router deciding simple response for follow-up questions."""
        # Setup mocks
        mock_llm = Mock()
        mock_llm_table.return_value.get_by_model.return_value = mock_llm

        mock_handler_instance = Mock()
        mock_handler_instance.invoke_async = AsyncMock(
            return_value=create_mock_llm_response("SIMPLE_RESPONSE")
        )
        mock_bedrock_handler.return_value = mock_handler_instance

        messages = [
            create_mock_message("user", "Check this", 1),
            create_mock_message("assistant", "Found violations", 2),
            create_mock_message("user", "Make it shorter", 3),
        ]

        result = await determine_style_guide_content_type(messages, has_documents=False)

        assert result == StyleGuideContentType.SIMPLE_RESPONSE

    @pytest.mark.asyncio
    @patch("app.style_guide.service.LLMTable")
    async def test_router_error_on_initial_message(self, mock_llm_table):
        """Test router defaults to QUERY_TEXT on error for initial messages."""
        mock_llm_table.return_value.get_by_model.return_value = None

        messages = [create_mock_message("user", "Check this")]

        result = await determine_style_guide_content_type(messages, has_documents=False)

        assert result == StyleGuideContentType.QUERY_TEXT

    @pytest.mark.asyncio
    @patch("app.style_guide.service.LLMTable")
    async def test_router_error_on_follow_up(self, mock_llm_table):
        """Test router defaults to QUERY_TEXT on error when LLM not available."""
        mock_llm_table.return_value.get_by_model.return_value = None

        messages = [
            create_mock_message("user", "Check this", 1),
            create_mock_message("assistant", "Done", 2),
            create_mock_message("user", "Follow up", 3),
        ]

        result = await determine_style_guide_content_type(messages, has_documents=False)

        # When LLM not available (returns None), defaults to QUERY_TEXT via early return
        assert result == StyleGuideContentType.QUERY_TEXT

    @pytest.mark.asyncio
    @patch("app.style_guide.service.LLMTable")
    @patch("app.style_guide.service.BedrockHandler")
    async def test_router_exception_during_follow_up_returns_simple_response(
        self, mock_bedrock_handler, mock_llm_table
    ):
        """Test that a real exception during the follow-up LLM call returns SIMPLE_RESPONSE.

        This tests the except-clause fallback path which is distinct from the
        'llm_obj is None' early-return path tested above.  When an exception is
        raised *after* the LLM model has been retrieved (e.g. the Bedrock call
        itself fails), the router defaults to SIMPLE_RESPONSE for follow-up
        conversations so that the main LLM handles the reply conversationally.
        """
        mock_llm = Mock()
        mock_llm_table.return_value.get_by_model.return_value = mock_llm

        mock_handler_instance = Mock()
        mock_handler_instance.invoke_async = AsyncMock(
            side_effect=RuntimeError("Bedrock unavailable")
        )
        mock_bedrock_handler.return_value = mock_handler_instance

        messages = [
            create_mock_message("user", "Check this text", 1),
            create_mock_message("assistant", "Found violations", 2),
            create_mock_message("user", "Follow up question", 3),
        ]

        result = await determine_style_guide_content_type(messages, has_documents=False)

        assert result == StyleGuideContentType.SIMPLE_RESPONSE
        mock_handler_instance.invoke_async.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.style_guide.service.LLMTable")
    @patch("app.style_guide.service.BedrockHandler")
    async def test_router_multi_message_check_documents_no_documents_falls_back(
        self, mock_bedrock_handler, mock_llm_table
    ):
        """Multi-message: LLM returns CHECK_DOCUMENTS but no docs selected → QUERY_TEXT.

        The existing ``test_router_fallback_when_no_documents`` uses a *single*
        message which takes the deterministic fast-path and never reaches the
        LLM.  This test exercises the actual LLM-fallback code branch: a
        multi-message conversation where the LLM recommends checking documents
        but none are available, so the router falls back to QUERY_TEXT.
        """
        mock_llm = Mock()
        mock_llm_table.return_value.get_by_model.return_value = mock_llm

        mock_handler_instance = Mock()
        mock_handler_instance.invoke_async = AsyncMock(
            return_value=create_mock_llm_response("CHECK_DOCUMENTS")
        )
        mock_bedrock_handler.return_value = mock_handler_instance

        messages = [
            create_mock_message("user", "Check my uploaded document", 1),
            create_mock_message("assistant", "Found 3 violations.", 2),
            create_mock_message("user", "Check the document again", 3),
        ]

        # No documents available
        result = await determine_style_guide_content_type(messages, has_documents=False)

        assert result == StyleGuideContentType.QUERY_TEXT
        # The LLM *was* called – this is the live fallback path, not the deterministic path
        mock_handler_instance.invoke_async.assert_called_once()


class TestDocumentContentRetrieval:
    """Tests for _get_document_content_for_style_guide function."""

    @pytest.mark.asyncio
    async def test_empty_document_uuids(self):
        """Test that empty document UUIDs returns empty content."""
        content, names = await _get_document_content_for_style_guide([], user_id=1)
        assert content == ""
        assert names == []

    @pytest.mark.asyncio
    async def test_none_document_uuids(self):
        """Test that None document UUIDs returns empty content."""
        content, names = await _get_document_content_for_style_guide(None, user_id=1)
        assert content == ""
        assert names == []

    @pytest.mark.asyncio
    @patch("app.style_guide.service.async_db_session")
    async def test_retrieves_document_content(self, mock_async_db_session):
        """Test successful document content retrieval."""
        # Setup mock document and chunks
        mock_document = Mock()
        mock_document.id = 1
        mock_document.name = "test_doc.pdf"
        mock_document.uuid = "test-uuid-123"

        mock_chunk1 = Mock()
        mock_chunk1.content = "First chunk content"
        mock_chunk2 = Mock()
        mock_chunk2.content = "Second chunk content"

        # Setup mock database session
        mock_session = AsyncMock()

        # Mock the access check query (returns something = has access)
        mock_access_result = Mock()
        mock_access_result.scalar_one_or_none.return_value = Mock()

        # Mock the document query
        mock_doc_result = Mock()
        mock_doc_result.scalar_one_or_none.return_value = mock_document

        # Mock the chunks query
        mock_chunks_result = Mock()
        mock_chunks_result.scalars.return_value.all.return_value = [mock_chunk1, mock_chunk2]

        # Configure mock session to return different results for each execute call
        mock_session.execute = AsyncMock(side_effect=[
            mock_access_result,
            mock_doc_result,
            mock_chunks_result,
        ])

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_session
        mock_context_manager.__aexit__.return_value = None
        mock_async_db_session.return_value = mock_context_manager

        content, names = await _get_document_content_for_style_guide(
            document_uuids=["test-uuid-123"],
            user_id=1
        )

        assert "test_doc.pdf" in content
        assert "First chunk content" in content
        assert "Second chunk content" in content
        assert "test_doc.pdf" in names

    @pytest.mark.asyncio
    @patch("app.style_guide.service.async_db_session")
    async def test_user_access_denied(self, mock_async_db_session):
        """Test that documents without user access are skipped."""
        # Setup mock database session
        mock_session = AsyncMock()

        # Mock the access check query (returns None = no access)
        mock_access_result = Mock()
        mock_access_result.scalar_one_or_none.return_value = None

        mock_session.execute = AsyncMock(return_value=mock_access_result)

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__.return_value = mock_session
        mock_context_manager.__aexit__.return_value = None
        mock_async_db_session.return_value = mock_context_manager

        content, names = await _get_document_content_for_style_guide(
            document_uuids=["test-uuid-123"],
            user_id=1
        )

        assert content == ""
        assert names == []
