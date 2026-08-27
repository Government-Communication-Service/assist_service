from unittest.mock import patch

import pytest

from app.chat.utils import cap_enhanced_prompt_size

pytestmark = [pytest.mark.unit]


class TestCapEnhancedPromptSize:
    def test_no_enhanced_content_returns_query_unchanged(self):
        result = cap_enhanced_prompt_size(
            query="what is the weather like",
            enhanced_segments=[None, ""],
            existing_conversation_tokens=0,
            reserved_output_tokens=0,
        )
        assert result == "what is the weather like"

    def test_short_enhanced_content_is_untouched(self):
        result = cap_enhanced_prompt_size(
            query="my query",
            enhanced_segments=["some short rag content"],
            existing_conversation_tokens=0,
            reserved_output_tokens=0,
        )
        assert result == "my query\n\nsome short rag content"

    def test_truncates_to_max_enhanced_prompt_chars(self):
        huge_segment = "x" * 1000
        with (
            patch("app.chat.utils.MAX_ENHANCED_PROMPT_CHARS", 100),
            patch("app.chat.utils.CHAT_MODEL_CONTEXT_WINDOW_TOKENS", 1_000_000),
        ):
            result = cap_enhanced_prompt_size(
                query="my query",
                enhanced_segments=[huge_segment],
                existing_conversation_tokens=0,
                reserved_output_tokens=0,
            )

        assert "my query" in result
        assert "x" * 100 in result
        assert "x" * 101 not in result
        assert "[content truncated]" in result

    def test_truncates_more_aggressively_as_conversation_grows(self):
        huge_segment = "x" * 1000
        with (
            patch("app.chat.utils.MAX_ENHANCED_PROMPT_CHARS", 1_000_000),
            patch("app.chat.utils.CHAT_MODEL_CONTEXT_WINDOW_TOKENS", 1000),
            patch("app.chat.utils.CHARS_PER_TOKEN_ESTIMATE", 1),
        ):
            # Conversation already fills almost the whole context window, leaving
            # very little room for enhanced content even though the fixed cap is huge.
            result = cap_enhanced_prompt_size(
                query="my query",
                enhanced_segments=[huge_segment],
                existing_conversation_tokens=990,
                reserved_output_tokens=0,
            )

        assert "x" * 10 in result
        assert "x" * 11 not in result
        assert "[content truncated]" in result

    def test_never_truncates_the_query_itself(self):
        huge_segment = "x" * 1000
        with (
            patch("app.chat.utils.MAX_ENHANCED_PROMPT_CHARS", 1_000_000),
            patch("app.chat.utils.CHAT_MODEL_CONTEXT_WINDOW_TOKENS", 1000),
            patch("app.chat.utils.CHARS_PER_TOKEN_ESTIMATE", 1),
        ):
            # No room at all left for enhanced content.
            result = cap_enhanced_prompt_size(
                query="my important query",
                enhanced_segments=[huge_segment],
                existing_conversation_tokens=1000,
                reserved_output_tokens=0,
            )

        assert result.startswith("my important query")
