from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.chat.utils import cap_enhanced_prompt_size, prepare_recent_turns_for_decision

pytestmark = [pytest.mark.unit]


def make_message(role, content, summary=None, content_enhanced_with_rag=None):
    return SimpleNamespace(
        role=role, content=content, summary=summary, content_enhanced_with_rag=content_enhanced_with_rag
    )


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


class TestPrepareRecentTurnsForDecision:
    def test_empty_messages_returns_empty_list(self):
        assert prepare_recent_turns_for_decision([], num_turns=6) == []

    def test_uses_raw_content_not_enhanced_content(self):
        """Regression: decision-making calls must not carry the expensive RAG-enhanced content."""
        messages = [make_message("user", "raw query", content_enhanced_with_rag="expensive rag content")]

        result = prepare_recent_turns_for_decision(messages, num_turns=6)

        assert result == [{"role": "user", "content": "raw query"}]

    def test_uses_summary_when_present(self):
        messages = [make_message("user", "raw query", summary="a short summary")]

        result = prepare_recent_turns_for_decision(messages, num_turns=6)

        assert result == [{"role": "user", "content": "a short summary"}]

    def test_truncates_assistant_content_to_preview(self):
        long_reply = "y" * 500
        messages = [make_message("assistant", long_reply)]

        result = prepare_recent_turns_for_decision(messages, num_turns=6)

        assert result == [{"role": "assistant", "content": "y" * 200}]

    def test_slices_to_last_num_turns_internally(self):
        messages = [make_message("user" if i % 2 == 0 else "assistant", f"message {i}") for i in range(10)]

        result = prepare_recent_turns_for_decision(messages, num_turns=3)

        assert [m["content"] for m in result] == ["message 7", "message 8", "message 9"]

    def test_num_turns_zero_returns_all_messages(self):
        messages = [make_message("user" if i % 2 == 0 else "assistant", f"message {i}") for i in range(3)]

        result = prepare_recent_turns_for_decision(messages, num_turns=0)

        assert len(result) == 3

    def test_merges_consecutive_user_messages(self):
        """Regression: the result must be safe to pass directly as a Messages API `messages`
        list, which requires strict user/assistant alternation."""
        messages = [make_message("user", "first"), make_message("user", "second")]

        result = prepare_recent_turns_for_decision(messages, num_turns=6)

        assert result == [{"role": "user", "content": "first\n\nsecond"}]

    def test_skips_empty_content_assistant_turns(self):
        """Regression: tool-only assistant turns are persisted with content="" - including
        them would emit an empty-content block, which the Messages API rejects."""
        messages = [
            make_message("user", "first query"),
            make_message("assistant", ""),
            make_message("user", "second query"),
        ]

        result = prepare_recent_turns_for_decision(messages, num_turns=6)

        assert result == [{"role": "user", "content": "first query\n\nsecond query"}]
