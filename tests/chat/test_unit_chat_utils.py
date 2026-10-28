from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.chat.utils import prepare_recent_turns_for_decision, truncate_middle, truncate_query_for_helper_call
from app.config import settings

pytestmark = [pytest.mark.unit]


def make_message(role, content, summary=None, content_enhanced_with_rag=None):
    return SimpleNamespace(
        role=role, content=content, summary=summary, content_enhanced_with_rag=content_enhanced_with_rag
    )


class TestPrepareRecentTurnsForDecision:
    def test_empty_messages_returns_empty_list(self):
        assert prepare_recent_turns_for_decision([], num_turns=6) == []

    def test_uses_raw_content_not_enhanced_content(self):
        """Regression: decision-making calls must not carry the expensive RAG-enhanced content."""
        messages = [make_message("user", "raw query", content_enhanced_with_rag="expensive rag content")]

        result = prepare_recent_turns_for_decision(messages, num_turns=6)

        assert result == [{"role": "user", "content": "raw query"}]

    def test_ignores_legacy_per_message_summary(self):
        """Message.summary is legacy (superseded by ChatCompaction) and must not be used."""
        messages = [make_message("user", "raw query", summary="a short summary")]

        result = prepare_recent_turns_for_decision(messages, num_turns=6)

        assert result == [{"role": "user", "content": "raw query"}]

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

    def test_cuts_long_user_turn_keeping_start_and_end(self):
        """Regression: a pasted document must not be sent whole to every helper call."""
        long_query = "a" * 3000 + "b" * 10_000 + "c" * 3000
        messages = [make_message("user", long_query)]

        with patch.object(settings, "helper_call_max_user_turn_chars", 4000):
            result = prepare_recent_turns_for_decision(messages, num_turns=6)

        content = result[0]["content"]
        assert content.startswith("a" * 2000)
        assert content.endswith("c" * 2000)
        assert "12,000 characters omitted" in content
        assert len(content) < 4100

    def test_drops_oldest_turns_past_total_limit(self):
        messages = [
            make_message("user", "u1" * 50),
            make_message("assistant", "a1" * 50),
            make_message("user", "u2" * 50),
            make_message("assistant", "a2" * 50),
            make_message("user", "latest"),
        ]

        with patch.object(settings, "helper_call_max_context_chars", 250):
            result = prepare_recent_turns_for_decision(messages, num_turns=6)

        # Dropping u1 and a1 brings it under the limit; the result starts with a user turn.
        assert [m["content"] for m in result] == ["u2" * 50, "a2" * 50, "latest"]

    def test_starts_with_user_turn_after_dropping(self):
        messages = [
            make_message("user", "u1" * 100),
            make_message("assistant", "a1" * 50),
            make_message("user", "latest"),
        ]

        with patch.object(settings, "helper_call_max_context_chars", 150):
            result = prepare_recent_turns_for_decision(messages, num_turns=6)

        assert result == [{"role": "user", "content": "latest"}]

    def test_latest_turn_is_kept_and_cut_when_it_alone_is_too_long(self):
        messages = [make_message("user", "x" * 3000), make_message("user", "y" * 3000)]

        with (
            patch.object(settings, "helper_call_max_user_turn_chars", 4000),
            patch.object(settings, "helper_call_max_context_chars", 1000),
        ):
            result = prepare_recent_turns_for_decision(messages, num_turns=6)

        assert len(result) == 1
        assert result[0]["content"].startswith("x" * 500)
        assert result[0]["content"].endswith("y" * 500)
        assert len(result[0]["content"]) < 1100


class TestTruncateMiddle:
    def test_short_text_is_unchanged(self):
        assert truncate_middle("hello", 10) == "hello"

    def test_long_text_keeps_head_and_tail(self):
        assert truncate_middle("abcdefghij", 4) == "ab\n\n[... 6 characters omitted ...]\n\nij"

    def test_query_for_helper_call_uses_setting(self):
        with patch.object(settings, "helper_call_max_query_chars", 6):
            assert truncate_query_for_helper_call("abcdefghij").startswith("abc\n")
        assert truncate_query_for_helper_call(None) == ""
