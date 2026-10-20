from unittest.mock import Mock

from app.smart_targets.service import CAMPAIGN_BENCHMARKS_RECENT_TURNS, SmartTargetsService


def _make_message(role: str, content: str) -> Mock:
    m = Mock()
    m.role = role
    m.content = content
    return m


class TestWrapChatMessages:
    """Regression tests: `_wrap_chat_messages` must bound the conversation it sends to
    LLM_CAMPAIGN_BENCHMARKS_MODEL, which has a much smaller context window than the main chat
    model and, unlike the main chat completion, is never given a compacted history.
    """

    def test_bounds_conversation_to_recent_turns(self):
        service = SmartTargetsService()
        num_full_turns = 20
        messages = []
        for i in range(num_full_turns):
            messages.append(_make_message("user", f"user-turn-{i}"))
            messages.append(_make_message("assistant", f"assistant-turn-{i}"))

        wrapped = service._wrap_chat_messages(messages)

        # Only the last CAMPAIGN_BENCHMARKS_RECENT_TURNS messages should survive.
        oldest_kept_index = len(messages) - CAMPAIGN_BENCHMARKS_RECENT_TURNS
        dropped_turn = oldest_kept_index // 2 - 1
        assert f"user-turn-{dropped_turn}" not in wrapped
        assert f"assistant-turn-{dropped_turn}" not in wrapped
        assert f"user-turn-{num_full_turns - 1}" in wrapped
        assert f"assistant-turn-{num_full_turns - 1}" in wrapped

    def test_excludes_rag_enhanced_content(self):
        """content_enhanced_with_rag can be very large; the LLM sent to the smaller-context
        smart-targets model must only ever see raw message content, never that blown-up
        RAG-enhanced version.
        """
        service = SmartTargetsService()
        message = _make_message("user", "short question about campaign performance")
        message.content_enhanced_with_rag = "X" * 300_000

        wrapped = service._wrap_chat_messages([message])

        assert "short question about campaign performance" in wrapped
        assert "X" * 300_000 not in wrapped
        assert len(wrapped) < 1_000
