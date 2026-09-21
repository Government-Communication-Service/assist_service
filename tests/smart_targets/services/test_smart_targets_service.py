from unittest.mock import Mock

from app.smart_targets.service import NUM_RECENT_TURNS_FOR_DECISION, SmartTargetsService


def _make_message(role: str, content: str) -> Mock:
    m = Mock()
    m.role = role
    m.content = content
    return m


class TestSmartTargetsService:
    async def test_no_metric_selected(self, mock_get_available_metrics, mock_llm_smart_targets_choice, mock_messages):
        service = SmartTargetsService()
        available_metrics = await service.get_available_metrics()
        r = await service._select_metrics(messages=mock_messages, available_metrics=available_metrics)
        assert len(r) == 0, f"Expected no metrics to be selected, got {len(r)}"


class TestWrapChatMessages:
    """Regression tests: `_wrap_chat_messages` must bound the conversation it sends to
    LLM_SMART_TARGETS_MODEL, which has a much smaller context window than the main chat
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

        # Only the last NUM_RECENT_TURNS_FOR_DECISION messages should survive.
        oldest_kept_index = len(messages) - NUM_RECENT_TURNS_FOR_DECISION
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
