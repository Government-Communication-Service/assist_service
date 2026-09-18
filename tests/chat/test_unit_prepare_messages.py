"""Unit tests for prepare_message_objects_for_llm.

Covers substituting a compaction summary for the history it replaces. The prompt-cache
breakpoint applied on top of this output is tested separately in
tests/chat/test_unit_cache_control.py.
"""

import logging
from unittest.mock import patch

from app.chat.utils import prepare_message_objects_for_llm
from app.config import settings
from app.database.models import ChatCompaction, Message

logger = logging.getLogger(__name__)


def make_message(message_id: int, role: str, content: str, content_enhanced_with_rag: str | None = None) -> Message:
    return Message(
        id=message_id,
        role=role,
        content=content,
        content_enhanced_with_rag=content_enhanced_with_rag,
    )


def make_compaction(up_to_message_id: int, summary: str = "The user asked about pensions.") -> ChatCompaction:
    return ChatCompaction(id=1, chat_id=1, up_to_message_id=up_to_message_id, summary=summary)


def test_formats_alternating_messages():
    messages = [
        make_message(1, "user", "first question"),
        make_message(2, "assistant", "first answer"),
        make_message(3, "user", "second question"),
    ]

    result = prepare_message_objects_for_llm(messages)

    assert result == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]


def test_merges_consecutive_user_messages():
    messages = [
        make_message(1, "user", "part one"),
        make_message(2, "user", "part two"),
    ]

    result = prepare_message_objects_for_llm(messages)

    assert result == [{"role": "user", "content": "part one\n\npart two"}]


def test_prefers_rag_enhanced_content():
    messages = [make_message(1, "user", "plain", content_enhanced_with_rag="enhanced with sources")]

    result = prepare_message_objects_for_llm(messages)

    assert result == [{"role": "user", "content": "enhanced with sources"}]


def test_legacy_summary_takes_priority_over_rag_enhanced_content():
    message = make_message(1, "user", "plain", content_enhanced_with_rag="enhanced with sources")
    message.summary = "a lossy summary"

    result = prepare_message_objects_for_llm([message])

    assert result == [{"role": "user", "content": "a lossy summary"}]


def test_legacy_per_message_summary_is_used_when_populated():
    """Historical rows compacted under the old per-message scheme still carry a summary.

    Those chats have no ChatCompaction row of their own (that table postdates them), so
    without this fallback they'd send full, uncompacted history instead of the old summary.
    """
    message = make_message(1, "user", "the full original content")
    message.summary = "a lossy summary"

    result = prepare_message_objects_for_llm([message])

    assert result == [{"role": "user", "content": "a lossy summary"}]


def test_new_messages_never_populate_summary_so_use_full_content():
    message = make_message(1, "user", "the full original content")

    result = prepare_message_objects_for_llm([message])

    assert result == [{"role": "user", "content": "the full original content"}]


# --- compaction summary substitution ------------------------------------------------------


def test_compaction_replaces_covered_messages_with_summary():
    messages = [
        make_message(1, "user", "ancient question"),
        make_message(2, "assistant", "ancient answer"),
        make_message(3, "user", "recent question"),
        make_message(4, "assistant", "recent answer"),
    ]
    compaction = make_compaction(up_to_message_id=2, summary="Earlier: the user asked something ancient.")

    result = prepare_message_objects_for_llm(messages, compaction=compaction)

    # Summary first, then only the messages after the cut point. The summary is a user
    # message, so the user message that follows it merges into the same entry.
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert "Earlier: the user asked something ancient." in result[0]["content"]
    assert "<conversation-summary>" in result[0]["content"]
    assert result[0]["content"].endswith("recent question")
    assert result[1] == {"role": "assistant", "content": "recent answer"}
    # None of the replaced content survives.
    assert not any("ancient answer" in str(msg["content"]) for msg in result)


def test_compaction_summary_merges_with_a_following_user_message():
    """The summary is a user message, so a user message immediately after it must merge.

    Two consecutive user entries would be an invalid request.
    """
    messages = [
        make_message(1, "user", "old"),
        make_message(2, "user", "new question"),
    ]
    compaction = make_compaction(up_to_message_id=1)

    result = prepare_message_objects_for_llm(messages, compaction=compaction)

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert "new question" in result[0]["content"]


def test_compaction_keeps_roles_alternating():
    messages = [make_message(i, "user" if i % 2 else "assistant", f"msg {i}") for i in range(1, 7)]
    compaction = make_compaction(up_to_message_id=3)

    result = prepare_message_objects_for_llm(messages, compaction=compaction)

    roles = [msg["role"] for msg in result]
    assert all(first != second for first, second in zip(roles, roles[1:], strict=False)), roles


def test_compaction_ignored_when_read_path_disabled():
    """The rollback switch: summaries keep being written but stop being used."""
    messages = [
        make_message(1, "user", "ancient question"),
        make_message(2, "assistant", "recent answer"),
    ]
    compaction = make_compaction(up_to_message_id=1)

    with patch.object(settings, "compaction_use_conversation_summary", False):
        result = prepare_message_objects_for_llm(messages, compaction=compaction)

    assert result == [
        {"role": "user", "content": "ancient question"},
        {"role": "assistant", "content": "recent answer"},
    ]


def test_unsaved_messages_are_never_dropped_by_the_cut_point():
    message_without_id = make_message(None, "user", "not yet persisted")
    compaction = make_compaction(up_to_message_id=99)

    result = prepare_message_objects_for_llm([message_without_id], compaction=compaction)

    assert any("not yet persisted" in str(msg["content"]) for msg in result)
