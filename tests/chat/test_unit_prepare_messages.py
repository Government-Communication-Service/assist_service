"""Unit tests for prepare_message_objects_for_llm.

Covers the two things it now does beyond formatting: substituting a compaction summary for
the history it replaces, and placing the prompt-cache breakpoint on the end of the
conversation.
"""

import logging
from unittest.mock import patch

from app.chat.utils import MIN_CACHEABLE_PREFIX_TOKENS, apply_final_turn_cache_control, prepare_message_objects_for_llm
from app.config import CacheTtl, settings
from app.database.models import ChatCompaction, Message

logger = logging.getLogger(__name__)

# Long enough to clear the minimum cacheable prefix (1024 tokens at ~3.5 chars per token).
LONG_CONTENT = "This sentence exists purely to add length to the conversation. " * 100


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


def test_legacy_per_message_summary_is_ignored():
    """The old mechanism substituted msg.summary for content. It must no longer do so.

    Historical rows in prod still carry summaries from the per-message approach; those chats
    should now send their full content rather than the lossy per-message summaries.
    """
    message = make_message(1, "user", "the full original content")
    message.summary = "a lossy summary"

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


# --- cache breakpoint on the last assistant message ---------------------------------------
#
# apply_final_turn_cache_control is now a separate call, made by the caller only once it has
# decided the turn is over the compaction threshold (prepare_message_objects_for_llm no
# longer takes a cache_final_turn flag).


def test_cache_control_marks_the_last_assistant_message():
    """Not the literal last entry: the reply call's request ends in the newest user turn, but
    the compaction call fired moments later ends in the compaction instruction instead — the
    last assistant message is the only point both requests share."""
    messages = [
        make_message(1, "user", LONG_CONTENT),
        make_message(2, "assistant", LONG_CONTENT),
        make_message(3, "user", "the newest question"),
    ]

    result = apply_final_turn_cache_control(prepare_message_objects_for_llm(messages))

    assert result[1]["content"] == [{"type": "text", "text": LONG_CONTENT, "cache_control": {"type": "ephemeral"}}]
    assert result[-1]["content"] == "the newest question"


def test_cache_control_skipped_when_no_assistant_message_exists():
    """The very first turn, or the turn right after a compaction with no reply since — there
    is no valid boundary to align with a future compaction call, so nothing is cached."""
    messages = [make_message(1, "user", LONG_CONTENT)]

    result = apply_final_turn_cache_control(prepare_message_objects_for_llm(messages))

    assert result == [{"role": "user", "content": LONG_CONTENT}]


def test_cache_control_left_off_unless_called():
    """Callers that flatten the result to text, or append their own entries, simply never call
    apply_final_turn_cache_control."""
    messages = [make_message(1, "user", LONG_CONTENT), make_message(2, "assistant", LONG_CONTENT)]

    result = prepare_message_objects_for_llm(messages)

    assert all(isinstance(msg["content"], str) for msg in result)


def test_no_cache_control_when_disabled():
    messages = [make_message(1, "user", LONG_CONTENT), make_message(2, "assistant", LONG_CONTENT)]

    prepared = prepare_message_objects_for_llm(messages)
    with patch.object(settings, "message_cache_control_enabled", False):
        result = apply_final_turn_cache_control(prepared)

    assert isinstance(result[-1]["content"], str)


def test_cache_control_skipped_below_minimum_cacheable_prefix():
    """Below the model's minimum prefix nothing is cached, so the breakpoint is wasted."""
    messages = [make_message(1, "user", "tiny"), make_message(2, "assistant", "tiny reply")]

    result = apply_final_turn_cache_control(prepare_message_objects_for_llm(messages))

    assert all(isinstance(msg["content"], str) for msg in result)


def test_cache_control_applied_once_prefix_is_large_enough():
    short = "a" * (MIN_CACHEABLE_PREFIX_TOKENS * 4)

    prepared = prepare_message_objects_for_llm([make_message(1, "user", "q"), make_message(2, "assistant", short)])
    result = apply_final_turn_cache_control(prepared)

    assert isinstance(result[-1]["content"], list)
    assert result[-1]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_one_hour_ttl_is_declared_explicitly():
    messages = [make_message(1, "user", LONG_CONTENT), make_message(2, "assistant", LONG_CONTENT)]

    prepared = prepare_message_objects_for_llm(messages)
    with patch.object(settings, "message_cache_ttl", CacheTtl.one_hour):
        result = apply_final_turn_cache_control(prepared)

    assert result[-1]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_cache_control_marks_a_real_reply_after_compaction_not_the_summary():
    """Once a compacted chat has had one real exchange since the cut point, that reply is the
    boundary — not the synthetic summary entry, even though the summary is also a message."""
    messages = [make_message(1, "user", LONG_CONTENT), make_message(2, "assistant", LONG_CONTENT)]
    compaction = make_compaction(up_to_message_id=0, summary="An earlier, now-summarised exchange.")

    prepared = prepare_message_objects_for_llm(messages, compaction=compaction)
    result = apply_final_turn_cache_control(prepared)

    assert result[-1]["role"] == "assistant"
    assert isinstance(result[-1]["content"], list)
