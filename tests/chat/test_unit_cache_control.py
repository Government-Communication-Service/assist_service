"""Unit tests for apply_compaction_aware_cache_control.

`should_compact=True` means a same-turn compaction call will reuse this cache using
`new_messages[:-1]`, so the breakpoint must land on the last assistant message, one turn
before the final user message. `should_compact=False` means there is no competing call, so
the breakpoint lands on the final message itself to cache as much of the prefix as possible.
"""

from unittest.mock import patch

from app.chat.cache_control import MIN_CACHEABLE_PREFIX_TOKENS, apply_compaction_aware_cache_control
from app.chat.utils import prepare_message_objects_for_llm
from app.config import CacheTtl, settings
from app.database.models import ChatCompaction, Message

# Long enough to clear the minimum cacheable prefix (1024 tokens at ~3.5 chars per token).
LONG_CONTENT = "This sentence exists purely to add length to the conversation. " * 100


def make_message(message_id: int, role: str, content: str) -> Message:
    return Message(id=message_id, role=role, content=content)


def make_compaction(up_to_message_id: int, summary: str = "The user asked about pensions.") -> ChatCompaction:
    return ChatCompaction(id=1, chat_id=1, up_to_message_id=up_to_message_id, summary=summary)


def test_marks_the_last_assistant_message_when_a_compaction_call_will_reuse_the_cache():
    """should_compact=True: the reply call's request ends in the newest user turn, but the
    compaction call fired moments later ends in the compaction instruction instead — the last
    assistant message is the only point both requests share."""
    messages = [
        make_message(1, "user", LONG_CONTENT),
        make_message(2, "assistant", LONG_CONTENT),
        make_message(3, "user", "the newest question"),
    ]

    result = apply_compaction_aware_cache_control(prepare_message_objects_for_llm(messages), should_compact=True)

    assert result[1]["content"] == [{"type": "text", "text": LONG_CONTENT, "cache_control": {"type": "ephemeral"}}]
    assert result[-1]["content"] == "the newest question"


def test_marks_the_final_message_when_no_compaction_call_is_competing():
    """should_compact=False: there's no compaction sub-call to align with this turn, so the
    breakpoint covers as much of the prefix as possible, including the newest user turn."""
    messages = [
        make_message(1, "user", LONG_CONTENT),
        make_message(2, "assistant", LONG_CONTENT),
        make_message(3, "user", "the newest question"),
    ]

    result = apply_compaction_aware_cache_control(prepare_message_objects_for_llm(messages), should_compact=False)

    assert result[1]["content"] == LONG_CONTENT
    assert result[-1]["content"] == [
        {"type": "text", "text": "the newest question", "cache_control": {"type": "ephemeral"}}
    ]


def test_skipped_when_compacting_and_no_assistant_message_exists():
    """The very first turn, or the turn right after a compaction with no reply since — there
    is no valid boundary to align with the compaction call, so nothing is cached."""
    messages = [make_message(1, "user", LONG_CONTENT)]

    result = apply_compaction_aware_cache_control(prepare_message_objects_for_llm(messages), should_compact=True)

    assert result == [{"role": "user", "content": LONG_CONTENT}]


def test_no_cache_control_when_disabled():
    messages = [make_message(1, "user", LONG_CONTENT), make_message(2, "assistant", LONG_CONTENT)]

    prepared = prepare_message_objects_for_llm(messages)
    with patch.object(settings, "message_cache_control_enabled", False):
        result = apply_compaction_aware_cache_control(prepared, should_compact=False)

    assert isinstance(result[-1]["content"], str)


def test_skipped_below_minimum_cacheable_prefix():
    """Below the model's minimum prefix nothing is cached, so the breakpoint is wasted."""
    messages = [make_message(1, "user", "tiny"), make_message(2, "assistant", "tiny reply")]

    result = apply_compaction_aware_cache_control(prepare_message_objects_for_llm(messages), should_compact=False)

    assert all(isinstance(msg["content"], str) for msg in result)


def test_applied_once_prefix_is_large_enough():
    short = "a" * (MIN_CACHEABLE_PREFIX_TOKENS * 4)

    prepared = prepare_message_objects_for_llm([make_message(1, "user", "q"), make_message(2, "assistant", short)])
    result = apply_compaction_aware_cache_control(prepared, should_compact=False)

    assert isinstance(result[-1]["content"], list)
    assert result[-1]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_one_hour_ttl_is_declared_explicitly():
    messages = [make_message(1, "user", LONG_CONTENT), make_message(2, "assistant", LONG_CONTENT)]

    prepared = prepare_message_objects_for_llm(messages)
    with patch.object(settings, "message_cache_ttl", CacheTtl.one_hour):
        result = apply_compaction_aware_cache_control(prepared, should_compact=False)

    assert result[-1]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_marks_a_real_reply_after_compaction_not_the_summary():
    """Once a compacted chat has had one real exchange since the cut point, that reply is the
    boundary — not the synthetic summary entry, even though the summary is also a message."""
    messages = [make_message(1, "user", LONG_CONTENT), make_message(2, "assistant", LONG_CONTENT)]
    compaction = make_compaction(up_to_message_id=0, summary="An earlier, now-summarised exchange.")

    prepared = prepare_message_objects_for_llm(messages, compaction=compaction)
    result = apply_compaction_aware_cache_control(prepared, should_compact=True)

    assert result[-1]["role"] == "assistant"
    assert isinstance(result[-1]["content"], list)
