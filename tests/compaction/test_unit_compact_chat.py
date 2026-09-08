"""Tests for the compaction call itself.

The point of these is the cache-hit invariant. Compaction is only cheap because it resends
the prefix the chat call just cached; if the system blocks or the leading messages differ by
so much as a byte, the whole history is re-read at full price and compaction costs more than
the per-message approach it replaced. That failure is invisible at runtime — nothing errors,
the bill just goes up — so it is pinned down here.
"""

import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.compaction.prompts import CONVERSATION_COMPACTION_INSTRUCTION
from app.compaction.service import compact_chat
from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_BLOCKS = [
    {"type": "text", "text": "static system prompt", "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": "dynamic resources block", "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": "session block"},
]

# The prefix through the last assistant message, exactly as the chat call built it — the
# current turn's own prompt is never part of what compact_chat receives.
FORMATTED_MESSAGES = [
    {"role": "user", "content": "first question"},
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "first answer", "cache_control": {"type": "ephemeral"}}],
    },
]


def make_response(summary: str = "A summary of the conversation."):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=summary)],
        usage=SimpleNamespace(output_tokens=120, cache_read_input_tokens=149_000, cache_creation_input_tokens=0),
        llm_internal_response_id=77,
    )


@asynccontextmanager
async def fake_session(session):
    yield session


def _acquire_result(acquired: bool):
    result = MagicMock()
    result.first.return_value = MagicMock() if acquired else None
    return result


def _get_latest_result(previous_compaction=None):
    result = MagicMock()
    result.scalars.return_value.first.return_value = previous_compaction
    return result


def make_session(previous_compaction=None, lock_acquired=True):
    """Models the exact sequence of execute() calls a successful run makes: acquire the
    lock, read the latest compaction, then (in `finally`) release the lock. If the lock is
    not acquired, only the first of those happens.
    """
    session = MagicMock()
    side_effect = [_acquire_result(lock_acquired)]
    if lock_acquired:
        side_effect += [_get_latest_result(previous_compaction), MagicMock()]
    session.execute = AsyncMock(side_effect=side_effect)
    session.commit = AsyncMock()
    session.add = MagicMock()
    return session


async def run_compaction(invoke_async, session=None, up_to_message_id=2, extra_api_kwargs=None):
    session = session or make_session()
    handler = MagicMock()
    handler.invoke_async = invoke_async

    with patch("app.compaction.service.async_db_session", lambda: fake_session(session)):
        with patch("app.compaction.service.BedrockHandler", return_value=handler):
            compaction = await compact_chat(
                chat_id=1,
                llm_obj=SimpleNamespace(id=3, model="anthropic.claude-sonnet-5", max_tokens=8192),
                system=SYSTEM_BLOCKS,
                formatted_messages=FORMATTED_MESSAGES,
                up_to_message_id=up_to_message_id,
                prefix_tokens=150_000,
                extra_api_kwargs=extra_api_kwargs,
            )
    return compaction, session


@pytest.mark.asyncio
async def test_sends_the_chat_prefix_unchanged():
    """The leading messages must be byte-identical to what the chat call sent."""
    invoke_async = AsyncMock(return_value=make_response())

    await run_compaction(invoke_async)

    sent_messages = invoke_async.call_args.args[0]
    assert sent_messages[: len(FORMATTED_MESSAGES)] == FORMATTED_MESSAGES


@pytest.mark.asyncio
async def test_appends_only_the_instruction_as_a_final_user_turn():
    """Compaction is decoupled from the current turn now — no reply is appended, because the
    current turn's prompt and reply are never part of what is being summarised."""
    invoke_async = AsyncMock(return_value=make_response())

    await run_compaction(invoke_async)

    sent_messages = invoke_async.call_args.args[0]
    assert len(sent_messages) == len(FORMATTED_MESSAGES) + 1
    assert sent_messages[:-1] == FORMATTED_MESSAGES
    assert sent_messages[-1] == {"role": "user", "content": CONVERSATION_COMPACTION_INSTRUCTION}


@pytest.mark.asyncio
async def test_does_not_override_the_system_prompt():
    """Overriding `system` would discard the cached system blocks and the messages behind them.

    The summarisation instruction has to travel in the appended user turn instead.
    """
    invoke_async = AsyncMock(return_value=make_response())

    await run_compaction(invoke_async)

    assert invoke_async.call_args.kwargs["system"] is SYSTEM_BLOCKS


@pytest.mark.asyncio
async def test_passes_thinking_kwargs_through_so_the_request_shape_matches():
    invoke_async = AsyncMock(return_value=make_response())

    await run_compaction(invoke_async, extra_api_kwargs={"thinking": {"type": "disabled"}})

    assert invoke_async.call_args.kwargs["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_caps_the_summary_length():
    invoke_async = AsyncMock(return_value=make_response())

    with patch.object(settings, "compaction_max_summary_tokens", 1234):
        await run_compaction(invoke_async)

    assert invoke_async.call_args.kwargs["max_tokens"] == 1234


@pytest.mark.asyncio
async def test_writes_the_compaction_row():
    invoke_async = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Earlier the user asked about pensions.")],
            usage=SimpleNamespace(output_tokens=120, cache_read_input_tokens=149_000, cache_creation_input_tokens=900),
            llm_internal_response_id=77,
        )
    )

    compaction, session = await run_compaction(invoke_async, up_to_message_id=42)

    assert compaction is not None
    assert compaction.chat_id == 1
    assert compaction.up_to_message_id == 42
    assert compaction.summary == "Earlier the user asked about pensions."
    assert compaction.prefix_tokens_at_compaction == 150_000
    assert compaction.summary_tokens == 120
    assert compaction.summary_cache_read_tokens == 149_000
    assert compaction.summary_cache_write_tokens == 900
    assert compaction.llm_internal_response_id == 77
    session.add.assert_called_once_with(compaction)
    # acquire the lock, read the latest compaction, release the lock. Commits are handled by
    # the real async_db_session()'s transaction block, not by application code, so nothing
    # here calls session.commit() directly.
    assert session.execute.await_count == 3


@pytest.mark.asyncio
async def test_skips_when_the_lock_is_already_held():
    """Another task is already compacting this chat — no LLM call, no second row."""
    invoke_async = AsyncMock(return_value=make_response())
    session = make_session(lock_acquired=False)

    compaction, session = await run_compaction(invoke_async, session=session)

    assert compaction is None
    invoke_async.assert_not_awaited()
    session.add.assert_not_called()
    # Only the failed acquire attempt — nothing to release, since nothing was acquired.
    assert session.execute.await_count == 1


@pytest.mark.asyncio
async def test_skips_when_the_previous_compaction_already_covers_the_cut_point():
    """Idempotence: no LLM call, no second row, no double-counted compaction."""
    invoke_async = AsyncMock(return_value=make_response())
    session = make_session(previous_compaction=SimpleNamespace(up_to_message_id=50))

    compaction, session = await run_compaction(invoke_async, session=session, up_to_message_id=42)

    assert compaction is None
    invoke_async.assert_not_awaited()
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_discards_an_empty_summary():
    invoke_async = AsyncMock(return_value=make_response("   "))

    compaction, session = await run_compaction(invoke_async)

    assert compaction is None
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_swallows_llm_failures():
    """Compaction is best effort — a failure must never surface to the user's turn."""
    invoke_async = AsyncMock(side_effect=RuntimeError("bedrock is down"))

    compaction, _ = await run_compaction(invoke_async)

    assert compaction is None


@pytest.mark.asyncio
async def test_releases_the_lock_even_on_llm_failure():
    invoke_async = AsyncMock(side_effect=RuntimeError("bedrock is down"))
    session = make_session()

    await run_compaction(invoke_async, session=session)

    # acquire, get_latest, release — the release still happens despite the LLM error.
    assert session.execute.await_count == 3
