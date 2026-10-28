"""Chat compaction: replace a long history with a single summary.

Compaction always covers everything up to and including the last assistant message before
the prompt that triggered it. The compaction call works with caching to reduce costs.

The compaction call uses a special prompt at the end of the chat rather than a system prompt
to ensure that we can reuse the cached prefix (system + messages).

Compaction occurs in the background. This works with the compaction lock (a db table) to avoid
redundant compaction calls if the chat moves on while compaction is underway. Otherwise a quickfire
round of prompts and responses could trigger several compaction calls, as the compaction
only comes into effect when the summary is written to the database.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.bedrock import BedrockHandler, RunMode
from app.bedrock.call_log import LlmCallContext
from app.compaction.prompts import (
    CONVERSATION_COMPACTION_INSTRUCTION,
    CONVERSATION_COMPACTION_SHORTER_INSTRUCTION,
)
from app.config import settings
from app.database.models import LLM, ChatCompaction, ChatCompactionLock, Message
from app.database.table import async_db_session

logger = logging.getLogger(__name__)

# Strong references to in-flight background tasks. asyncio only holds a weak reference to a
# running task, so without this a compaction can be garbage-collected mid-call.
_background_tasks: set[asyncio.Task] = set()


def _content_chars(content: str | list[dict]) -> int:
    """Count the characters in message content, whether a string or a list of content blocks."""
    if isinstance(content, str):
        return len(content)
    total = 0
    for block in content:
        block_type = block.get("type")
        if block_type == "text":
            total += len(block.get("text", ""))
        elif block_type == "tool_use":
            total += len(block.get("name", "")) + len(json.dumps(block.get("input", {})))
        elif block_type == "tool_result":
            total += _content_chars(block.get("content") or "")
        elif block_type == "thinking":
            total += len(block.get("thinking", ""))
        else:
            total += len(json.dumps(block))
    return total


def estimate_message_tokens(content: str | list[dict] | None) -> int:
    """Estimate token count using rule of thumb: 3.5 letters per token.

    Args:
        content: Message content: a string, or a list of content blocks (text, tool_use,
            tool_result, thinking; others are counted by their JSON length).

    Returns:
        Estimated token count
    """
    if not content:
        return 0
    return int(_content_chars(content) / 3.5)


def estimate_prefix_tokens(formatted_messages: list[dict]) -> int:
    """Estimate the size of an assembled message list.

    Used to decide whether to compact *before* the LLM call happens, so there is no real
    usage number yet.
    """
    return sum(estimate_message_tokens(msg["content"]) for msg in formatted_messages)


def find_last_assistant_message_id(messages: list[Message]) -> Optional[int]:
    """Find the cut point: the last assistant-role message in the given (oldest-first) list.

    This is always the true end of the last *complete* turn: a turn only ends once a final,
    user-visible assistant reply exists, so this lands on the right boundary even once a
    turn can contain several rows (eg a tool-call loop).
    """
    for msg in reversed(messages):
        if msg.role == "assistant":
            return msg.id
    return None


def messages_since_compaction(messages: list[Message], compaction: Optional[ChatCompaction]) -> int:
    """Count messages not covered by the given compaction (or all of them, if there is none)."""
    if compaction is None:
        return len(messages)
    return sum(1 for msg in messages if msg.id > compaction.up_to_message_id)


def should_compact(estimated_tokens: int, messages_since_last_compaction: int) -> bool:
    """Decide whether this turn's estimated prefix size warrants compacting the chat.

    An estimate is used rather than measured token counts because there may be a new
    compaction summary in the database that will significantly reduce the input tokens
    for the next LLM request, compared to those measured in the last request.
    """
    if not settings.compaction_enabled:
        return False

    if messages_since_last_compaction < settings.compaction_min_messages:
        logger.debug(
            f"{messages_since_last_compaction} messages since the last compaction is below the "
            f"minimum of {settings.compaction_min_messages}, not compacting"
        )
        return False

    if estimated_tokens < settings.compaction_token_threshold:
        logger.debug(
            f"Estimated prefix of {estimated_tokens} tokens is below the compaction threshold "
            f"({settings.compaction_token_threshold}), not compacting"
        )
        return False

    return True


async def get_latest_compaction(chat_id: int, db_session: AsyncSession) -> Optional[ChatCompaction]:
    """Return the most recent compaction for a chat, or None if it has never been compacted."""
    stmt = (
        select(ChatCompaction)
        .where(ChatCompaction.chat_id == chat_id)
        .where(ChatCompaction.deleted_at.is_(None))
        .order_by(ChatCompaction.created_at.desc(), ChatCompaction.id.desc())
        .limit(1)
    )
    result = await db_session.execute(stmt)
    return result.scalars().first()


async def is_compaction_locked(chat_id: int, db_session: AsyncSession) -> bool:
    """Read-only check of whether a chat's compaction lock is currently held.

    Does not claim the lock; not a substitute for `acquire_compaction_lock`. A held lock
    older than `compaction_lock_stale_after_minutes` is treated as free.
    """
    stale_cutoff = datetime.now() - timedelta(minutes=settings.compaction_lock_stale_after_minutes)
    stmt = select(ChatCompactionLock.id).where(
        ChatCompactionLock.chat_id == chat_id,
        ChatCompactionLock.compaction_lock.is_(True),
        ChatCompactionLock.locked_at >= stale_cutoff,
    )
    result = await db_session.execute(stmt)
    return result.first() is not None


async def acquire_compaction_lock(chat_id: int) -> bool:
    """Atomically claim the compaction lock for a chat, across all workers.

    Opens and commits its own short-lived session. No session is held open during the
    (potentially long) LLM call.

    A held lock older than `compaction_lock_stale_after_minutes` is treated as free.
    """
    now = datetime.now()
    stale_cutoff = now - timedelta(minutes=settings.compaction_lock_stale_after_minutes)
    stmt = (
        pg_insert(ChatCompactionLock)
        .values(chat_id=chat_id, compaction_lock=True, locked_at=now)
        .on_conflict_do_update(
            constraint="uq_chat_compaction_lock_chat_id",
            set_={"compaction_lock": True, "locked_at": now},
            where=(ChatCompactionLock.compaction_lock.is_(False)) | (ChatCompactionLock.locked_at < stale_cutoff),
        )
        .returning(ChatCompactionLock.id)
    )
    async with async_db_session() as session:
        result = await session.execute(stmt)
        return result.first() is not None


async def release_compaction_lock(chat_id: int) -> None:
    """Free the compaction lock, in its own short-lived session (see `acquire_compaction_lock`)."""
    try:
        async with async_db_session() as session:
            await session.execute(
                update(ChatCompactionLock).where(ChatCompactionLock.chat_id == chat_id).values(compaction_lock=False)
            )
    except Exception:
        logger.exception(f"Failed to release the compaction lock for chat {chat_id}")


async def _invoke_compaction(
    bedrock_handler: BedrockHandler,
    formatted_messages: list[dict],
    instruction: str,
    chat_id: int,
    system: str | list | None,
    extra_api_kwargs: Optional[dict],
):
    """Send the chat prefix plus the compaction instruction as the final user turn."""
    messages = [*formatted_messages, {"role": "user", "content": instruction}]
    return await bedrock_handler.invoke_async(
        messages,
        context=LlmCallContext(purpose="compaction", chat_id=chat_id),
        system=system,
        max_tokens=settings.compaction_max_summary_tokens,
        **(extra_api_kwargs or {}),
    )


async def compact_chat(
    chat_id: int,
    llm_obj: LLM,
    system: str | list | None,
    formatted_messages: list[dict],
    up_to_message_id: int,
    prefix_tokens: int,
    extra_api_kwargs: Optional[dict] = None,
) -> Optional[ChatCompaction]:
    """Summarise a chat's history into a single ChatCompaction row.

    Runs in the background to avoid blocking the conversation. Includes lock acquisition
    and release.

    Args:
        chat_id: Chat to compact.
        llm_obj: The LLM the chat call used — the same model must be used, or the cache misses.
        system: The exact system blocks the chat call sent.
        formatted_messages: The messages before the current turn, with the cache breakpoint on
            the last assistant message as in the chat call. It may end with a tool_result user
            message, which the API merges with the compaction instruction. The prefix through
            the breakpoint is the chat call's verbatim, which is what makes this a cache read
            instead of a full re-read.
        up_to_message_id: Cut point — the last assistant message's id, the same one the chat
            call's cache breakpoint sits on.
        prefix_tokens: Estimated prefix size that triggered this, recorded for ROI analysis.
        extra_api_kwargs: Thinking/output config from the chat call, passed through so the
            request shape matches and the message cache still applies.

    Returns:
        The created ChatCompaction, or None if compaction was skipped or failed.
    """
    acquired = False
    try:
        acquired = await acquire_compaction_lock(chat_id)
        if not acquired:
            logger.info(f"Chat {chat_id} is already being compacted, skipping")
            return None

        # Three short steps so no transaction is held open across the (long) LLM call:
        # read in one session, call the LLM with no session open, write in a new session.
        async with async_db_session() as db_session:
            previous = await get_latest_compaction(chat_id, db_session)
        if previous is not None and previous.up_to_message_id >= up_to_message_id:
            logger.warning(
                f"Chat {chat_id} is already compacted up to message {previous.up_to_message_id}, "
                f"which covers the proposed cut point {up_to_message_id}; skipping"
            )
            return None

        bedrock_handler = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)
        response = await _invoke_compaction(
            bedrock_handler, formatted_messages, CONVERSATION_COMPACTION_INSTRUCTION, chat_id, system, extra_api_kwargs
        )

        if response.stop_reason == "max_tokens":
            logger.warning(
                f"Compaction for chat {chat_id} hit max_tokens, retrying with a request for a shorter summary"
            )
            shorter_instruction = (
                CONVERSATION_COMPACTION_INSTRUCTION
                + CONVERSATION_COMPACTION_SHORTER_INSTRUCTION.format(max_tokens=settings.compaction_max_summary_tokens)
            )
            response = await _invoke_compaction(
                bedrock_handler, formatted_messages, shorter_instruction, chat_id, system, extra_api_kwargs
            )

        summary = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        # Tools are still offered to this call (changing tool_choice would invalidate the message
        # cache), so the model can reply with a preamble plus a tool_use, or run out of tokens.
        # Either fragment would permanently replace the history, so only a clean end_turn is kept.
        if response.stop_reason != "end_turn" or not summary.strip():
            logger.error(
                f"Compaction for chat {chat_id} did not produce a usable summary "
                f"(stop_reason={response.stop_reason!r}, summary length {len(summary)} chars), discarding"
            )
            return None

        compaction = ChatCompaction(
            chat_id=chat_id,
            up_to_message_id=up_to_message_id,
            summary=summary,
            prefix_tokens_at_compaction=prefix_tokens,
            summary_tokens=response.usage.output_tokens,
            summary_cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0),
            summary_cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0),
            llm_internal_response_id=response.llm_internal_response_id,
        )
        async with async_db_session() as db_session:
            db_session.add(compaction)
            # Committed automatically when this block exits cleanly — no explicit commit here.

        logger.info(
            f"Compacted chat {chat_id} up to message {up_to_message_id}: "
            f"{prefix_tokens} estimated prefix tokens summarised into {response.usage.output_tokens} tokens "
            f"(cache read {getattr(response.usage, 'cache_read_input_tokens', 0)})"
        )
        return compaction

    except Exception as e:
        logger.exception(f"Error compacting chat {chat_id}: {e}")
        return None
    finally:
        if acquired:
            await release_compaction_lock(chat_id)


def schedule_compaction(
    chat_id: int,
    llm_obj: LLM,
    system: str | list | None,
    formatted_messages: list[dict],
    up_to_message_id: int,
    prefix_tokens: int,
    extra_api_kwargs: Optional[dict] = None,
) -> asyncio.Task:
    """Kick off compaction in the background.

    This is designed to take as long as it takes, not blocking the conversation.
    Subsequent turns that happen before the summary lands in the database don't benefit from
    compaction, but the lock ensures that they don't fire redundant compaction requests.

    Losing a compaction to a worker restart is harmless — the next turn after the lock
    is released will retry.
    """
    logger.info(f"Chat {chat_id} reached {prefix_tokens} estimated prefix tokens, scheduling background compaction")
    task = asyncio.create_task(
        compact_chat(
            chat_id=chat_id,
            llm_obj=llm_obj,
            system=system,
            formatted_messages=formatted_messages,
            up_to_message_id=up_to_message_id,
            prefix_tokens=prefix_tokens,
            extra_api_kwargs=extra_api_kwargs,
        )
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task
