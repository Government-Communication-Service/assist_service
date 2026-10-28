import asyncio
import logging
from dataclasses import dataclass, replace

from anthropic.types import Message as AnthropicMessage
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.bedrock.service import calculate_completion_cost, extract_cache_tokens
from app.database.db_operations import DbOperations
from app.database.models import LLM, Message
from app.database.table import async_db_session

logger = logging.getLogger(__name__)

LOG_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class LlmCallContext:
    """What a Bedrock call was for, recorded on its llm_internal_response row.

    Args:
        purpose: Short label for the kind of call, e.g. "chat_loop" or "title".
        chat_id: The chat the call served, if the call site has it.
        message_id: The user message of the turn the call served, if the call site has it. The chat is
            looked up from it when chat_id is not given.
    """

    purpose: str
    chat_id: int | None = None
    message_id: int | None = None


async def log_llm_call(llm: LLM, response: AnthropicMessage, context: LlmCallContext) -> int | None:
    """Record one Bedrock call in llm_internal_response, without ever raising.

    The row is written on its own session and committed immediately, so a logging failure
    cannot fail the model call or leave a caller's session in a failed transaction, and a
    caller's later rollback cannot lose the record of a call that was billed.

    Args:
        llm: The model that was called.
        response: The complete response.
        context: What the call was for.

    Returns:
        The new row's id, or None if it couldn't be written (already logged).
    """
    try:
        return await asyncio.wait_for(_write(llm, response, context), timeout=LOG_TIMEOUT_SECONDS)
    except Exception:
        logger.exception("Failed to log LLM call to llm_internal_response")
        return None


async def _write(llm: LLM, response: AnthropicMessage, context: LlmCallContext) -> int:
    try:
        return await _insert(llm, response, context)
    except IntegrityError:
        if context.chat_id is None and context.message_id is None:
            raise
        logger.warning("llm_internal_response insert rejected the chat/message ids; retrying without them")
        return await _insert(llm, response, replace(context, chat_id=None, message_id=None))


async def _insert(llm: LLM, response: AnthropicMessage, context: LlmCallContext) -> int:
    cache_read_tokens, cache_write_tokens = extract_cache_tokens(response.usage)
    chat_id = context.chat_id
    message_id = context.message_id
    async with async_db_session() as db_session:
        if chat_id is None and message_id is not None:
            chat_id = await db_session.scalar(select(Message.chat_id).where(Message.id == message_id))
        row = await DbOperations.insert_llm_internal_response_id_query(
            db_session=db_session,
            web_browsing_llm=llm,
            content=str(response.content),
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            completion_cost=calculate_completion_cost(
                llm,
                response.usage.input_tokens,
                response.usage.output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
            ),
            purpose=context.purpose,
            chat_id=chat_id,
            message_id=message_id,
        )
        return row.id
