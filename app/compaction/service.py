# ruff: noqa: E501
"""Chat compaction service for summarising messages to handle token limits"""

import asyncio
import logging
from typing import Optional, Tuple

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.bedrock import BedrockHandler, RunMode
from app.bedrock.service import calculate_completion_cost
from app.compaction import config as compaction_config
from app.database.models import LLM, LlmInternalResponse, Message
from app.database.table import LLMTable

logger = logging.getLogger(__name__)


async def save_llm_internal_response(
    db_session: AsyncSession,
    llm: LLM,
    content: str,
    input_tokens: int,
    output_tokens: int,
    system_prompt_id: Optional[int] = None,
) -> LlmInternalResponse:
    """Save LLM response and usage to database for analytics."""
    stmt = (
        insert(LlmInternalResponse)
        .values(
            llm_id=llm.id,
            system_prompt_id=system_prompt_id,
            content=content,
            tokens_in=input_tokens,
            tokens_out=output_tokens,
            completion_cost=calculate_completion_cost(llm, input_tokens, output_tokens),
        )
        .returning(LlmInternalResponse)
    )
    result = await db_session.execute(stmt)
    return result.scalar_one()


async def summarise_message(message: Message, db_session: AsyncSession) -> Optional[LlmInternalResponse]:
    """
    Summarise a single message using the configured LLM model.

    Args:
        message: The message to summarise
        db_session: Database session

    Returns:
        LlmInternalResponse record containing the summary, or None if already summarised
    """
    # Skip if message already has a summary
    if message.summary is not None:
        logger.debug(f"Message {message.id} already has a summary, skipping")
        return None

    try:
        # Get the summarisation LLM model
        llm_table = LLMTable()
        llm = llm_table.get_by_model(compaction_config.LLM_COMPACTION_SUMMARISATION_MODEL)

        # Create Bedrock handler for summarisation
        bedrock_handler = BedrockHandler(llm=llm, mode=RunMode.ASYNC)

        # Prepare the message content for summarisation
        content_to_summarise = (
            message.content_enhanced_with_rag if message.content_enhanced_with_rag else message.content
        )

        # Create messages for the LLM with pseudo-XML tags
        messages = [
            {
                "role": "user",
                "content": f"Please summarise this {message.role} message:\n\n<message-content>\n{content_to_summarise}\n</message-content>",
            }
        ]

        # Call the LLM to generate summary
        response = await bedrock_handler.invoke_async(
            max_tokens=llm.max_tokens,
            model=f"us.{llm.model}",
            system=compaction_config.SUMMARISATION_SYSTEM_PROMPT,
            messages=messages,
        )

        # Extract the summary from the response
        summary_content = response.content[0].text if response.content else ""

        # Save the LLM response to track costs
        llm_response = await save_llm_internal_response(
            db_session=db_session,
            llm=llm,
            content=summary_content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        # Update the message with the summary and LLM response ID
        stmt = (
            update(Message)
            .where(Message.id == message.id)
            .values(
                summary=summary_content,
                summary_llm_response_id=llm_response.id,
            )
        )
        await db_session.execute(stmt)

        logger.info(f"Successfully summarised message {message.id}")
        return llm_response

    except Exception as e:
        logger.exception(f"Error summarising message {message.id}: {e}")
        return None


def estimate_message_tokens(content: str) -> int:
    """
    Estimate token count using rule of thumb: 3.5 letters per token.

    Args:
        content: The message content to estimate tokens for

    Returns:
        Estimated token count
    """
    if not content:
        return 0
    return int(len(content) / 3.5)


async def calculate_chat_token_count_with_current_message(
    chat_id: int, current_message_content: str, db_session: AsyncSession
) -> int:
    """
    Calculate the total token count for all messages in a chat, including the current message.

    Args:
        chat_id: The chat ID to calculate tokens for
        current_message_content: The content of the current message being processed (with RAG content)
        db_session: Database session

    Returns:
        Total token count for the chat including the current message
    """
    try:
        # Get all existing messages for the chat, ordered by creation date
        stmt = (
            select(Message)
            .where(Message.chat_id == chat_id)
            .where(Message.deleted_at.is_(None))
            .order_by(Message.created_at)
        )
        result = await db_session.execute(stmt)
        messages = result.scalars().all()

        # Sum up all the existing token counts
        # If a message has a summary, use that for token estimation instead of the stored tokens value
        existing_tokens = 0
        for message in messages:
            if message.summary is not None:
                existing_tokens += estimate_message_tokens(message.summary)
            elif message.tokens:
                existing_tokens += message.tokens

        # Estimate tokens for the current message with RAG content
        current_message_tokens = estimate_message_tokens(current_message_content)

        total_tokens = existing_tokens + current_message_tokens

        logger.debug(
            f"Chat {chat_id} has {existing_tokens} existing tokens + {current_message_tokens} current message tokens = {total_tokens} total"
        )
        return total_tokens

    except Exception as e:
        logger.exception(f"Error calculating token count for chat {chat_id}: {e}")
        return 0


async def should_trigger_compaction(chat_id: int, current_message_content: str, db_session: AsyncSession) -> bool:
    """
    Determine if compaction should be triggered for a chat based on token threshold.

    Args:
        chat_id: The chat ID to check
        current_message_content: The content of the current message being processed (with RAG content)
        db_session: Database session

    Returns:
        True if compaction should be triggered, False otherwise
    """
    total_tokens = await calculate_chat_token_count_with_current_message(chat_id, current_message_content, db_session)
    should_compact = total_tokens >= compaction_config.COMPACTION_TOKEN_THRESHOLD

    if should_compact:
        logger.info(
            f"Chat {chat_id} has {total_tokens} tokens, triggering compaction (threshold: {compaction_config.COMPACTION_TOKEN_THRESHOLD})"
        )
    else:
        logger.debug(
            f"Chat {chat_id} has {total_tokens} tokens, no compaction triggered (threshold: {compaction_config.COMPACTION_TOKEN_THRESHOLD})"
        )
    return should_compact


async def compact_chat_messages(chat_id: int, db_session: AsyncSession) -> int:
    """
    Compact all unsummarised messages in a chat by creating summaries.

    Args:
        chat_id: The chat ID to compact
        db_session: Database session

    Returns:
        Number of messages that were successfully summarised
    """
    try:
        # Get all messages that don't have summaries yet
        stmt = (
            select(Message)
            .where(Message.chat_id == chat_id)
            .where(Message.deleted_at.is_(None))
            .where(Message.summary.is_(None))
            .order_by(Message.created_at)
        )
        result = await db_session.execute(stmt)
        messages_to_summarise = result.scalars().all()

        # Summarise all messages in parallel
        summarisation_tasks = [summarise_message(message, db_session) for message in messages_to_summarise]

        llm_responses = await asyncio.gather(*summarisation_tasks, return_exceptions=True)

        # Count successful summarisations
        summarised_count = sum(
            1 for response in llm_responses if response is not None and not isinstance(response, Exception)
        )

        # Commit the changes
        await db_session.commit()

        logger.info(f"Compaction completed for chat {chat_id}: {summarised_count} messages summarised")
        return summarised_count

    except Exception as e:
        logger.exception(f"Error during compaction for chat {chat_id}: {e}")
        await db_session.rollback()
        return 0


async def perform_chat_compaction(
    chat_id: int, current_message_content: str, db_session: AsyncSession
) -> Tuple[bool, int]:
    """
    Perform chat compaction by summarising messages.

    Args:
        chat_id: The chat ID to compact
        current_message_content: The content of the current message being processed (with RAG content)
        db_session: Database session

    Returns:
        Tuple of (compaction_performed, messages_summarised)
    """
    try:
        if not await should_trigger_compaction(chat_id, current_message_content, db_session):
            return False, 0

        # Summarise all unsummarised messages
        summarised_count = await compact_chat_messages(chat_id, db_session)

        return True, summarised_count

    except Exception as e:
        logger.exception(f"Error in perform_chat_compaction for chat {chat_id}: {e}")
        return False, 0


async def trigger_compaction_if_needed(chat_id: int, current_message_content: str, db_session: AsyncSession) -> bool:
    """
    Check if compaction is needed and trigger it if necessary.
    This is the main entry point for compaction logic.

    Args:
        chat_id: The chat ID to check and potentially compact
        current_message_content: The content of the current message being processed (with RAG content)
        db_session: Database session

    Returns:
        True if compaction was triggered and completed, False otherwise
    """
    try:
        compaction_performed, summarised_count = await perform_chat_compaction(
            chat_id, current_message_content, db_session
        )
        return compaction_performed

    except Exception as e:
        logger.exception(f"Error in trigger_compaction_if_needed for chat {chat_id}: {e}")
        return False
