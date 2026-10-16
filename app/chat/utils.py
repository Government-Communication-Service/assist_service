# ruff: noqa: B008
from logging import getLogger
from uuid import UUID

from fastapi import Body, Depends, HTTPException, Path, status

from app.auth.exceptions import UuidInvalidError, UuidMissingError
from app.auth.utils import verify_and_parse_uuid
from app.auth.verify_service import verify_and_get_user_from_header
from app.chat.constants import HELPER_CALL_TRUNCATION_MARKER, PRIVATE_SHARE_ACCESS_DENIED
from app.compaction.prompts import build_compaction_summary_message
from app.config import settings
from app.database.database_exception import DatabaseError, DatabaseExceptionErrorCode
from app.database.db_operations import DbOperations
from app.database.db_session import DbSession
from app.database.models import Chat, ChatCompaction, Message, User
from app.database.table import (
    ChatTable,
    UserTable,
)

logger = getLogger(__name__)


def chat_validator(chat_uuid: str = Path(..., description="Chat UUID"), user_uuid: str = Path(...)):
    try:
        chat_uuid = UUID(chat_uuid)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'id' parameter '{chat_uuid}' is not a valid UUID",
        ) from e

    chat = ChatTable().get_by_uuid(chat_uuid)

    if not chat:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No chat found with UUID '{chat_uuid}'",
        )

    user = UserTable().get_one_by("id", chat.user_id)
    if str(user.uuid) != user_uuid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Access denied to chat '{chat_uuid}'",
        )

    if chat.deleted_at is not None:
        logger.info("Attempt to access archived chat '%s'", chat_uuid)
        raise DatabaseError(
            code=DatabaseExceptionErrorCode.GET_BY_UUID_ERROR,
            message="Chat has been archived",
        )

    return chat


def shared_chat_validator(share_code: str = Path(..., description="Share code")):
    try:
        chat = ChatTable().get_one_by("share_code", share_code)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid share code",
        ) from e

    if not chat.share:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This chat is not shared",
        )

    return chat


async def verify_shared_chat_access(
    db_session: DbSession,
    chat: Chat = Depends(shared_chat_validator),
    user: User = Depends(verify_and_get_user_from_header),
) -> Chat:
    """
    Verifies that the requesting user may view a shared chat.

    Publicly shared chats are visible to any authenticated user (existing behaviour).
    Privately shared chats are only visible to the chat owner and to users the owner
    has added to the share.
    """
    if not chat.share_private:
        return chat

    if chat.user_id == user.id:
        return chat

    mapping = await DbOperations.get_chat_share_user_mapping(db_session, chat.id, user.id)
    if mapping is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error_code": PRIVATE_SHARE_ACCESS_DENIED,
                "message": "You do not have access to this shared chat",
            },
        )

    return chat


def verify_shared_user_uuid_from_body(shared_user_uuid: str = Body(..., embed=True)) -> UUID:
    """Validates the shared_user_uuid provided in the request body when adding a user to a private share."""
    try:
        return verify_and_parse_uuid(shared_user_uuid)
    except (UuidMissingError, UuidInvalidError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'shared_user_uuid' parameter '{shared_user_uuid}' is not a valid UUID",
        ) from e


def verify_shared_user_uuid_from_path(shared_user_uuid: str = Path(..., description="Shared user UUID")) -> UUID:
    """Validates the shared_user_uuid path parameter when removing a user from a private share."""
    try:
        return verify_and_parse_uuid(shared_user_uuid)
    except (UuidMissingError, UuidInvalidError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'shared_user_uuid' parameter '{shared_user_uuid}' is not a valid UUID",
        ) from e


def prepare_message_objects_for_llm(
    all_messages: list[Message],
    *,
    compaction: ChatCompaction | None = None,
) -> list[dict]:
    """Format message rows into the list the Anthropic API expects.

    Args:
        all_messages: The chat's messages, oldest first.
        compaction: If set, messages up to its cut point are replaced by its summary.
    """
    new_messages: list[dict] = []
    messages_to_send = all_messages

    if compaction is not None and settings.compaction_use_conversation_summary:
        # Everything at or before the cut point is represented by the summary. Ids increase
        # with creation time within a chat, so comparing them is enough to find the boundary.
        messages_to_send = [
            msg for msg in all_messages if getattr(msg, "id", None) is None or msg.id > compaction.up_to_message_id
        ]
        new_messages.append({"role": "user", "content": build_compaction_summary_message(compaction.summary)})
        logger.info(
            f"Using compaction {compaction.id} for chat {compaction.chat_id}: "
            f"{len(all_messages) - len(messages_to_send)} messages replaced by a summary, "
            f"{len(messages_to_send)} sent verbatim"
        )

    for msg in messages_to_send:
        # Assistant rows: the agent loop's turn, exactly as sent. User rows: this turn's context
        # block plus the query, exactly as sent (see Message.raw_messages).
        raw_messages = getattr(msg, "raw_messages", None)
        if raw_messages:
            for raw_message in raw_messages:
                _append_message(new_messages, raw_message["role"], raw_message["content"])
            continue

        # Determine content to use based on priority:
        # 1. Legacy per-message summary, if present — historical rows compacted under the old
        #    per-message scheme, before ChatCompaction existed. New messages never populate this.
        # 2. If RAG-enhanced content exists, use it
        # 3. Otherwise, use original content
        if hasattr(msg, "summary") and msg.summary is not None:
            content_to_use = msg.summary
        elif hasattr(msg, "content_enhanced_with_rag") and msg.content_enhanced_with_rag is not None:
            content_to_use = msg.content_enhanced_with_rag
        else:
            content_to_use = msg.content

        if msg.role == "user":
            _append_message(new_messages, "user", content_to_use)
        elif msg.content:
            # Assistant rows without content (e.g. interrupted before any text) are skipped.
            new_messages.append({"role": "assistant", "content": content_to_use})

    logger.debug("Messages formatted for submission to LLM for final response")
    return new_messages


def _append_message(new_messages: list[dict], role: str, content: str | list[dict]) -> None:
    """Append a message, merging it into the previous one if both are user messages.

    Args:
        new_messages: Messages built so far; modified in place. Existing content is replaced,
            never mutated, so stored rows' content is left untouched.
        role: "user" or "assistant".
        content: A string or a list of content blocks.
    """
    last_msg = new_messages[-1] if new_messages else None
    if role == "user" and last_msg and last_msg["role"] == "user":
        new_messages[-1] = {"role": "user", "content": _merge_content(last_msg["content"], content)}
    else:
        new_messages.append({"role": role, "content": content})


def _merge_content(first: str | list[dict], second: str | list[dict]) -> str | list[dict]:
    """Join two messages' content: strings with a blank line, otherwise as one block list."""
    if isinstance(first, str) and isinstance(second, str):
        return first + "\n\n" + second
    return _as_blocks(first) + _as_blocks(second)


def _as_blocks(content: str | list[dict]) -> list[dict]:
    """Return content as a new list of content blocks."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content)


def truncate_middle(text: str, max_chars: int) -> str:
    """Cut text to about max_chars, keeping its start and end, for a helper LLM call.

    A pasted document can be far longer than a helper model's context window; its opening
    and closing usually say what the user wants.

    Args:
        text: The text to shorten.
        max_chars: The most characters kept from text; the omission marker is added on top.

    Returns:
        text unchanged if it fits, else its first and last max_chars // 2 characters joined by a
        marker saying how many characters were left out.
    """
    if len(text) <= max_chars:
        return text
    head = max_chars - max_chars // 2
    tail = max_chars // 2
    omitted = len(text) - head - tail
    return text[:head] + HELPER_CALL_TRUNCATION_MARKER.format(omitted=omitted) + (text[-tail:] if tail else "")


def truncate_query_for_helper_call(query: str) -> str:
    """This turn's user message, cut to `helper_call_max_query_chars` for a relevance or title call."""
    return truncate_middle(query or "", settings.helper_call_max_query_chars)


def prepare_recent_turns_for_decision(all_messages: list[Message], num_turns: int) -> list[dict]:
    """Lightweight conversation context for background RAG/search decision calls.

    Args:
        all_messages: Full message history for the chat, oldest first.
        num_turns: Number of most recent messages to include; 0 or falsy includes all.

    Returns:
        List of {"role", "content"} dicts, safe to pass directly as a Messages API `messages`
        list (consecutive user turns are merged to preserve strict role alternation). Each user
        message is cut to `helper_call_max_user_turn_chars` (start and end kept); assistant
        messages are truncated to a 200-char preview. If the turns together exceed
        `helper_call_max_context_chars`, the oldest are dropped; the latest turn is always kept.
    """
    recent_messages = all_messages[-num_turns:] if num_turns else all_messages
    new_messages: list[dict] = []
    for msg in recent_messages:
        if msg.role == "user":
            content_to_use = truncate_middle(msg.content or "", settings.helper_call_max_user_turn_chars)
        else:
            content_to_use = (msg.content or "")[:200]
            if not content_to_use:
                # Tool-only assistant turns are persisted with empty content; skipping them
                # keeps this safe to pass directly as a Messages API `messages` list, which
                # rejects empty-content blocks.
                continue

        if msg.role == "user" and new_messages and new_messages[-1]["role"] == "user":
            new_messages[-1]["content"] += "\n\n" + content_to_use
        else:
            new_messages.append({"role": msg.role, "content": content_to_use})
    return _drop_oldest_turns_to_fit(new_messages, settings.helper_call_max_context_chars)


def _drop_oldest_turns_to_fit(turns: list[dict], max_chars: int) -> list[dict]:
    """Drop the oldest turns until the rest fit in max_chars.

    The latest turn is always kept (cut to max_chars if it is too long on its own, e.g. several
    merged user turns), and if anything was dropped the result still starts with a user turn.
    """
    total = sum(len(t["content"]) for t in turns)
    start = 0
    while start < len(turns) - 1 and total > max_chars:
        total -= len(turns[start]["content"])
        start += 1
    if start:
        while start < len(turns) - 1 and turns[start]["role"] != "user":
            start += 1
    kept = turns[start:]
    if len(kept) == 1 and len(kept[0]["content"]) > max_chars:
        kept[0] = {**kept[0], "content": truncate_middle(kept[0]["content"], max_chars)}
    return kept
