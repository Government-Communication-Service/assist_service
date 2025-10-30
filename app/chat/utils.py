from logging import getLogger
from uuid import UUID

from fastapi import HTTPException, Path, status

from app.database.models import Message
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

    return chat


def prepare_message_objects_for_llm(all_messages: list[Message]) -> list[dict]:
    new_messages: list[dict] = []
    for msg in all_messages:
        # Determine content to use based on priority:
        # 1. If summary exists, use summary (for compacted messages)
        # 2. If RAG-enhanced content exists, use it
        # 3. Otherwise, use original content
        if hasattr(msg, "summary") and msg.summary is not None:
            content_to_use = msg.summary
            logger.debug(f"Using summary for message {getattr(msg, 'id', 'unknown')}: {len(content_to_use)} chars")
        elif hasattr(msg, "content_enhanced_with_rag") and msg.content_enhanced_with_rag is not None:
            content_to_use = msg.content_enhanced_with_rag
            logger.debug(f"Using RAG content for message {getattr(msg, 'id', 'unknown')}: {len(content_to_use)} chars")
        else:
            content_to_use = msg.content
            logger.debug(
                f"Using original content for message {getattr(msg, 'id', 'unknown')}: {len(content_to_use)} chars"
            )

        # check if this is a user message and if the last message was also a user message
        # then merge this message to the previous user message collapsing them into a single one.
        if msg.role == "user":
            last_msg = new_messages[-1] if new_messages else None
            if last_msg and last_msg["role"] == "user":
                new_messages[-1]["content"] += "\n\n" + content_to_use
            else:
                # if the last message was not a user message, then add this message as a new user message.
                new_messages.append({"role": "user", "content": content_to_use})
        else:
            # assistant messages are always added as new messages.
            if msg.content:
                new_messages.append({"role": "assistant", "content": content_to_use})

    logger.debug("Messages formatted for submission to LLM for final response")
    return new_messages
