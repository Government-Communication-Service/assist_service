# ruff: noqa: B008
from logging import getLogger
from uuid import UUID

from fastapi import Body, Depends, HTTPException, Path, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.exceptions import UuidInvalidError, UuidMissingError
from app.auth.utils import verify_and_parse_uuid
from app.auth.verify_service import verify_and_get_user_from_header
from app.chat.constants import PRIVATE_SHARE_ACCESS_DENIED
from app.database.db_operations import DbOperations
from app.database.db_session import get_db_session
from app.database.models import Chat, Message, User
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
    chat: Chat = Depends(shared_chat_validator),
    user: User = Depends(verify_and_get_user_from_header),
    db_session: AsyncSession = Depends(get_db_session),
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
