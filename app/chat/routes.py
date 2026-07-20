# ruff: noqa: B008
import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from app.api.endpoints import ENDPOINTS
from app.auth.verify_service import (
    verify_and_get_auth_session_from_header,
    verify_and_get_user_from_path_and_header,
    verify_auth_token,
)
from app.chat.schemas import (
    ChatCreateInput,
    ChatRequest,
    ChatSharedUser,
    ChatShareResponse,
    ChatShareUsersResponse,
    ChatSuccessResponse,
    ChatWithAllMessages,
    ChatWithLatestMessage,
    MessageCleanupResponse,
    UserChatsResponse,
)
from app.chat.service import (
    add_chat_share_user,
    chat_add_message,
    chat_add_message_stream,
    chat_archive,
    chat_create,
    chat_create_stream,
    chat_get_messages,
    chat_request_data,
    clean_expired_message_content,
    get_all_user_chats,
    get_chat_share_users,
    patch_chat_favourite,
    patch_chat_share,
    patch_chat_title,
    remove_chat_share_user,
    set_chat_share_user_notified,
    update_chat_title,
)
from app.chat.utils import (
    chat_validator,
    verify_shared_chat_access,
    verify_shared_user_uuid_from_body,
    verify_shared_user_uuid_from_path,
)
from app.database.db_session import get_db_session
from app.database.models import User
from app.database.table import (
    async_db_session,
)

router = APIRouter()

logger = logging.getLogger()


@router.post(
    path=ENDPOINTS.CHATS,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
    ],
    response_model=ChatWithLatestMessage,
)
async def create_new_chat(data=Depends(chat_request_data)):
    chat_input = ChatCreateInput(**data.dict())
    return await chat_create(chat_input)


@router.get(
    path=ENDPOINTS.CHAT_ITEM,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
    ],
    response_model=ChatWithAllMessages,
)
async def get_chat_entry(
    chat=Depends(chat_validator),
):
    return await chat_get_messages(chat)


@router.put(
    path=ENDPOINTS.CHAT_ITEM,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
    ],
    response_model=ChatWithLatestMessage,
)
async def add_new_chat_message(chat=Depends(chat_validator), data=Depends(chat_request_data)):
    return await chat_add_message(chat, data)


@router.get(
    path=ENDPOINTS.CHAT_MESSAGES,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
    ],
    response_model=ChatWithAllMessages,
)
async def get_chat_messages(
    chat=Depends(chat_validator),
):
    logger.info("Calling chat messages")
    return await chat_get_messages(chat)


@router.put(
    path=ENDPOINTS.CHAT_TITLE,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
    ],
)
async def create_chat_title(
    chat=Depends(chat_validator),
    data: ChatRequest = Body(...),
) -> ChatSuccessResponse:
    async with async_db_session() as db_session:
        return await update_chat_title(db_session=db_session, chat=chat, data=data)


@router.get(
    path=ENDPOINTS.CHAT_TITLE,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
    ],
)
async def get_chat_title(chat=Depends(chat_validator)) -> ChatSuccessResponse:
    """
    Get the title of a chat.

    Args:
        chat: Chat object from chat_validator dependency

    Returns:
        ChatSuccessResponse: Response containing chat details including title
    """
    return ChatSuccessResponse(uuid=chat.uuid, created_at=chat.created_at, updated_at=chat.updated_at, title=chat.title)


@router.patch(
    path=ENDPOINTS.CHAT_TITLE,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
    ],
)
async def user_update_chat_title(
    chat=Depends(chat_validator),
    title: str = Body(..., embed=True),
    db_session: AsyncSession = Depends(get_db_session),
) -> ChatSuccessResponse:
    """
    Update the title of an existing chat.

    Args:
        chat (Chat): Chat object obtained from chat_validator dependency.
            Contains the existing chat details and validates user permissions.
        title (str): The new title to be assigned to the chat.

    Returns:
        ChatSuccessResponse: Response object containing:
            - uuid: The chat's unique identifier
            - created_at: Original creation timestamp
            - updated_at: Last update timestamp
            - title: The newly updated chat title
            - status: Success status
            - status_message: Success message
    """
    return await patch_chat_title(db_session=db_session, chat=chat, title=title)


@router.get(
    path=ENDPOINTS.USER_GET_CHATS,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_auth_session_from_header),
    ],
    response_model=UserChatsResponse,
)
async def get_user_chats(
    db_session: AsyncSession = Depends(get_db_session),
    user: User = Depends(verify_and_get_user_from_path_and_header),
):
    """
    Fetch a user's chat history by their ID. Expandable down the line to include filters / recent slices.
    """
    return await get_all_user_chats(db_session=db_session, user=user)


@router.post(
    path=ENDPOINTS.CHAT_CREATE_STREAM,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
    ],
)
async def create_new_chat_stream(data=Depends(chat_request_data)) -> StreamingResponse:
    logger.info("Calling new chat stream")
    return await chat_create_stream(ChatCreateInput(**data.dict()))


@router.put(
    path=ENDPOINTS.CHAT_UPDATE_STREAM,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
    ],
)
async def add_new_message_stream(chat=Depends(chat_validator), data=Depends(chat_request_data)):
    logger.info("Calling add message to chat stream")
    return await chat_add_message_stream(chat, data)


@router.delete(
    path=ENDPOINTS.CHAT_CLEANUP_EXPIRED_CONTENT,
    dependencies=[Depends(verify_auth_token)],
    response_model=MessageCleanupResponse,
)
async def delete_expired_message_content(db_session: AsyncSession = Depends(get_db_session)):
    """
    Remove content from messages older than 1 year for data protection compliance.

    This endpoint cleans up message content by setting it to empty string and updating
    the deleted_at timestamp for messages that are older than 365 days.
    Authentication token is required to call this endpoint.

    Returns:
        MessageCleanupResponse: Response containing the number of messages cleaned and success status.
    """
    logger.info("Calling delete expired message content")

    # Call service layer which returns simple data
    cleanup_data = await clean_expired_message_content(db_session)

    # Commit the transaction
    await db_session.commit()

    # Format response in API layer
    return MessageCleanupResponse(
        message=(
            f"Successfully cleaned content from {cleanup_data['cleaned_count']} expired messages "
            f"and marked {cleanup_data['cleaned_chats']} chats as deleted"
        ),
        cleaned_count=cleanup_data["cleaned_count"],
    )


@router.patch(
    path=ENDPOINTS.CHAT_FAVOURITE,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
    ],
    response_model=ChatSuccessResponse,
)
async def update_chat_favourite(
    chat=Depends(chat_validator),
    favourite: bool = Body(False, embed=True),
    db_session: AsyncSession = Depends(get_db_session),
) -> ChatSuccessResponse:
    """
    Update the favourite status of a chat.

    Args:
        chat: Chat object from chat_validator dependency
        favourite (bool): The new favourite status for the chat. If null, defaults to False.
        db_session: Database session

    Returns:
        ChatSuccessResponse: Response containing updated chat details
    """
    return await patch_chat_favourite(db_session=db_session, chat=chat, favourite=favourite)


@router.patch(
    path=ENDPOINTS.CHAT_SHARE,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
    ],
    response_model=ChatShareResponse,
)
async def update_chat_share(
    chat=Depends(chat_validator),
    share: bool = Body(False, embed=True),
    share_private: Optional[bool] = Body(None, embed=True),
    db_session: AsyncSession = Depends(get_db_session),
) -> ChatShareResponse:
    """
    Update the share status of a chat.

    Args:
        chat: Chat object from chat_validator dependency
        share (bool): The new share status for the chat. If omitted, defaults to False.
        share_private (Optional[bool]): Whether the share is private (visible only to users
            the owner has added). If omitted, the current value is left unchanged so that
            clients unaware of private shares keep working.
        db_session: Database session

    Returns:
        ChatSuccessResponse: Response containing updated chat details
    """
    return await patch_chat_share(db_session=db_session, chat=chat, share=share, share_private=share_private)


def _chat_share_users_response(chat, shared_users: list[dict]) -> ChatShareUsersResponse:
    """Builds the shared-users response from the service layer's list of user dicts."""
    return ChatShareUsersResponse(
        uuid=chat.uuid,
        shared_user_uuids=[u["uuid"] for u in shared_users],
        shared_users=[ChatSharedUser(**u) for u in shared_users],
    )


@router.get(
    path=ENDPOINTS.CHAT_SHARE_USERS,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
    ],
    response_model=ChatShareUsersResponse,
)
async def get_chat_share_users_list(
    chat=Depends(chat_validator),
    db_session: AsyncSession = Depends(get_db_session),
) -> ChatShareUsersResponse:
    """
    List the users allowed to view this chat's private share. Only the chat owner can call this.

    Returns:
        ChatShareUsersResponse: Response containing the shared users and their notification state
    """
    shared_users = await get_chat_share_users(db_session=db_session, chat=chat)
    return _chat_share_users_response(chat, shared_users)


@router.post(
    path=ENDPOINTS.CHAT_SHARE_USERS,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_auth_session_from_header),
    ],
    response_model=ChatShareUsersResponse,
)
async def add_chat_share_user_entry(
    user: User = Depends(verify_and_get_user_from_path_and_header),
    chat=Depends(chat_validator),
    shared_user_uuid: UUID = Depends(verify_shared_user_uuid_from_body),
    db_session: AsyncSession = Depends(get_db_session),
) -> ChatShareUsersResponse:
    """
    Add a user (by UUID) to this chat's private share. Only the chat owner can call this.
    Adding a user who already has access is a no-op.

    Args:
        chat: Chat object from chat_validator dependency
        user: The chat owner, from path and header validation
        shared_user_uuid: UUID of the user to grant access to, from the request body
        db_session: Database session

    Returns:
        ChatShareUsersResponse: Response containing the updated shared users' UUIDs
    """
    if shared_user_uuid == user.uuid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The chat owner already has access to this chat",
        )

    shared_users = await add_chat_share_user(db_session=db_session, chat=chat, shared_user_uuid=shared_user_uuid)
    return _chat_share_users_response(chat, shared_users)


@router.delete(
    path=ENDPOINTS.CHAT_SHARE_USER,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
    ],
    response_model=ChatShareUsersResponse,
)
async def remove_chat_share_user_entry(
    chat=Depends(chat_validator),
    shared_user_uuid: UUID = Depends(verify_shared_user_uuid_from_path),
    db_session: AsyncSession = Depends(get_db_session),
) -> ChatShareUsersResponse:
    """
    Remove a user (by UUID) from this chat's private share. Only the chat owner can call this.

    Args:
        chat: Chat object from chat_validator dependency
        shared_user_uuid: UUID of the user to revoke access from, from the path
        db_session: Database session

    Returns:
        ChatShareUsersResponse: Response containing the updated shared users' UUIDs
    """
    shared_users = await remove_chat_share_user(db_session=db_session, chat=chat, shared_user_uuid=shared_user_uuid)
    if shared_users is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{shared_user_uuid}' is not part of this chat's private share",
        )
    return _chat_share_users_response(chat, shared_users)


@router.patch(
    path=ENDPOINTS.CHAT_SHARE_USER,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
    ],
    response_model=ChatShareUsersResponse,
)
async def update_chat_share_user_notified(
    chat=Depends(chat_validator),
    shared_user_uuid: UUID = Depends(verify_shared_user_uuid_from_path),
    notified: bool = Body(..., embed=True),
    db_session: AsyncSession = Depends(get_db_session),
) -> ChatShareUsersResponse:
    """
    Record that a user has been notified that this chat was privately shared with them.
    Only the chat owner can call this. The notified_at timestamp is set server-side;
    sending {"notified": false} clears it.

    Args:
        chat: Chat object from chat_validator dependency
        shared_user_uuid: UUID of the shared user being notified, from the path
        notified (bool): True to stamp the notification time, False to clear it
        db_session: Database session

    Returns:
        ChatShareUsersResponse: Response containing the updated shared users and notification state
    """
    shared_users = await set_chat_share_user_notified(
        db_session=db_session, chat=chat, shared_user_uuid=shared_user_uuid, notified=notified
    )
    if shared_users is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{shared_user_uuid}' is not part of this chat's private share",
        )
    return _chat_share_users_response(chat, shared_users)


@router.get(
    path=ENDPOINTS.CHAT_SHARED,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_auth_session_from_header),
    ],
    response_model=ChatWithAllMessages,
)
async def get_shared_chat_messages(
    chat=Depends(verify_shared_chat_access),
):
    """
    Retrieve messages for a shared chat using share_code.
    Requires authentication but not ownership. For privately shared chats, the requesting
    user must be the chat owner or have been added to the share by the owner.
    """
    return await chat_get_messages(chat)


@router.patch(
    path=ENDPOINTS.CHAT_ARCHIVE,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
    ],
    response_model=ChatSuccessResponse,
)
async def archive_chat(
    chat=Depends(chat_validator),
    db_session: AsyncSession = Depends(get_db_session),
) -> ChatSuccessResponse:
    """
    Archive a chat by setting the deleted_at timestamp.

    Args:
        chat: Chat object from chat_validator dependency
        db_session: Database session

    Returns:
        ChatSuccessResponse: Response containing updated chat details
    """
    return await chat_archive(db_session=db_session, chat=chat)
