# ruff: noqa: B008
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints import ENDPOINTS
from app.auth.verify_service import (
    verify_and_get_auth_session_from_header,
    verify_and_get_user_from_path_and_header,
    verify_auth_token,
)
from app.chat.schemas import (
    ChatSharedUser,
    ChatShareResponse,
    ChatShareUsersResponse,
    ChatWithAllMessages,
)
from app.chat.service import (
    add_chat_share_user,
    chat_get_messages,
    get_chat_share_users,
    patch_chat_share,
    remove_chat_share_user,
    set_chat_share_user_notified,
)
from app.chat.utils import (
    chat_validator,
    verify_shared_chat_access,
    verify_shared_user_uuid_from_body,
    verify_shared_user_uuid_from_path,
)
from app.database.db_session import get_db_session
from app.database.models import User

router = APIRouter()


def _chat_share_users_response(chat, shared_users: list[dict]) -> ChatShareUsersResponse:
    """Builds the shared-users response from the service layer's list of user dicts."""
    return ChatShareUsersResponse(
        uuid=chat.uuid,
        shared_user_uuids=[u["uuid"] for u in shared_users],
        shared_users=[ChatSharedUser(**u) for u in shared_users],
    )


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
