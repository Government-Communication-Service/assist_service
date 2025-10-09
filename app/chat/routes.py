# ruff: noqa: B008
import logging

from fastapi import APIRouter, Body, Depends
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
    ChatSuccessResponse,
    ChatWithAllMessages,
    ChatWithLatestMessage,
    MessageCleanupResponse,
    UserChatsResponse,
)
from app.chat.service import (
    chat_add_message,
    chat_add_message_stream,
    chat_create,
    chat_create_stream,
    chat_get_messages,
    chat_request_data,
    clean_expired_message_content,
    get_all_user_chats,
    patch_chat_favourite,
    patch_chat_title,
    update_chat_title,
)
from app.chat.utils import chat_validator
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
):
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
