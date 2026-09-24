# ruff: noqa: B008
from fastapi import APIRouter, Body, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints import ENDPOINTS
from app.auth.verify_service import (
    verify_and_get_auth_session_from_header,
    verify_and_get_user_from_path_and_header,
    verify_auth_token,
)
from app.chat.schemas import ChatRequest, ChatSuccessResponse
from app.chat.utils import chat_validator
from app.classification_and_title.service import patch_chat_title, update_chat_title
from app.database.db_session import get_db_session
from app.database.models import Chat

router = APIRouter()


@router.put(
    path=ENDPOINTS.CHAT_TITLE,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
    ],
)
async def create_chat_title(
    chat: Chat = Depends(chat_validator),
    data: ChatRequest = Body(...),
) -> ChatSuccessResponse:
    return await update_chat_title(chat=chat, data=data)


@router.get(
    path=ENDPOINTS.CHAT_TITLE,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
    ],
)
async def get_chat_title(chat=Depends(chat_validator)) -> ChatSuccessResponse:
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
    return await patch_chat_title(db_session=db_session, chat=chat, title=title)
