# ruff: noqa: B008

from typing import List

from fastapi import APIRouter, Body, Depends, HTTPException, status

from app.api.endpoints import ENDPOINTS
from app.auth.verify_service import (
    verify_and_get_auth_session_from_header,
    verify_and_get_user_from_header,
    verify_and_get_user_from_path_and_header,
    verify_and_parse_uuid,
    verify_auth_token,
)
from app.chat.schemas import FeedbackLabelResponse, MessageResponse
from app.database.db_operations import DbOperations
from app.database.table import (
    async_db_session,
)
from app.feedback.feedback_methods import (
    FeedbackRequest,
    process_message_feedback,
)

router = APIRouter()


async def message_validator(message_uuid: str) -> MessageResponse:
    message_uuid = verify_and_parse_uuid(message_uuid)

    async with async_db_session() as db_session:
        message = await DbOperations.get_message_by_uuid(db_session=db_session, message_uuid=message_uuid)

    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No message found with UUID '{message_uuid}'",
        )

    return MessageResponse(**message.client_response())


@router.get(
    path=ENDPOINTS.FEEDBACK_LABELS,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_header),
        Depends(verify_and_get_auth_session_from_header),
    ],
)
async def message_feedback_labels() -> List[FeedbackLabelResponse]:
    async with async_db_session() as db_session:
        labels = await DbOperations.get_message_feedback_labels_list(db_session=db_session)
        return [FeedbackLabelResponse(**label.client_response()) for label in labels]


@router.put(
    ENDPOINTS.MESSAGE_FEEDBACK,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
    ],
)
async def add_message_feedback(message=Depends(message_validator), data: FeedbackRequest = Body(...)):
    async with async_db_session() as db_session:
        return await process_message_feedback(db_session=db_session, message=message, feedback_request=data)
