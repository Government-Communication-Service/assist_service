# ruff: noqa: B008

from typing import List

from fastapi import APIRouter, Body, Depends, HTTPException, status

from app.api import ENDPOINTS, endpoint_defaults
from app.api.paths import ApiPaths
from app.chat.schemas import FeedbackLabelResponse, MessageResponse
from app.database.db_operations import DbOperations
from app.database.table import (
    async_db_session,
)
from app.feedback.feedback_methods import (
    FeedbackRequest,
    process_message_feedback,
)
from app.verify_uuid import verify_uuid

router = APIRouter()


async def message_validator(message_uuid: str) -> MessageResponse:
    message_uuid = verify_uuid("message_uuid path", message_uuid)

    async with async_db_session() as db_session:
        message = await DbOperations.get_message_by_uuid(db_session=db_session, message_uuid=message_uuid)

    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No message found with UUID '{message_uuid}'",
        )

    return MessageResponse(**message.client_response())


@router.get(ENDPOINTS.FEEDBACK_LABELS, **endpoint_defaults())
async def message_feedback_labels() -> List[FeedbackLabelResponse]:
    async with async_db_session() as db_session:
        labels = await DbOperations.get_message_feedback_labels_list(db_session=db_session)
        return [FeedbackLabelResponse(**label.client_response()) for label in labels]


@router.put(
    ENDPOINTS.MESSAGE_FEEDBACK,
    **endpoint_defaults(extra_dependencies=[ApiPaths.USER_UUID]),
)
async def add_message_feedback(message=Depends(message_validator), data: FeedbackRequest = Body(...)):
    async with async_db_session() as db_session:
        return await process_message_feedback(db_session=db_session, message=message, feedback_request=data)
