from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from app.chat.schemas import ItemResponse, SuccessResponse
from app.schemas.request import RequestModel


class MessageFeedbackEnum(int, Enum):
    positive = 1
    negative = -1
    removed = 0


class FeedbackRequest(RequestModel):
    score: int
    freetext: Optional[str] = None
    label: Optional[str] = None


class FeedbackLabelResponse(BaseModel):
    uuid: UUID
    created_at: datetime
    updated_at: Optional[datetime] = Field(default=None)
    deleted_at: Optional[datetime] = Field(default=None)
    label: str


class MessageFeedbackResponse(SuccessResponse, ItemResponse):
    pass
