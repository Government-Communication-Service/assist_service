from uuid import UUID

from pydantic import BaseModel

from app.schemas.request import RequestModel


class ClassificationInput(BaseModel):
    title: str
    description: str | None = None


class ClassificationResponse(BaseModel):
    uuid: UUID
    title: str
    description: str | None = None


class ClassificationsResponse(BaseModel):
    classifications: list[ClassificationResponse]


class BulkSyncResult(BaseModel):
    created: int
    updated: int
    deprecated: int


class ChatTitleRequest(RequestModel):
    query: str = ""
    system: str = ""
