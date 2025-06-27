from datetime import datetime
from typing import List, Optional, Union

from pydantic import UUID1, UUID4, BaseModel, Field


class DocumentSchema(BaseModel):
    name: str
    uuid: UUID4
    created_at: datetime
    expired_at: Optional[datetime] = Field(default=None)
    deleted_at: Optional[datetime] = Field(default=None)
    last_used: Optional[datetime] = Field(default=None)


class UploadDocumentResponse(BaseModel):
    message: str
    document_uuid: str


class ListDocumentResponse(BaseModel):
    user_documents: List[DocumentSchema]
    central_documents: List[DocumentSchema]


class DocumentResponse(BaseModel):
    message: str
    document_uuid: str


class UserInput(BaseModel):
    job_title: str | None = Field(None, max_length=255)
    region: str | None = Field(None, max_length=255)
    sector: str | None = Field(None, max_length=255)
    organisation: str | None = Field(None, max_length=255)
    grade: str | None = Field(None, max_length=255)
    communicator_role: bool | None = None


class UserCreationInput(UserInput):
    uuid: Union[UUID1, UUID4]


class UserCreationResponse(BaseModel):
    success: bool
    message: str
