from datetime import datetime
from typing import List, Optional, Union

from pydantic import UUID1, UUID4, BaseModel, Field

from app.request_schemas import RequestStandard


class UserPromptRequestBody(BaseModel):
    title: str
    content: str


class UserPromptRequestData(UserPromptRequestBody, RequestStandard):
    user_id: int
    user_uuid: Union[UUID1, UUID4]
    uuid: Optional[UUID4]


class NoBodyUserPromptRequestData(RequestStandard):
    user_id: int
    uuid: Optional[UUID4]


class ItemUserPromptResponse(BaseModel):
    id: int
    uuid: UUID4
    user_id: int
    title: str
    content: str
    created_at: datetime
    updated_at: Optional[datetime] = Field(default=None)
    deleted_at: Optional[datetime] = Field(default=None)


class ItemUserPromptResponseWithDescription(ItemUserPromptResponse):
    description: str


class ListUserPromptResponse(BaseModel):
    user_prompts: List[ItemUserPromptResponseWithDescription]


class UserPromptInput(BaseModel):
    user_id: int
    title: str
    content: str
