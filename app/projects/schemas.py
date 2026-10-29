from datetime import datetime
from typing import Annotated, List, Optional
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, StringConstraints, model_validator

from app.chat.schemas import ChatBasicResponse

ProjectName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
# A blank description is stored as no description.
ProjectDescription = Annotated[
    str, StringConstraints(strip_whitespace=True, max_length=1000), AfterValidator(lambda value: value or None)
]


class ProjectCreateRequest(BaseModel):
    name: ProjectName
    description: Optional[ProjectDescription] = None


class ProjectUpdateRequest(BaseModel):
    """Partial update: only the fields sent are changed. Send `description: null` to clear it."""

    name: Optional[ProjectName] = None
    description: Optional[ProjectDescription] = None

    @model_validator(mode="after")
    def validate_fields(self):
        if not self.model_fields_set:
            raise ValueError("Provide at least one of 'name' or 'description'.")
        if "name" in self.model_fields_set and self.name is None:
            raise ValueError("'name' cannot be null.")
        return self


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    uuid: UUID
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ProjectListResponse(BaseModel):
    projects: List[ProjectResponse] = []


class ProjectChatsResponse(BaseModel):
    project_uuid: UUID
    chats: List[ChatBasicResponse] = []
