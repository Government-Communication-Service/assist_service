from datetime import datetime
from enum import Enum
from typing import Any, List, Optional
from uuid import UUID, uuid4

from fastapi import HTTPException, status
from pydantic import BaseModel, Field, field_validator

from app.request_schemas import RequestModel, RequestStandard
from app.themes_use_cases.schemas import PrebuiltPrompt
from app.user.schemas import DocumentSchema


class RoleEnum(str, Enum):
    user = "user"
    assistant = "assistant"


class MessageDefaults(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    chat_id: int
    auth_session_id: int
    interrupted: bool = False
    llm_id: int
    tokens: int = 0


class ChatResponse(BaseModel):
    id: int
    uuid: UUID
    created_at: datetime
    updated_at: Optional[datetime] = Field(default=None)
    deleted_at: Optional[datetime] = Field(default=None)
    user_id: int
    use_case_id: int
    title: str
    from_open_chat: bool
    use_rag: bool


class MessageResponse(BaseModel):
    uuid: UUID
    created_at: datetime
    updated_at: Optional[datetime] = Field(default=None)
    content: str
    role: RoleEnum
    interrupted: bool
    citation: str


class DocumentAccessError(Exception):
    def __init__(self, *args, document_uuids: List[str]):
        super().__init__(*args)
        self.document_uuids = document_uuids


class ChatLLMResponse(BaseModel):
    content: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]


class ChatBaseRequest(RequestStandard):
    query: str = ""
    system: str = ""
    stream: bool = False
    parent_message_id: Optional[str] = None
    user_group_ids: Optional[List[int]] = []
    use_case_id: Optional[int]
    use_rag: bool = True
    use_gov_uk_search_api: bool = False
    enable_web_browsing: bool = False
    document_uuids: Optional[List[str]] = None


class ChatQueryRequest(RequestModel):
    query: str


class ChatRequest(ChatQueryRequest):
    use_case_id: Optional[str] = ""
    use_rag: bool = True
    use_gov_uk_search_api: bool = False
    document_uuids: Optional[List[str]] = None

    @field_validator("query")
    def validate_query(cls, v):
        if not v:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="'query' is not present in the request body.",
            )

        return v


class ChatPost(ChatRequest):
    pass


class ChatPut(ChatPost):
    parent_message_id: str = Field(
        None,
        description="The UUID of the parent message to which the new message should be appended.",
    )


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


class FeedbackLabelListResponse(BaseModel):
    feedback_labels: List[FeedbackLabelResponse]


class ChatCreateMessageInput(ChatBaseRequest):
    use_case: Optional[Any] = None
    initial_call: bool = False


class SuccessResponse(BaseModel):
    status: str = "success"
    status_message: str = "success"


class ItemResponse(BaseModel):
    uuid: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        arbitrary_types_allowed = True
        from_attributes = True


class ItemTitleResponse(ItemResponse):
    title: str


class ThemeResponseData(ItemTitleResponse):
    subtitle: str
    position: Optional[int] = None


class UseCaseResponseData(ItemTitleResponse):
    theme_uuid: UUID
    instruction: str
    user_input_form: str
    position: Optional[int] = None


class ThemeResponse(SuccessResponse, ThemeResponseData):
    pass


class ThemesResponse(SuccessResponse):
    themes: List[ThemeResponseData]


class UseCaseResponse(SuccessResponse, UseCaseResponseData):
    pass


class UseCasesResponse(SuccessResponse, ItemTitleResponse):
    use_cases: List[UseCaseResponseData]


class PrebuiltPromptsResponse(SuccessResponse):
    prompts: List[PrebuiltPrompt]


class MessageBasicResponse(ItemResponse):
    content: str
    role: RoleEnum
    redacted: bool = False
    redaction_message: str = ""
    redaction_alert_level: bool = False
    interrupted: bool = False
    citation: str = ""


class ChatBasicResponse(ItemTitleResponse):
    from_open_chat: bool
    use_rag: bool = True
    use_gov_uk_search_api: bool = False
    documents: Optional[List[DocumentSchema]] = None


class UserChatsResponse(SuccessResponse, ItemResponse):
    chats: List[ChatBasicResponse] = []


class ChatSuccessResponse(SuccessResponse, ItemTitleResponse):
    pass


class ChatWithLatestMessage(SuccessResponse, ChatBasicResponse):
    message: MessageBasicResponse


class ChatWithAllMessages(SuccessResponse, ChatBasicResponse):
    messages: List[MessageBasicResponse] = []


class MessageFeedbackResponse(SuccessResponse, ItemResponse):
    pass


class ChatRequestData(ChatPost, RequestStandard):
    user_group_ids: List = []
    use_case_id: int = None


class ChatCreateInput(ChatBaseRequest):
    pass


class ChatTitleRequest(RequestModel):
    query: str = ""
    system: str = ""


class MessageCleanupResponse(SuccessResponse):
    """Response schema for message content cleanup operation."""

    message: str
    cleaned_count: int
