from enum import Enum

from pydantic import BaseModel


class FileInfo(BaseModel):
    name: str
    media_type: str
    data: bytes


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class HistoricalMessage(BaseModel):
    content: str
    role: MessageRole


class AgentChatRequest(BaseModel):
    use_case_id: int = None
    file_info: FileInfo = None
    query: str


class AgentFinalResponse(BaseModel):
    final_response: str
    total_input_tokens: int
    total_output_tokens: int
