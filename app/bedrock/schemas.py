from typing import Optional

from anthropic.types import TextBlock, ToolUseBlock
from pydantic import BaseModel


class LLMResponse(BaseModel):
    content: str | list[Optional[str | TextBlock | ToolUseBlock]]
    input_tokens: int
    output_tokens: int


class LLMTransaction(LLMResponse):
    input_cost: float
    output_cost: float
    completion_cost: float
