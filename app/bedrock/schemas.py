from typing import Optional

from anthropic.types import RedactedThinkingBlock, TextBlock, ThinkingBlock, ToolUseBlock
from pydantic import BaseModel


class LLMResponse(BaseModel):
    content: str | list[Optional[str | TextBlock | ThinkingBlock | RedactedThinkingBlock | ToolUseBlock]]
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class LLMTransaction(LLMResponse):
    input_cost: float
    output_cost: float
    completion_cost: float
