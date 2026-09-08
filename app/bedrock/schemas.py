from typing import Optional

from anthropic.types import RedactedThinkingBlock, TextBlock, ThinkingBlock, ToolUseBlock
from pydantic import BaseModel


class LLMResponse(BaseModel):
    content: str | list[Optional[str | TextBlock | ThinkingBlock | RedactedThinkingBlock | ToolUseBlock]]
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        """The full input size for this call, cached portions included."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


class LLMTransaction(LLMResponse):
    input_cost: float
    output_cost: float
    completion_cost: float
