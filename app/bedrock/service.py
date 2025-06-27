import logging
from decimal import Decimal

from app.bedrock.schemas import LLMResponse, LLMTransaction
from app.database.models import LLM

logger = logging.getLogger(__name__)


def llm_transaction(llm: LLM, response: LLMResponse) -> LLMTransaction:
    input_cost = response.input_tokens * llm.input_cost_per_token
    output_cost = response.output_tokens * llm.output_cost_per_token

    return LLMTransaction(
        content=response.content,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        completion_cost=input_cost + output_cost,
    )


def calculate_completion_cost(llm: LLM, input_tokens: int, output_tokens: int) -> Decimal:
    return (Decimal(str(llm.input_cost_per_token)) * input_tokens) + (
        Decimal(str(llm.output_cost_per_token)) * output_tokens
    )
