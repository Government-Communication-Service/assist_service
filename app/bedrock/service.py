import logging
from decimal import Decimal

from app.bedrock.schemas import LLMResponse, LLMTransaction
from app.config import settings
from app.database.models import LLM

logger = logging.getLogger(__name__)


def llm_transaction(llm: LLM, response: LLMResponse) -> LLMTransaction:
    input_cost = llm.input_cost_per_token * (
        response.input_tokens
        + (response.cache_read_tokens * settings.cache_read_cost_multiplier)
        + (response.cache_write_tokens * settings.cache_write_cost_multiplier)
    )
    output_cost = response.output_tokens * llm.output_cost_per_token

    return LLMTransaction(
        content=response.content,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cache_read_tokens=response.cache_read_tokens,
        cache_write_tokens=response.cache_write_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        completion_cost=input_cost + output_cost,
    )


def calculate_completion_cost(
    llm: LLM,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> Decimal:
    input_cost_per_token = Decimal(str(llm.input_cost_per_token))
    billable_input_tokens = (
        Decimal(input_tokens)
        + (Decimal(cache_read_tokens) * Decimal(str(settings.cache_read_cost_multiplier)))
        + (Decimal(cache_write_tokens) * Decimal(str(settings.cache_write_cost_multiplier)))
    )
    return (input_cost_per_token * billable_input_tokens) + (Decimal(str(llm.output_cost_per_token)) * output_tokens)
