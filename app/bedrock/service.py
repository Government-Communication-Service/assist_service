import logging
from decimal import Decimal
from typing import Any

from app.bedrock.schemas import LLMResponse, LLMTransaction
from app.config import settings
from app.database.models import LLM

logger = logging.getLogger(__name__)


def _as_token_count(value: Any) -> int:
    """Coerce a usage field to an int, falling back to 0 for anything unusable.

    These values feed straight into Decimal arithmetic for cost, so a non-numeric value must
    never propagate — None when caching is off, and test doubles, both land here.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def extract_cache_tokens(usage: Any) -> tuple[int, int]:
    """Pull (cache_read_tokens, cache_write_tokens) out of an Anthropic usage object.

    Both fields are absent or None when caching is not in play, so they are read
    defensively rather than assumed present.
    """
    return (
        _as_token_count(getattr(usage, "cache_read_input_tokens", None)),
        _as_token_count(getattr(usage, "cache_creation_input_tokens", None)),
    )


def cache_read_cost_multiplier(llm: LLM) -> float:
    """The share of the input price that a cache read costs: the model's own, else the global default."""
    if llm.cache_read_cost_multiplier is not None:
        return llm.cache_read_cost_multiplier
    return settings.cache_read_cost_multiplier


def llm_transaction(llm: LLM, response: LLMResponse) -> LLMTransaction:
    input_cost = llm.input_cost_per_token * (
        response.input_tokens
        + (response.cache_read_tokens * cache_read_cost_multiplier(llm))
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
        + (Decimal(cache_read_tokens) * Decimal(str(cache_read_cost_multiplier(llm))))
        + (Decimal(cache_write_tokens) * Decimal(str(settings.cache_write_cost_multiplier)))
    )
    return (input_cost_per_token * billable_input_tokens) + (Decimal(str(llm.output_cost_per_token)) * output_tokens)
