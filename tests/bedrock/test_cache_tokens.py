"""Tests for prompt-cache token accounting.

These values feed straight into Decimal cost arithmetic, so anything non-numeric reaching
them is a crash rather than a wrong number. The extraction is deliberately forgiving.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from app.bedrock.bedrock import extract_cache_tokens
from app.bedrock.schemas import LLMResponse
from app.bedrock.service import calculate_completion_cost, llm_transaction

logger = logging.getLogger(__name__)


def make_llm(input_cost_per_token=3e-6, output_cost_per_token=15e-6):
    return SimpleNamespace(
        input_cost_per_token=input_cost_per_token,
        output_cost_per_token=output_cost_per_token,
    )


def test_extracts_real_cache_tokens():
    usage = SimpleNamespace(cache_read_input_tokens=9000, cache_creation_input_tokens=120)

    assert extract_cache_tokens(usage) == (9000, 120)


def test_missing_cache_fields_are_zero():
    """Absent when the request had no cache_control at all."""
    assert extract_cache_tokens(SimpleNamespace()) == (0, 0)


def test_null_cache_fields_are_zero():
    usage = SimpleNamespace(cache_read_input_tokens=None, cache_creation_input_tokens=None)

    assert extract_cache_tokens(usage) == (0, 0)


def test_non_numeric_cache_fields_do_not_leak():
    """Regression: a Mock usage object used to yield Mocks, which then hit Decimal() and blew
    up the whole request with 'Mock object cannot be interpreted as an integer'."""
    read, write = extract_cache_tokens(Mock())

    assert isinstance(read, int)
    assert isinstance(write, int)


def test_magic_mock_usage_yields_ints():
    read, write = extract_cache_tokens(MagicMock())

    assert isinstance(read, int)
    assert isinstance(write, int)


def test_cache_reads_are_discounted_and_writes_are_premium():
    response = LLMResponse(
        content="reply",
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=9000,
        cache_write_tokens=1000,
    )

    transaction = llm_transaction(make_llm(), response)

    # 100 uncached + (9000 * 0.1) + (1000 * 1.25) = 2250 billable input tokens.
    assert transaction.input_cost == 2250 * 3e-6
    assert transaction.output_cost == 50 * 15e-6
    assert transaction.completion_cost == transaction.input_cost + transaction.output_cost


def test_transaction_carries_cache_tokens_through():
    response = LLMResponse(
        content="reply",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=7,
        cache_write_tokens=3,
    )

    transaction = llm_transaction(make_llm(), response)

    assert transaction.cache_read_tokens == 7
    assert transaction.cache_write_tokens == 3


def test_total_input_tokens_reports_the_real_conversation_size():
    """input_tokens alone hides the cached history, which is most of a long chat."""
    response = LLMResponse(
        content="reply",
        input_tokens=180,
        output_tokens=400,
        cache_read_tokens=149_000,
        cache_write_tokens=900,
    )

    assert response.total_input_tokens == 150_080


def test_cache_tokens_default_to_zero():
    response = LLMResponse(content="reply", input_tokens=10, output_tokens=5)

    assert response.cache_read_tokens == 0
    assert response.cache_write_tokens == 0
    assert response.total_input_tokens == 10


def test_completion_cost_without_cache_tokens_is_unchanged():
    """The two-argument form is still used by callers that never see cache tokens."""
    cost = calculate_completion_cost(make_llm(), 100, 50)

    assert float(cost) == pytest.approx((100 * 3e-6) + (50 * 15e-6))


def test_completion_cost_with_cache_tokens():
    cost = calculate_completion_cost(make_llm(), 100, 50, cache_read_tokens=9000, cache_write_tokens=1000)

    assert float(cost) == pytest.approx((2250 * 3e-6) + (50 * 15e-6))
