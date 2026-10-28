import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import pytest
from anthropic._exceptions import ServiceUnavailableError
from anthropic.types import Message
from anthropic.types.usage import Usage

from app.bedrock import BedrockHandler, RunMode
from app.bedrock.bedrock_types import (
    AsyncAnthropicBedrock,
    AsyncAnthropicBedrockProvider,
    BedrockError,
    BedrockErrorType,
)
from app.bedrock.retry import _backoff_seconds
from app.bedrock.schemas import LLMTransaction
from app.config import AWS_BEDROCK_REGION1, AWS_BEDROCK_REGION2


@pytest.fixture(autouse=True)
def no_call_logging():
    """These tests are about region failover, not llm_internal_response logging."""
    with patch("app.bedrock.bedrock.log_llm_call", new_callable=AsyncMock, return_value=None):
        yield


@pytest.fixture(autouse=True)
def no_backoff():
    """Retry without sleeping between attempts."""
    with patch("app.bedrock.retry._backoff_seconds", return_value=0):
        yield


_REQUEST = httpx.Request("POST", "https://bedrock-runtime.example/model/invoke")


def _status_error(cls, status_code: int, message: str = "error"):
    return cls(message, response=httpx.Response(status_code, request=_REQUEST), body=None)


def _ok_message():
    return Message(
        id="msg_ok",
        type="message",
        role="assistant",
        content=[],
        model="claude-test",
        stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


bedrock = BedrockHandler()
bedrock_async = BedrockHandler(mode=RunMode.ASYNC)
original_invoke_async_function = bedrock_async._invoke_async
original_invoke_async_with_call_cost_details_function = bedrock_async._invoke_async_with_call_cost_details


@patch("app.bedrock.bedrock_types.AsyncAnthropicBedrockProvider.get")
@patch("app.bedrock.bedrock.BedrockHandler._invoke_async")
async def test_failover_switches_from_west_to_east(mock_invoke_async, mock_provider_get):
    """Simulate us-west-2 outage and verify we switch to us-east-1 and recover."""

    # first provider call during handler init returns a west client
    # second call (after failover) should request the east client
    def _provider_side_effect(region):
        return SimpleNamespace(aws_region=region)

    mock_provider_get.side_effect = _provider_side_effect
    ok = Message(
        id="msg_ok",
        type="message",
        role="assistant",
        content=[],
        model="claude-test",
        stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=1),
    )
    mock_invoke_async.side_effect = [anthropic.APIConnectionError(request=_REQUEST), ok]

    bedrock = BedrockHandler(mode=RunMode.ASYNC)

    result = await bedrock.invoke_async([{"role": "user", "content": "hello"}])

    assert result.id == "msg_ok"
    # first call for init (west), second call after failover (east)
    assert mock_provider_get.call_args_list[0].args[0] == AWS_BEDROCK_REGION1
    assert mock_provider_get.call_args_list[1].args[0] == AWS_BEDROCK_REGION2


async def test_failover_moves_real_provider_client_region(monkeypatch):
    """Use real provider with a fake client to assert region swap and retry succeed."""

    # reset provider cache
    AsyncAnthropicBedrockProvider._AsyncAnthropicBedrockProvider__clients.clear()

    call_regions: list[str] = []

    # Create a mock httpx Response for the ServiceUnavailableError
    mock_response = MagicMock()
    mock_response.request = MagicMock()
    mock_response.status_code = 503

    fake_message = Message(
        id="msg_test",
        type="message",
        role="assistant",
        content=[],
        model="anthropic.claude-haiku-5-5",
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=5, output_tokens=3, cache_creation_input_tokens=0, cache_read_input_tokens=0),
    )

    side_effects = [
        ServiceUnavailableError(
            "Error code: 503 - {'message': 'Bedrock is unable to process your request.'}",
            response=mock_response,
            body={"message": "Bedrock is unable to process your request."},
        ),
        fake_message,
    ]

    class FakeMessages:
        def __init__(self, region: str):
            self.region = region

        async def create(self, *args, **kwargs):
            call_regions.append(self.region)
            value = side_effects.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

    class FakeAsyncBedrock(AsyncAnthropicBedrock):
        def __init__(self, aws_region: str, **kwargs):
            # bypass real init
            self.aws_region = aws_region
            self.messages = FakeMessages(aws_region)

    # swap the SDK client with our fake
    monkeypatch.setattr("app.bedrock.bedrock_types.AsyncAnthropicBedrock", FakeAsyncBedrock)

    bedrock = BedrockHandler(mode=RunMode.ASYNC)
    result = await bedrock.invoke_async([{"role": "user", "content": "hello"}])

    assert result.model == "anthropic.claude-haiku-5-5"
    assert call_regions == [AWS_BEDROCK_REGION1, AWS_BEDROCK_REGION2]


@patch("app.bedrock.bedrock.BedrockHandler._invoke_async")
async def test_aws_region_failover_for_llm_invoke_success(mock_invoke, caplog):
    bedrock = BedrockHandler(mode=RunMode.ASYNC)
    messages = [{"role": "user", "content": "hello"}]
    # Create a mock Message object as the successful return value
    mock_message = Message(
        id="msg_test",
        type="message",
        role="assistant",
        content=[{"type": "text", "text": "Hello!"}],
        model="claude-test",
        stop_reason="end_turn",
        usage={"input_tokens": 10, "output_tokens": 5},
    )
    mock_invoke.side_effect = [_status_error(anthropic.RateLimitError, 429, "fail"), mock_message]
    result = await bedrock.invoke_async(messages)
    assert isinstance(result, Message)
    assert "Error in bedrock handler: fail, swapping" in caplog.text


@patch("app.bedrock.bedrock.BedrockHandler._invoke_async_with_call_cost_details")
async def test_aws_region_failover_for_llm_invoke_async_success(mock_invoke, caplog):
    bedrock = BedrockHandler(mode=RunMode.ASYNC)
    messages = [{"role": "user", "content": "hello"}]
    # Create a mock LLMTransaction object as the successful return value
    mock_transaction = LLMTransaction(
        content="Hello!", input_tokens=10, output_tokens=5, input_cost=0.001, output_cost=0.002, completion_cost=0.003
    )
    mock_invoke.side_effect = [_status_error(anthropic.InternalServerError, 500, "fail"), mock_transaction]
    result = await bedrock.invoke_async_with_call_cost_details(messages)
    assert isinstance(result, LLMTransaction)
    assert "Error in bedrock handler: fail, swapping" in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        _status_error(anthropic.RateLimitError, 429),
        _status_error(anthropic.InternalServerError, 500),
        _status_error(ServiceUnavailableError, 503),
        anthropic.APIConnectionError(request=_REQUEST),
        anthropic.APITimeoutError(request=_REQUEST),
        httpx.ConnectError("refused"),
    ],
)
@patch("app.bedrock.bedrock.BedrockHandler._invoke_async")
async def test_transient_errors_are_retried(mock_invoke, error):
    mock_invoke.side_effect = [error, _ok_message()]

    result = await BedrockHandler(mode=RunMode.ASYNC).invoke_async([{"role": "user", "content": "hello"}])

    assert result.id == "msg_ok"
    assert mock_invoke.call_count == 2


@pytest.mark.parametrize(
    "error",
    [
        _status_error(anthropic.BadRequestError, 400),
        _status_error(anthropic.AuthenticationError, 401),
        _status_error(anthropic.PermissionDeniedError, 403),
        ValueError("bug in the caller"),
    ],
)
@patch("app.bedrock.bedrock.BedrockHandler._invoke_async")
async def test_deterministic_errors_are_not_retried(mock_invoke, error):
    mock_invoke.side_effect = [error, _ok_message()]

    with pytest.raises(BedrockError) as exc_info:
        await BedrockHandler(mode=RunMode.ASYNC).invoke_async([{"role": "user", "content": "hello"}])

    assert exc_info.value.__cause__ is error
    assert exc_info.value.error_type is None
    assert mock_invoke.call_count == 1


@pytest.mark.parametrize(
    "message",
    [
        "validationException: Input is too long for requested model.",
        "validationException: input length and `max_tokens` exceed context limit: 200387 + 8192 > 204698",
    ],
)
@patch("app.bedrock.bedrock.BedrockHandler._invoke_async")
async def test_input_too_long_is_mapped_and_not_retried(mock_invoke, message):
    mock_invoke.side_effect = _status_error(anthropic.BadRequestError, 400, message)

    with pytest.raises(BedrockError) as exc_info:
        await BedrockHandler(mode=RunMode.ASYNC).invoke_async([{"role": "user", "content": "hello"}])

    assert exc_info.value.error_type == BedrockErrorType.INPUT_TOO_LONG
    assert mock_invoke.call_count == 1


@patch("app.bedrock.retry.AWS_BEDROCK_REGIONS_MAX_RETRIES", 2)
@patch("app.bedrock.bedrock.BedrockHandler._invoke_async")
async def test_final_failure_is_logged_with_its_stack(mock_invoke, caplog):
    error = _status_error(anthropic.InternalServerError, 500, "still down")
    mock_invoke.side_effect = error

    with caplog.at_level(logging.ERROR), pytest.raises(BedrockError) as exc_info:
        await BedrockHandler(mode=RunMode.ASYNC).invoke_async([{"role": "user", "content": "hello"}])

    assert mock_invoke.call_count == 3
    assert exc_info.value.__cause__ is error
    [record] = [r for r in caplog.records if r.getMessage().startswith("AWS Bedrock call failed")]
    assert record.exc_info[1] is error
    assert "after 3 attempt(s)" in record.getMessage()


@patch("app.bedrock.retry.AWS_BEDROCK_REGIONS_MAX_RETRIES", 2)
@patch("app.bedrock.bedrock.BedrockHandler._invoke_async")
async def test_final_timeout_is_mapped_to_timeout(mock_invoke):
    mock_invoke.side_effect = anthropic.APITimeoutError(request=_REQUEST)

    with pytest.raises(BedrockError) as exc_info:
        await BedrockHandler(mode=RunMode.ASYNC).invoke_async([{"role": "user", "content": "hello"}])

    assert exc_info.value.error_type == BedrockErrorType.TIMEOUT
    assert mock_invoke.call_count == 3


@patch("app.bedrock.retry.BEDROCK_RETRY_BUDGET_SECONDS", 0.05)
@patch("app.bedrock.bedrock.BedrockHandler._invoke_async")
async def test_hung_call_is_cut_off_by_the_budget(mock_invoke):
    async def hang(*args, **kwargs):
        await asyncio.sleep(10)

    mock_invoke.side_effect = hang

    with pytest.raises(BedrockError) as exc_info:
        await BedrockHandler(mode=RunMode.ASYNC).invoke_async([{"role": "user", "content": "hello"}])

    assert exc_info.value.error_type == BedrockErrorType.TIMEOUT
    assert mock_invoke.call_count == 1


@patch("app.bedrock.retry.BEDROCK_RETRY_BACKOFF_BASE_SECONDS", 0.5)
@patch("app.bedrock.retry.BEDROCK_RETRY_BACKOFF_MAX_SECONDS", 4.0)
def test_backoff_is_jittered_and_capped():
    for attempt, ceiling in [(0, 0.5), (1, 1.0), (2, 2.0), (3, 4.0), (10, 4.0)]:
        delays = [_backoff_seconds(attempt) for _ in range(200)]
        assert all(0 <= d <= ceiling for d in delays)
        assert len(set(delays)) > 1


def test_retry_budget_must_fit_inside_request_timeout():
    from app.config import AppSettings

    with pytest.raises(ValueError, match="bedrock_retry_budget_seconds"):
        AppSettings(request_timeout_seconds=60, bedrock_retry_budget_seconds=60)
