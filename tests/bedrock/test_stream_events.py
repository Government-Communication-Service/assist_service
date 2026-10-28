"""Tests for per-call streaming failover and the streaming British spelling helper."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.bedrock import BedrockHandler
from app.bedrock.bedrock_stream import _cache_prefix_key
from app.bedrock.bedrock_types import BedrockError, BedrockErrorType
from app.bedrock.retry import stream_with_region_failover
from app.chat.service import chat_stream_error_message
from app.config import AWS_BEDROCK_REGION1, AWS_BEDROCK_REGION2


def _handler():
    return SimpleNamespace(async_client=SimpleNamespace(aws_region=AWS_BEDROCK_REGION1))


def _provider(region):
    return SimpleNamespace(aws_region=region)


async def _collect(gen):
    return [item async for item in gen]


@patch("app.bedrock.retry.AsyncAnthropicBedrockProvider.get", side_effect=_provider)
async def test_retries_in_other_region_before_first_event(_mock_get):
    regions = []

    async def stream_func(client, **kwargs):
        regions.append(client.aws_region)
        if len(regions) == 1:
            raise RuntimeError("region down")
        yield "first"
        yield "second"

    handler = _handler()
    events = await _collect(stream_with_region_failover(handler, stream_func))

    assert events == ["first", "second"]
    assert regions == [AWS_BEDROCK_REGION1, AWS_BEDROCK_REGION2]


@patch("app.bedrock.retry.AsyncAnthropicBedrockProvider.get", side_effect=_provider)
async def test_error_after_first_event_is_raised_not_retried(_mock_get):
    calls = []

    async def stream_func(client, **kwargs):
        calls.append(client.aws_region)
        yield "first"
        raise RuntimeError("mid-stream")

    received = []
    with pytest.raises(RuntimeError, match="mid-stream"):
        async for event in stream_with_region_failover(_handler(), stream_func):
            received.append(event)

    assert received == ["first"]
    assert len(calls) == 1


@patch("app.bedrock.retry.AsyncAnthropicBedrockProvider.get", side_effect=_provider)
async def test_input_too_long_is_not_retried(_mock_get):
    calls = []

    async def stream_func(client, **kwargs):
        calls.append(client.aws_region)
        raise ValueError("validationException: Input is too long for requested model.")
        yield  # pragma: no cover

    with pytest.raises(BedrockError) as exc_info:
        await _collect(stream_with_region_failover(_handler(), stream_func))

    assert exc_info.value.error_type == BedrockErrorType.INPUT_TOO_LONG
    assert len(calls) == 1


@patch("app.bedrock.retry.AsyncAnthropicBedrockProvider.get", side_effect=_provider)
@patch("app.bedrock.retry.AWS_BEDROCK_REGIONS_MAX_RETRIES", 1)
async def test_raises_bedrock_error_when_all_attempts_fail(_mock_get):
    async def stream_func(client, **kwargs):
        raise RuntimeError("down")
        yield  # pragma: no cover

    with pytest.raises(BedrockError):
        await _collect(stream_with_region_failover(_handler(), stream_func))


@patch("app.bedrock.retry.AsyncAnthropicBedrockProvider.get", side_effect=_provider)
@patch("app.bedrock.retry.STREAM_FIRST_CHUNK_TIMEOUT", 0.01)
async def test_first_event_timeout_fails_over(_mock_get):
    regions = []

    async def stream_func(client, **kwargs):
        regions.append(client.aws_region)
        if len(regions) == 1:
            await asyncio.sleep(1)
        yield "ok"

    events = await _collect(stream_with_region_failover(_handler(), stream_func))

    assert events == ["ok"]
    assert regions == [AWS_BEDROCK_REGION1, AWS_BEDROCK_REGION2]


INPUT_TOO_LONG_ERRORS = [
    "validationException: Input is too long for requested model.",
    "validationException: input length and `max_tokens` exceed context limit: 200387 + 8192 > 204698",
]


@pytest.mark.parametrize("message", INPUT_TOO_LONG_ERRORS)
@patch("app.bedrock.retry.AsyncAnthropicBedrockProvider.get", side_effect=_provider)
async def test_both_input_too_long_errors_are_recognised(_mock_get, message):
    async def stream_func(client, **kwargs):
        raise ValueError(message)
        yield  # pragma: no cover

    with pytest.raises(BedrockError) as exc_info:
        await _collect(stream_with_region_failover(_handler(), stream_func))

    assert exc_info.value.error_type == BedrockErrorType.INPUT_TOO_LONG


@pytest.mark.parametrize(
    ("has_documents", "is_initial_call", "expected"),
    [
        (True, True, "Input is too long, too many documents selected, select fewer documents"),
        (False, True, "Input is too long, reduce input text"),
        (False, False, "Input is too long, reduce input text or start a new chat with reduced input text"),
    ],
)
def test_input_too_long_stream_error_messages(has_documents, is_initial_call, expected):
    chat = MagicMock()
    chat.client_response.return_value = {}
    error = BedrockError("Input is too long", BedrockErrorType.INPUT_TOO_LONG)

    body = json.loads(
        chat_stream_error_message(chat, error, has_documents=has_documents, is_initial_call=is_initial_call)
    )

    assert body == {"error_code": "BEDROCK_SERVICE_INPUT_TOO_LONG_ERROR", "error_message": expected}


def test_cache_prefix_key_stable_for_identical_system_and_tools():
    system = [{"type": "text", "text": "prompt"}]
    tools = [{"name": "search"}]

    assert _cache_prefix_key(system, tools) == _cache_prefix_key(system, tools)


@pytest.mark.parametrize(
    ("other_system", "other_tools"),
    [
        ([{"type": "text", "text": "different prompt"}], [{"name": "search"}]),
        ([{"type": "text", "text": "prompt"}], [{"name": "search"}, {"name": "another"}]),
        ([{"type": "text", "text": "prompt"}], None),
    ],
)
def test_cache_prefix_key_changes_with_system_or_tools(other_system, other_tools):
    system = [{"type": "text", "text": "prompt"}]
    tools = [{"name": "search"}]

    assert _cache_prefix_key(system, tools) != _cache_prefix_key(other_system, other_tools)


async def test_stream_events_passes_handler_config_and_request_params():
    handler = SimpleNamespace(config={"model": "m", "max_tokens": 10, "system": "s"})

    async def no_events(*args, **kwargs):
        return
        yield

    with patch("app.bedrock.bedrock.stream_with_region_failover", side_effect=no_events) as failover:
        async for _ in BedrockHandler.stream_events(
            handler, [{"role": "user", "content": "hi"}], output_config={"effort": "high"}
        ):
            pass

    kwargs = failover.call_args.kwargs
    assert kwargs == {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "m",
        "max_tokens": 10,
        "system": "s",
        "output_config": {"effort": "high"},
    }
