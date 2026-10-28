import asyncio
import logging
from contextlib import asynccontextmanager
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import pytest
from anthropic.types.message import Message as AnthropicMessage
from anthropic.types.usage import Usage

from app.api.endpoints import ENDPOINTS
from app.bedrock import BedrockHandler, BedrockMessage, RunMode
from app.bedrock.bedrock_stream import StreamDone, StreamStarted, StreamText
from app.bedrock.bedrock_types import BedrockError
from app.bedrock.call_log import LlmCallContext
from app.bedrock.service import calculate_completion_cost
from app.config import AWS_BEDROCK_REGION1, LLM_CHAT_RESPONSE_MODEL, LLM_DEFAULT_MODEL
from app.database.models import LLM

api = ENDPOINTS()
logger = logging.getLogger(__name__)


def _transient_error():
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://bedrock-runtime.example"))


@asynccontextmanager
async def fake_db_session():
    yield MagicMock()


def test_bedrock_service_with_cross_region_inference_with_selected_llm_model():
    llm = MagicMock()
    llm.model = LLM_CHAT_RESPONSE_MODEL
    bedrock = BedrockHandler(llm=llm)

    expected_prefix = AWS_BEDROCK_REGION1.split("-")[0]
    assert bedrock.model == f"{expected_prefix}.{LLM_CHAT_RESPONSE_MODEL}"


def test_bedrock_service_with_no_cross_region_inference_with_selected_llm_model():
    llm = MagicMock()
    llm.model = "gpt-4o-2024-05-13"
    bedrock = BedrockHandler(llm=llm)

    assert bedrock.model == "gpt-4o-2024-05-13"


@patch("app.bedrock.bedrock.BedrockHandler._invoke_async")
async def test_aws_region_failover_for_llm_invoke_success(mock_invoke):
    bedrock = BedrockHandler(mode=RunMode.ASYNC)
    mock_invoke.side_effect = [_transient_error(), make_anthropic_message()]
    messages = [{"role": "user", "content": "content"}]
    result = await bedrock.invoke_async(messages)
    assert result.id == "msg_test"


@patch("app.bedrock.bedrock.BedrockHandler._invoke_async")
async def test_aws_region_failover_for_llm_invoke_fail(mock_invoke, caplog):
    bedrock = BedrockHandler(mode=RunMode.ASYNC)
    mock_invoke.side_effect = Exception("Transient error invoke")

    messages = [{"role": "user", "content": "content"}]

    with pytest.raises(Exception, match="Transient error invoke"):
        await bedrock.invoke_async(messages)

    assert "Transient error invoke" in caplog.text


def make_anthropic_message(model="anthropic.claude-haiku-5-5"):
    return AnthropicMessage(
        id="msg_test",
        type="message",
        role="assistant",
        content=[],
        model=model,
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=10, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=0),
    )


async def test_invoke_async_returns_bedrock_message():
    """invoke_async wraps the SDK response in a BedrockMessage."""
    llm = MagicMock()
    llm.model = LLM_DEFAULT_MODEL
    bedrock = BedrockHandler(llm=llm, mode=RunMode.ASYNC)

    fake_response = make_anthropic_message(model=bedrock.model)
    bedrock.async_client = MagicMock()
    bedrock.async_client.messages.create = AsyncMock(return_value=fake_response)

    with patch("app.bedrock.bedrock.log_llm_call", new_callable=AsyncMock, return_value=None):
        result = await bedrock.invoke_async([{"role": "user", "content": "hello"}])

    assert isinstance(result, BedrockMessage)
    assert result.model == bedrock.model
    assert result.llm_internal_response_id is None


async def test_invoke_async_logs_the_call_and_returns_its_id():
    """invoke_async logs the response with the caller's context and sets llm_internal_response_id."""
    llm = MagicMock()
    llm.model = LLM_DEFAULT_MODEL
    bedrock = BedrockHandler(llm=llm, mode=RunMode.ASYNC)

    fake_response = make_anthropic_message(model=bedrock.model)
    bedrock.async_client = MagicMock()
    bedrock.async_client.messages.create = AsyncMock(return_value=fake_response)
    context = LlmCallContext(purpose="test", chat_id=1, message_id=2)

    with patch("app.bedrock.bedrock.log_llm_call", new_callable=AsyncMock) as mock_log:
        mock_log.return_value = 42
        result = await bedrock.invoke_async([{"role": "user", "content": "hello"}], context=context)

    mock_log.assert_awaited_once_with(llm, fake_response, context)
    assert result.llm_internal_response_id == 42
    # The context is ours, not a Messages API param.
    assert "context" not in bedrock.async_client.messages.create.call_args.kwargs


async def test_invoke_async_logging_failure_neither_fails_nor_reruns_the_call():
    """A DB error while logging must not be mistaken for a Bedrock failure and retried (and billed) again."""
    llm = MagicMock()
    llm.model = LLM_DEFAULT_MODEL
    bedrock = BedrockHandler(llm=llm, mode=RunMode.ASYNC)

    fake_response = make_anthropic_message(model=bedrock.model)
    bedrock.async_client = MagicMock()
    bedrock.async_client.messages.create = AsyncMock(return_value=fake_response)

    patch_target = "app.bedrock.call_log.DbOperations.insert_llm_internal_response_id_query"
    with (
        patch("app.bedrock.call_log.async_db_session", fake_db_session),
        patch(patch_target, new_callable=AsyncMock, side_effect=RuntimeError("db down")),
    ):
        result = await bedrock.invoke_async(
            [{"role": "user", "content": "hello"}], context=LlmCallContext(purpose="test")
        )

    assert isinstance(result, BedrockMessage)
    assert result.llm_internal_response_id is None
    bedrock.async_client.messages.create.assert_awaited_once()


async def test_invoke_async_without_context_is_not_logged():
    llm = MagicMock()
    llm.model = LLM_DEFAULT_MODEL
    bedrock = BedrockHandler(llm=llm, mode=RunMode.ASYNC)
    bedrock.async_client = MagicMock()
    bedrock.async_client.messages.create = AsyncMock(return_value=make_anthropic_message(model=bedrock.model))

    with patch("app.bedrock.bedrock.log_llm_call", new_callable=AsyncMock) as mock_log:
        result = await bedrock.invoke_async([{"role": "user", "content": "hello"}])

    mock_log.assert_not_called()
    assert result.llm_internal_response_id is None


async def test_stream_events_without_context_is_not_logged():
    llm = MagicMock()
    llm.model = LLM_DEFAULT_MODEL
    bedrock = BedrockHandler(llm=llm, mode=RunMode.ASYNC)
    final = make_anthropic_message(model=bedrock.model)

    async def _stream(*args, **kwargs):
        yield StreamStarted()
        yield StreamDone(final)

    with (
        patch("app.bedrock.bedrock.stream_message_events", side_effect=_stream),
        patch("app.bedrock.bedrock.log_llm_call", new_callable=AsyncMock) as mock_log,
    ):
        events = [e async for e in bedrock.stream_events([{"role": "user", "content": "hello"}])]

    mock_log.assert_not_called()
    assert events[-1] == StreamDone(final)


async def test_stream_events_sets_the_log_row_id_on_stream_done():
    llm = MagicMock()
    llm.model = LLM_DEFAULT_MODEL
    bedrock = BedrockHandler(llm=llm, mode=RunMode.ASYNC)
    final = make_anthropic_message(model=bedrock.model)

    async def _stream(*args, **kwargs):
        yield StreamStarted()
        yield StreamText("hi")
        yield StreamDone(final)

    context = LlmCallContext(purpose="test")
    with (
        patch("app.bedrock.bedrock.stream_message_events", side_effect=_stream),
        patch("app.bedrock.bedrock.log_llm_call", new_callable=AsyncMock, return_value=7) as mock_log,
    ):
        events = [e async for e in bedrock.stream_events([{"role": "user", "content": "hello"}], context=context)]

    assert events[-1] == StreamDone(final, llm_internal_response_id=7)
    mock_log.assert_awaited_once_with(llm, final, context)


def test_format_response_extracts_cache_tokens():
    """format_response reads cache_read_input_tokens/cache_creation_input_tokens off response.usage."""
    llm = MagicMock()
    llm.model = LLM_DEFAULT_MODEL
    llm.input_cost_per_token = 1e-6
    llm.output_cost_per_token = 5e-6
    llm.cache_read_cost_multiplier = None
    bedrock = BedrockHandler(llm=llm)

    fake_response = AnthropicMessage(
        id="msg_test",
        type="message",
        role="assistant",
        content=[],
        model=bedrock.model,
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=10, output_tokens=5, cache_creation_input_tokens=7, cache_read_input_tokens=3),
    )

    transaction = bedrock.format_response(fake_response)

    assert transaction.cache_read_tokens == 3
    assert transaction.cache_write_tokens == 7


@pytest.mark.asyncio
@patch("app.bedrock.bedrock.BedrockHandler._invoke_async_with_call_cost_details")
async def test_aws_region_failover_for_llm_invoke_async_success(mock_invoke):
    bedrock = BedrockHandler(mode=RunMode.ASYNC)
    mock_invoke.side_effect = [_transient_error(), {"result": "success"}]
    messages = [{"role": "user", "content": "content"}]
    result = await bedrock.invoke_async_with_call_cost_details(messages)
    assert result == {"result": "success"}


@pytest.mark.asyncio
@patch("app.bedrock.bedrock.BedrockHandler._invoke_async_with_call_cost_details")
async def test_aws_region_failover_for_llm_invoke_async_fail(mock_invoke, caplog):
    bedrock = BedrockHandler(mode=RunMode.ASYNC)
    mock_invoke.side_effect = Exception("Transient error invoke")

    messages = [{"role": "user", "content": "content"}]

    with pytest.raises(BedrockError, match="Transient error invoke"):
        await bedrock.invoke_async_with_call_cost_details(messages)

    assert "Transient error invoke" in caplog.text


@pytest.mark.asyncio
@patch("app.bedrock.bedrock.stream_message_events")
async def test_chat_returns_bedrock_error(mock_bedrock, async_client, user_id, async_http_requester, caplog):
    """
    Simulates and  tests AWS Bedrock error handled during a new chat request and Bedrock error returned as
    custom json message
    """

    async def _stream(*args, **kwargs):
        yield StreamStarted()
        yield StreamText("partial answer")
        raise Exception("Some Bedrock error")

    mock_bedrock.side_effect = _stream

    url = api.create_chat_stream(user_uuid=user_id)
    response = await async_http_requester(
        "chat_endpoint_bedrock_error",
        async_client.post,
        url,
        response_type="text",
        response_code=200,
        json={"query": "hello"},
    )
    text_response = str(response)
    assert '"content": "partial answer"' in text_response
    assert '"error_code": "BEDROCK_SERVICE_ERROR", "error_message": "Some Bedrock error"' in text_response

    # check error is logged
    assert "Agent loop failed for chat" in caplog.text


@pytest.mark.asyncio
@patch("app.bedrock.bedrock.stream_message_events")
async def test_chat_add_message_returns_bedrock_error(
    mock_bedrock, chat, async_client, user_id, async_http_requester, caplog
):
    """
    Simulates and  tests AWS Bedrock error handled in a chat  and Bedrock error returned as
    http status code 200 and payload contains below json structure.

    """

    async def _stream(*args, **kwargs):
        yield StreamStarted()
        yield StreamText("partial answer")
        raise Exception("Some Bedrock error")

    mock_bedrock.side_effect = _stream
    url = api.get_chat_stream(user_uuid=user_id, chat_uuid=chat.uuid)
    response = await async_http_requester(
        "chat_add_message_endpoint_bedrock_error",
        async_client.put,
        url,
        response_code=200,
        response_type="text",
        json={"query": "hello"},
    )

    text_response = str(response)
    assert '"content": "partial answer"' in text_response
    assert '"error_code": "BEDROCK_SERVICE_ERROR", "error_message": "Some Bedrock error"' in text_response

    # check error is logged
    assert "Agent loop failed for chat" in caplog.text


@pytest.mark.asyncio
@patch("app.bedrock.bedrock.stream_message_events")
async def test_slow_streaming_process_does_not_block_fastapi(
    mock_bedrock, chat, async_client, user_id, async_http_requester, caplog
):
    """
    Simulates a slow streaming response and checks in other requests that FastAPI is not blocked.

    Note: First event is yielded immediately to satisfy the region failover first-event timeout
    (see STREAM_FIRST_CHUNK_TIMEOUT in app.bedrock.retry). The 10-second delay happens AFTER
    the first event, which still tests that slow streaming doesn't block FastAPI.
    """

    async def _stream(*args, **kwargs):
        yield StreamStarted()  # First event - satisfies failover timeout
        await asyncio.sleep(10)  # Slow streaming after first event
        yield StreamText("success")
        yield StreamDone(
            AnthropicMessage(
                id="msg",
                type="message",
                role="assistant",
                model="m",
                content=[{"type": "text", "text": "success"}],
                stop_reason="end_turn",
                stop_sequence=None,
                usage=Usage(input_tokens=1, output_tokens=1),
            )
        )

    mock_bedrock.side_effect = _stream
    url = api.get_chat_stream(user_uuid=user_id, chat_uuid=chat.uuid)
    chat_stream_request = async_http_requester(
        "chat_add_message_endpoint_slow_response_timeout",
        async_client.put,
        url,
        response_code=200,
        response_type="text",
        json={"query": "hello"},
    )

    # check health endpoint
    url = api.get_chat_item(user_uuid=user_id, chat_uuid=chat.uuid)
    chat_item_requests = [
        async_http_requester(
            "chat_add_message_endpoint_slow_response_timeout",
            async_client.get,
            url,
            response_code=200,
            response_type="json",
        )
        for _ in range(3)
    ]

    results = await asyncio.gather(chat_stream_request, *chat_item_requests, return_exceptions=True)
    # Response contains both chunks; verify final chunk is present
    assert b'"content": "success"' in results[0]
    # assert no exceptions
    assert all(not isinstance(r, Exception) for r in results)


@pytest.mark.asyncio
async def test_completion_cost_calculation():
    """
    Tests that the completion cost is returned accurately.
    Starts by creating an LLM instance
    Then uses the calculate_completion_cost_method to create calculation cost
    Then independently performs the calculation
    Then asserts that the values should be the same.

    Exact figures can be independently verified:
    https://yourgpt.ai/tools/openai-and-other-llm-api-pricing-calculator
    """
    llm = LLM(
        model=LLM_DEFAULT_MODEL,
        provider="bedrock",
        input_cost_per_token=3e-06,
        output_cost_per_token=1.5e-05,
        max_tokens=4000,
    )
    input_tokens = 1000
    output_tokens = 100
    calculated_cost = calculate_completion_cost(
        llm,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    independent_calculation = ((Decimal(str(llm.input_cost_per_token))) * input_tokens) + (
        (Decimal(str(llm.output_cost_per_token))) * output_tokens
    )
    logger.info(f"Production cost calculation = {calculated_cost}")
    logger.info(f"Independent cost calculation = {independent_calculation}")
    assert calculated_cost == independent_calculation, (
        "The completion cost of an LLM response was not calculated correctly."
    )
