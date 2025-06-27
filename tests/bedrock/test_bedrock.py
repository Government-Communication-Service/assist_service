import asyncio
import json
import logging
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from app.api import ENDPOINTS
from app.bedrock import BedrockHandler, RunMode
from app.bedrock.bedrock_types import BedrockError
from app.bedrock.service import calculate_completion_cost
from app.database.models import LLM

api = ENDPOINTS()
logger = logging.getLogger(__name__)


def test_bedrock_service_with_cross_region_inference_with_selected_llm_model():
    llm = MagicMock()
    llm.model = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    bedrock = BedrockHandler(llm=llm)

    assert bedrock.model == "us.anthropic.claude-3-5-sonnet-20240620-v1:0"


def test_bedrock_service_with_no_cross_region_inference_with_selected_llm_model():
    llm = MagicMock()
    llm.model = "gpt-4o-2024-05-13"
    bedrock = BedrockHandler(llm=llm)

    assert bedrock.model == "gpt-4o-2024-05-13"


@patch("app.bedrock.bedrock.BedrockHandler._create_chat_title")
async def test_aws_region_failover_for_create_chat_title_success(mock_create_chat_title):
    bedrock = BedrockHandler(mode=RunMode.ASYNC)
    mock_create_chat_title.side_effect = [Exception("Transient error"), {"result": "success"}]
    title_message = {"role": "user", "content": "content"}
    result = await bedrock.create_chat_title([title_message])
    assert result == {"result": "success"}


@patch("app.bedrock.bedrock.BedrockHandler._create_chat_title")
async def test_aws_region_failover_for_create_chat_title_fail(mock_create_chat_title):
    bedrock = BedrockHandler(mode=RunMode.ASYNC)
    mock_create_chat_title.side_effect = Exception("Transient error1")
    title_message = {"role": "user", "content": "content"}
    with pytest.raises(BedrockError, match="Transient error1"):
        await bedrock.create_chat_title([title_message])


@patch("app.bedrock.bedrock.BedrockHandler._invoke_async")
async def test_aws_region_failover_for_llm_invoke_success(mock_invoke):
    bedrock = BedrockHandler(mode=RunMode.ASYNC)
    mock_invoke.side_effect = [Exception("Transient error invoke"), {"result": "success"}]
    messages = [{"role": "user", "content": "content"}]
    result = await bedrock.invoke_async(messages)
    assert result == {"result": "success"}


@patch("app.bedrock.bedrock.BedrockHandler._invoke_async")
async def test_aws_region_failover_for_llm_invoke_fail(mock_invoke, caplog):
    bedrock = BedrockHandler(mode=RunMode.ASYNC)
    mock_invoke.side_effect = Exception("Transient error invoke")

    messages = [{"role": "user", "content": "content"}]

    with pytest.raises(Exception, match="Transient error invoke"):
        await bedrock.invoke_async(messages)

    assert "Transient error invoke" in caplog.text


@pytest.mark.asyncio
@patch("app.bedrock.bedrock.BedrockHandler._invoke_async_with_call_cost_details")
async def test_aws_region_failover_for_llm_invoke_async_success(mock_invoke):
    bedrock = BedrockHandler(mode=RunMode.ASYNC)
    mock_invoke.side_effect = [Exception("Transient error invoke"), {"result": "success"}]
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


@patch("app.bedrock.bedrock.BedrockHandler._stream")
async def test_aws_region_failover_llm_stream_fail(mock_bedrock_stream, caplog):
    bedrock = BedrockHandler(mode=RunMode.ASYNC)
    mock_bedrock_stream.side_effect = Exception("Transient error invoke")

    messages = [{"role": "user", "content": "content"}]
    system_message = "test system message"

    def _error(ex: Exception):
        return f"{ex}"

    result = bedrock.stream(messages, system=system_message, on_error=_error)

    assert await anext(result) == "Transient error invoke"
    assert "Transient error invoke" in caplog.text


@patch("app.bedrock.bedrock.BedrockHandler._stream")
async def test_aws_region_failover_llm_stream_success(mock_bedrock_stream):
    bedrock = BedrockHandler(mode=RunMode.ASYNC)

    # Creating an async generator to simulate the stream
    async def mock_stream_generator(messages, system, on_error, other_param):
        yield {"result": "success"}

    mock_bedrock_stream.side_effect = [
        Exception("Transient error invoke"),
        mock_stream_generator(None, None, None, None),
    ]
    messages = [{"role": "user", "content": "content"}]
    system_message = "test system message"

    def _error(ex: Exception):
        return f"{ex}"

    stream_result = bedrock.stream(messages, system=system_message, on_error=_error)
    assert await anext(stream_result) == {"result": "success"}


@pytest.mark.asyncio
@patch("app.bedrock.bedrock.BedrockHandler._stream")
async def test_chat_returns_bedrock_error(mock_bedrock, async_client, user_id, async_http_requester, caplog):
    """
    Simulates and  tests AWS Bedrock error handled during a new chat request and Bedrock error returned as
    custom json message
    """

    async def _stream(*args, **kwargs):
        yield json.dumps({"result": "success"})
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
    assert '{"result": "success"}' in text_response
    assert '"error_code": "BEDROCK_SERVICE_ERROR", "error_message": "Some Bedrock error"' in text_response

    # check error is logged
    assert "AWS Bedrock error through streaming, exception: Some Bedrock error" in caplog.text


@pytest.mark.asyncio
@patch("app.bedrock.bedrock.BedrockHandler._stream")
async def test_chat_add_message_returns_bedrock_error(
    mock_bedrock, chat, async_client, user_id, async_http_requester, caplog
):
    """
    Simulates and  tests AWS Bedrock error handled in a chat  and Bedrock error returned as
    http status code 200 and payload contains below json structure.

    """

    async def _stream(*args, **kwargs):
        yield json.dumps({"result": "success"})
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
    assert '{"result": "success"}' in text_response
    assert '"error_code": "BEDROCK_SERVICE_ERROR", "error_message": "Some Bedrock error"' in text_response

    # check error is logged
    assert "AWS Bedrock error through streaming, exception: Some Bedrock error" in caplog.text


@pytest.mark.asyncio
@patch("app.bedrock.bedrock.BedrockHandler._stream")
async def test_slow_streaming_process_does_not_block_fastapi(
    mock_bedrock, chat, async_client, user_id, async_http_requester, caplog
):
    """
    Simulates a slow streaming response and checks in  other requests that FastAPI is not blocked.
    """

    async def _stream(*args, **kwargs):
        await asyncio.sleep(10)
        yield json.dumps({"result": "success"})

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
    # expect a string response
    assert results[0] == b'{"result": "success"}'
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
        model="anthropic.claude-sonnet-4-20250514-v1:0",
        provider="bedrock",
        input_cost_per_token=3e-06,
        output_cost_per_token=1.5e-05,
        max_tokens=8000,
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
