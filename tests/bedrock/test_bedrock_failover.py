from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from anthropic._exceptions import ServiceUnavailableError
from anthropic.types import Message

from app.bedrock import BedrockHandler, RunMode
from app.bedrock.bedrock_types import AsyncAnthropicBedrock, AsyncAnthropicBedrockProvider
from app.bedrock.schemas import LLMTransaction
from app.chat.service import chat_stream_error_message
from app.config import AWS_BEDROCK_REGION1, AWS_BEDROCK_REGION2

bedrock = BedrockHandler()
bedrock_async = BedrockHandler(mode=RunMode.ASYNC)
original_create_chat_title_function = bedrock_async._create_chat_title
original_invoke_async_function = bedrock_async._invoke_async
original_invoke_async_with_call_cost_details_function = bedrock_async._invoke_async_with_call_cost_details
original_bedrock_stream_function = bedrock_async._stream


@patch("app.bedrock.bedrock_types.AsyncAnthropicBedrockProvider.get")
@patch("app.bedrock.bedrock.BedrockHandler._invoke_async")
async def test_failover_switches_from_west_to_east(mock_invoke_async, mock_provider_get):
    """Simulate us-west-2 outage and verify we switch to us-east-1 and recover."""

    # first provider call during handler init returns a west client
    # second call (after failover) should request the east client
    def _provider_side_effect(region):
        return SimpleNamespace(aws_region=region)

    mock_provider_get.side_effect = _provider_side_effect
    mock_invoke_async.side_effect = [Exception("west-down"), {"result": "ok"}]

    bedrock = BedrockHandler(mode=RunMode.ASYNC)

    result = await bedrock.invoke_async([{"role": "user", "content": "hello"}])

    assert result == {"result": "ok"}
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

    side_effects = [
        ServiceUnavailableError(
            "Error code: 503 - {'message': 'Bedrock is unable to process your request.'}",
            response=mock_response,
            body={"message": "Bedrock is unable to process your request."},
        ),
        {"result": "ok"},
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

    assert result == {"result": "ok"}
    assert call_regions == [AWS_BEDROCK_REGION1, AWS_BEDROCK_REGION2]


async def test_failover_streams_before_first_chunk(monkeypatch):
    """Fail once before streaming starts; ensure retry switches to east and completes."""

    AsyncAnthropicBedrockProvider._AsyncAnthropicBedrockProvider__clients.clear()

    stream_regions: list[str] = []

    class FakeStreamCtx:
        def __init__(self, region: str, attempt: int):
            self.region = region
            self.attempt = attempt

        async def __aenter__(self):
            if self.region == AWS_BEDROCK_REGION1 and self.attempt == 0:
                raise Exception("west-stream-down")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        @property
        def text_stream(self):
            async def gen():
                yield "Hi"

            return gen()

        async def get_final_message(self):
            return {"result": "ok"}

    class FakeMessages:
        def __init__(self, region: str):
            self.region = region
            self.calls = 0

        def stream(self, *args, **kwargs):
            ctx = FakeStreamCtx(self.region, self.calls)
            stream_regions.append(self.region)
            self.calls += 1
            return ctx

    class FakeAsyncBedrock(AsyncAnthropicBedrock):
        def __init__(self, aws_region: str, **kwargs):
            self.aws_region = aws_region
            self.messages = FakeMessages(aws_region)

    monkeypatch.setattr("app.bedrock.bedrock_types.AsyncAnthropicBedrock", FakeAsyncBedrock)

    bedrock = BedrockHandler(mode=RunMode.ASYNC)

    def _error(ex: Exception):
        return f"{ex}"

    stream_result = bedrock.stream([{"role": "user", "content": "hello"}], system="", on_error=_error)
    collected = ""
    async for chunk in stream_result:
        collected += chunk

    assert collected == "Hi"
    assert stream_regions == [AWS_BEDROCK_REGION1, AWS_BEDROCK_REGION2]


async def test_failover_on_stream_timeout_no_first_chunk(monkeypatch):
    """Test that streaming failover triggers when first chunk times out (stream hangs with no data)."""
    import asyncio

    AsyncAnthropicBedrockProvider._AsyncAnthropicBedrockProvider__clients.clear()

    # Use a short timeout for testing to keep tests fast
    test_timeout = 0.5
    monkeypatch.setattr("app.bedrock.retry.STREAM_FIRST_CHUNK_TIMEOUT", test_timeout)

    # Use delay slightly longer than configured timeout to ensure timeout triggers
    test_delay = test_timeout + 0.5

    stream_regions: list[str] = []

    class FakeStreamCtx:
        def __init__(self, region: str):
            self.region = region

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        @property
        def text_stream(self):
            async def gen():
                if self.region == AWS_BEDROCK_REGION1:
                    # Simulate hanging stream - wait longer than timeout
                    await asyncio.sleep(test_delay)
                yield "Hi"

            return gen()

        async def get_final_message(self):
            return {"result": "ok"}

    class FakeMessages:
        def __init__(self, region: str):
            self.region = region

        def stream(self, *args, **kwargs):
            ctx = FakeStreamCtx(self.region)
            stream_regions.append(self.region)
            return ctx

    class FakeAsyncBedrock(AsyncAnthropicBedrock):
        def __init__(self, aws_region: str, **kwargs):
            self.aws_region = aws_region
            self.messages = FakeMessages(aws_region)

    monkeypatch.setattr("app.bedrock.bedrock_types.AsyncAnthropicBedrock", FakeAsyncBedrock)

    bedrock = BedrockHandler(mode=RunMode.ASYNC)

    def _error(ex: Exception):
        return f"{ex}"

    stream_result = bedrock.stream([{"role": "user", "content": "hello"}], system="", on_error=_error)
    collected = ""
    async for chunk in stream_result:
        collected += chunk

    assert collected == "Hi"
    # Should have tried West (timed out), then East (succeeded)
    assert stream_regions == [AWS_BEDROCK_REGION1, AWS_BEDROCK_REGION2]


@patch("app.bedrock.bedrock.BedrockHandler._create_chat_title")
async def test_aws_region_failover_for_create_chat_title_success(mock_create_chat_title, caplog):
    title_message = {"role": "user", "content": "hello"}
    params = [title_message]
    mock_create_chat_title.side_effect = [Exception("fail"), await original_create_chat_title_function(params)]
    llm_transaction = await bedrock_async.create_chat_title(params)
    assert isinstance(llm_transaction, LLMTransaction)
    assert "Error in bedrock handler: fail, swapping" in caplog.text


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
        usage={"input_tokens": 10, "output_tokens": 5}
    )
    mock_invoke.side_effect = [Exception("fail"), mock_message]
    result = await bedrock.invoke_async(messages)
    assert isinstance(result, Message)
    assert "Error in bedrock handler: fail, swapping" in caplog.text


@patch("app.bedrock.bedrock.BedrockHandler._invoke_async_with_call_cost_details")
async def test_aws_region_failover_for_llm_invoke_async_success(mock_invoke, caplog):
    bedrock = BedrockHandler(mode=RunMode.ASYNC)
    messages = [{"role": "user", "content": "hello"}]
    # Create a mock LLMTransaction object as the successful return value
    mock_transaction = LLMTransaction(
        content="Hello!",
        input_tokens=10,
        output_tokens=5,
        input_cost=0.001,
        output_cost=0.002,
        completion_cost=0.003
    )
    mock_invoke.side_effect = [Exception("fail"), mock_transaction]
    result = await bedrock.invoke_async_with_call_cost_details(messages)
    assert isinstance(result, LLMTransaction)
    assert "Error in bedrock handler: fail, swapping" in caplog.text


@patch("app.bedrock.bedrock.BedrockHandler._stream")
async def test_aws_region_failover_llm_stream_success(mock_bedrock_stream, caplog):
    bedrock = BedrockHandler(mode=RunMode.ASYNC)
    messages = [{"role": "user", "content": "hello"}]
    system_message = "Respond only with Hi"

    mock_bedrock_stream.side_effect = [
        Exception("fail"),
        original_bedrock_stream_function(messages, system=system_message),
    ]

    def _error(ex: Exception):
        return f"{ex}"

    stream_result = bedrock.stream(messages, system=system_message, on_error=_error)
    result = ""
    async for t in stream_result:
        result += t

    assert result == "Hi"
    assert "Error in bedrock handler: fail, swapping" in caplog.text


@patch("app.bedrock.bedrock.BedrockHandler._stream")
async def test_skip_failover_when_documents_present_and_input_too_long(mock_bedrock_stream):
    bedrock = BedrockHandler(mode=RunMode.ASYNC)
    messages = [{"role": "user", "content": "hello"}]
    system_message = "Respond only with Hi"
    mock_bedrock_stream.side_effect = [
        ValueError("""Bad response code, expected 200:
        {'status_code': 400,
        'headers': {':exception-type': 'validationException'},
        'body': b'{"message":"Input is too long for requested model."}'}""")
    ]
    chat = MagicMock()

    def _error_function(ex):
        return chat_stream_error_message(chat, ex, has_documents=True, is_initial_call=True)

    stream_result = bedrock.stream(messages, system=system_message, on_error=_error_function)
    expexted_result = [
        '{"error_code": "BEDROCK_SERVICE_INPUT_TOO_LONG_ERROR", '
        '"error_message": "Input is too long, too many documents selected, select fewer documents"}'
    ]

    stream_result = await anext(stream_result)
    assert [stream_result] == expexted_result


@patch("app.bedrock.bedrock.BedrockHandler._stream")
async def test_skip_failover_when_context_limit_reached(mock_bedrock_stream):
    bedrock = BedrockHandler(mode=RunMode.ASYNC)
    messages = [{"role": "user", "content": "hello"}]
    system_message = "Respond only with Hi"
    mock_bedrock_stream.side_effect = [
        ValueError("""Bad response code, expected 200:
        {'status_code': 400,
        'headers': {':exception-type': 'validationException'},
        'body': b'{"input length and `max_tokens` exceed context limit: 200387 + 8192 > 204698,
        decrease input length or `max_tokens` and try again"}'}""")
    ]
    chat = MagicMock()

    def _error_function(ex):
        return chat_stream_error_message(chat, ex, has_documents=False, is_initial_call=False)

    stream_result = bedrock.stream(messages, system=system_message, on_error=_error_function)
    expexted_result = [
        '{"error_code": "BEDROCK_SERVICE_INPUT_TOO_LONG_ERROR", '
        '"error_message": "Input is too long, reduce input text or start a new chat with reduced input text"}'
    ]

    stream_result = await anext(stream_result)
    assert [stream_result] == expexted_result


@patch("app.bedrock.bedrock.BedrockHandler._stream")
async def test_skip_failover_when_documents_not_present_and_input_too_long(mock_bedrock_stream):
    bedrock = BedrockHandler(mode=RunMode.ASYNC)
    messages = [{"role": "user", "content": "hello"}]
    system_message = "Respond only with Hi"
    mock_bedrock_stream.side_effect = [
        ValueError("""Bad response code, expected 200:
        {'status_code': 400,
        'headers': {':exception-type': 'validationException'},
        'body': b'{"message":"Input is too long for requested model."}'}""")
    ]
    chat = MagicMock()

    def _error_function(ex):
        return chat_stream_error_message(chat, ex, has_documents=False, is_initial_call=True)

    stream_result = bedrock.stream(messages, system=system_message, on_error=_error_function)
    expexted_result = [
        '{"error_code": "BEDROCK_SERVICE_INPUT_TOO_LONG_ERROR", '
        '"error_message": "Input is too long, reduce input text"}'
    ]

    stream_result = await anext(stream_result)
    assert [stream_result] == expexted_result
