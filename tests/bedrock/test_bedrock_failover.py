from unittest.mock import MagicMock, patch

from anthropic.types import Message

from app.bedrock import BedrockHandler, RunMode
from app.bedrock.schemas import LLMTransaction
from app.chat.service import chat_stream_error_message

bedrock = BedrockHandler()
bedrock_async = BedrockHandler(mode=RunMode.ASYNC)
original_create_chat_title_function = bedrock_async._create_chat_title
original_invoke_async_function = bedrock_async._invoke_async
original_invoke_async_with_call_cost_details_function = bedrock_async._invoke_async_with_call_cost_details
original_bedrock_stream_function = bedrock_async._stream


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
    mock_invoke.side_effect = [Exception("fail"), await original_invoke_async_function(messages)]
    result = await bedrock.invoke_async(messages)
    assert isinstance(result, Message)
    assert "Error in bedrock handler: fail, swapping" in caplog.text


@patch("app.bedrock.bedrock.BedrockHandler._invoke_async_with_call_cost_details")
async def test_aws_region_failover_for_llm_invoke_async_success(mock_invoke, caplog):
    bedrock = BedrockHandler(mode=RunMode.ASYNC)
    messages = [{"role": "user", "content": "hello"}]
    mock_invoke.side_effect = [Exception("fail"), await original_invoke_async_with_call_cost_details_function(messages)]
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
