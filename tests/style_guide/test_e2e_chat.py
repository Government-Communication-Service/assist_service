"""
End-to-end tests for the style guide checker in the chat flow, with the chat model scripted
and the checker itself mocked (its own behaviour is covered by the service tests).
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.api.endpoints import ENDPOINTS
from app.style_guide.tools import CHECK_STYLE_GUIDE

api = ENDPOINTS()

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.streaming,
]

CHECKED_TEXT = "The Prime Minister met with the Home Secretary today."


def parse_streaming_response(response_data: bytes) -> tuple[list[dict], dict]:
    """Return all JSON objects in a streamed response and its final message_streamed packet."""
    text = response_data.decode("utf-8").strip()
    decoder = json.JSONDecoder()
    json_objects, pos = [], 0
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        obj, pos = decoder.raw_decode(text, pos)
        json_objects.append(obj)

    message_streamed_packets = [p for p in json_objects if "message_streamed" in p]
    assert message_streamed_packets, "No message_streamed packets found in response"
    return json_objects, message_streamed_packets[-1]


def _tool_results(request: dict) -> list[dict]:
    return [block for block in request["messages"][-1]["content"] if block.get("type") == "tool_result"]


@pytest.fixture
def mock_style_guide_check():
    with patch("app.chat.tools.check_style_guide_for_tool", new_callable=AsyncMock) as check:
        check.return_value = ("<issues>Use 'meet', not 'meet with'.</issues>", False)
        yield check


async def _stream_style_guide_chat(async_client, user_id, default_headers, use_case_id) -> dict:
    # Connect sends a pre-built prompt as "{instruction}\n\n{user input}"
    response = await async_client.post(
        api.create_chat_stream(user_uuid=user_id),
        json={
            "query": f"Check the provided content against the GOV.UK Style Guide.\n\nContent: {CHECKED_TEXT}",
            "use_rag": False,
            "use_gov_uk_search_api": False,
            "use_case_id": use_case_id,
            "use_smart_targets": False,
        },
        headers=default_headers,
    )
    assert response.status_code == 200
    json_objects, final_packet = parse_streaming_response(response.content)
    assert json_objects[0].get("uuid") is not None
    return final_packet["message_streamed"]


@pytest.mark.asyncio
async def test_style_guide_check_result_is_given_to_the_model_and_cited(
    async_client, user_id, default_headers, style_guide_use_case_id, mock_chat_model, mock_style_guide_check
):
    mock_chat_model.script([(CHECK_STYLE_GUIDE.name, {"text": CHECKED_TEXT})], "One issue: use 'meet'.")

    message = await _stream_style_guide_chat(async_client, user_id, default_headers, style_guide_use_case_id)

    assert mock_style_guide_check.await_args.kwargs["text"] == CHECKED_TEXT
    [tool_result] = _tool_results(mock_chat_model.requests[1])
    assert "Use 'meet', not 'meet with'." in str(tool_result["content"])
    assert not tool_result.get("is_error")

    assert message["content"] == "One issue: use 'meet'."
    sources = json.loads(message["sources"]) if isinstance(message["sources"], str) else message["sources"]
    assert sources["style_guide_sources"][0]["pretty_name"] == "GOV.UK style guide"


@pytest.mark.asyncio
async def test_failed_style_guide_check_is_reported_to_the_model_as_an_error(
    async_client, user_id, default_headers, style_guide_use_case_id, mock_chat_model, mock_style_guide_check
):
    mock_style_guide_check.return_value = ("The style guide check failed.", True)
    mock_chat_model.script([(CHECK_STYLE_GUIDE.name, {"text": CHECKED_TEXT})], "Sorry, the check failed.")

    message = await _stream_style_guide_chat(async_client, user_id, default_headers, style_guide_use_case_id)

    [tool_result] = _tool_results(mock_chat_model.requests[1])
    assert tool_result["is_error"] is True
    assert message["content"] == "Sorry, the check failed."
