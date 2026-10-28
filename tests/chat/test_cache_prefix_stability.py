"""Guard tests for prompt-cache prefix stability across turns of one chat.

Per thoughts/2026-09-28-cache-stability-handoff.md item 5. These drive consecutive turns of one
chat through the API with `mock_chat_model` and inspect what was actually sent to the model:
`tools`, `system` and `messages` must stay stable for the prefix to stay cached, across checkbox
toggles, documents attached mid-chat, date rollovers and hitting the iteration cap.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from app.api.endpoints import ENDPOINTS
from app.config import settings

api = ENDPOINTS()

pytestmark = pytest.mark.usefixtures("mock_llm_response")


def _strip_cache_control(messages: list[dict]) -> list[dict]:
    def _block(block):
        return {k: v for k, v in block.items() if k != "cache_control"} if isinstance(block, dict) else block

    return [
        {**m, "content": [_block(b) for b in m["content"]]} if isinstance(m.get("content"), list) else m
        for m in messages
    ]


def _tool_names(request: dict) -> set[str]:
    return {tool["name"] for tool in request.get("tools", [])}


def _tool_call(name: str, args: dict) -> list[tuple[str, dict]]:
    return [(name, args)]


def _assert_messages_form_growing_prefixes(requests: list[dict]) -> None:
    stripped = [_strip_cache_control(r["messages"]) for r in requests]
    for earlier, later in zip(stripped, stripped[1:], strict=False):
        assert later[: len(earlier)] == earlier


async def test_prefix_stable_within_a_single_tool_calling_turn(
    async_client, user_id, async_http_requester, mock_chat_model
):
    """Baseline: within one turn nothing about the session changes, so the prefix is already stable."""
    mock_chat_model.script(
        _tool_call("check_style_guide", {"text": "Colour our organisation."}), "Looks good."
    )

    url = api.chats(user_uuid=user_id)
    await async_http_requester(
        "cache_prefix_single_turn", async_client.post, url, json={"query": "check my style guide compliance"}
    )

    requests = mock_chat_model.requests
    assert len(requests) == 2
    assert not any("tool_choice" in r for r in requests)
    assert requests[0]["tools"] == requests[1]["tools"]
    assert requests[0]["system"] == requests[1]["system"]
    _assert_messages_form_growing_prefixes(requests)


async def test_checkbox_toggle_between_turns_does_not_change_the_tool_list_or_system(
    async_client, user_id, async_http_requester, mock_chat_model
):
    mock_chat_model.script("Hi there.", "Hi again.")

    url = api.chats(user_uuid=user_id)
    chat_response = await async_http_requester(
        "cache_prefix_checkbox_toggle_on",
        async_client.post,
        url,
        json={"query": "hi", "use_gov_uk_search_api": True},
    )
    chat_uuid = chat_response["uuid"]

    add_msg_url = api.get_chat_item(user_uuid=user_id, chat_uuid=chat_uuid)
    await async_http_requester(
        "cache_prefix_checkbox_toggle_off",
        async_client.put,
        add_msg_url,
        json={"query": "hi again", "use_gov_uk_search_api": False},
    )

    requests = mock_chat_model.requests
    assert _tool_names(requests[0]) == _tool_names(requests[1])
    assert requests[0]["system"] == requests[1]["system"]


async def test_document_attached_mid_chat_does_not_change_the_tool_list_or_system(
    async_client, user_id, async_http_requester, mock_chat_model, file_uploader
):
    mock_chat_model.script("Hi there.", "Hi again.")

    url = api.chats(user_uuid=user_id)
    chat_response = await async_http_requester("cache_prefix_no_document", async_client.post, url, json={"query": "hi"})
    chat_uuid = chat_response["uuid"]

    upload_response = await file_uploader("tests/resources/DNA_Topics_UK.docx", "cache_prefix_mid_chat_upload")
    document_uuid = upload_response["document_uuid"]

    add_msg_url = api.get_chat_item(user_uuid=user_id, chat_uuid=chat_uuid)
    await async_http_requester(
        "cache_prefix_document_attached",
        async_client.put,
        add_msg_url,
        json={"query": "hi again", "document_uuids": [document_uuid]},
    )

    requests = mock_chat_model.requests
    assert _tool_names(requests[0]) == _tool_names(requests[1])
    assert requests[0]["system"] == requests[1]["system"]


async def test_date_rollover_between_turns_does_not_change_the_system(
    async_client, user_id, async_http_requester, mock_chat_model
):
    mock_chat_model.script("Hi there.", "Hi again.")

    url = api.chats(user_uuid=user_id)
    with patch("app.chat.prompts.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(2026, 1, 1)
        chat_response = await async_http_requester(
            "cache_prefix_date_day_one", async_client.post, url, json={"query": "hi"}
        )
        chat_uuid = chat_response["uuid"]

        add_msg_url = api.get_chat_item(user_uuid=user_id, chat_uuid=chat_uuid)
        mock_datetime.now.return_value = datetime(2026, 1, 2)
        await async_http_requester(
            "cache_prefix_date_day_two", async_client.put, add_msg_url, json={"query": "hi again"}
        )

    requests = mock_chat_model.requests
    assert requests[0]["system"] == requests[1]["system"]


async def test_iteration_cap_nudges_the_model_instead_of_forcing_tool_choice(
    async_client, user_id, async_http_requester, mock_chat_model
):
    """Cache-stability item 2: a model that complies with the nudge never gets tool_choice."""
    mock_chat_model.script(
        _tool_call("check_style_guide", {"text": "Colour our organisation."}), "Final answer."
    )

    url = api.chats(user_uuid=user_id)
    with patch.object(settings, "agent_loop_max_iterations", 2):
        await async_http_requester(
            "cache_prefix_iteration_cap", async_client.post, url, json={"query": "check my style guide compliance"}
        )

    assert not any("tool_choice" in r for r in mock_chat_model.requests)
