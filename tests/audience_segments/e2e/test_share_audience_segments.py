import json
from logging import getLogger

import pytest

from app.api.endpoints import ENDPOINTS
from tests.audience_segments.conftest import AUDIENCE_SEGMENT_1

logger = getLogger(__name__)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.audience_segments,
]


class TestShareAudienceSegments:
    async def test_shared_chat_includes_audience_segment_references(
        self,
        user_id,
        async_client,
        async_http_requester,
        mock_get_audience_segment_1_response,
        mock_llm_response,
    ):
        """
        Test that audience segment references are preserved when a chat is shared.
        This is a regression test for the bug where audience segment sources
        disappear when viewing a shared chat link.
        """
        api = ENDPOINTS()

        # Step 1: Create a chat with audience segments
        chat_url = api.chats(user_uuid=user_id)
        chat_response = await async_http_requester(
            "create chat with audience segment",
            async_client.post,
            chat_url,
            json={
                "query": "Write 3 words about this audience segment.",
                "audience_segment_uuids": [AUDIENCE_SEGMENT_1["uuid"]],
            },
        )

        chat_uuid = chat_response["uuid"]
        logger.info(f"Created chat {chat_uuid}")

        # Verify the response has audience_segments_sources
        sources = json.loads(chat_response["message"]["sources"])
        assert "audience_segments_sources" in sources, "No audience_segments_sources in original chat response"
        assert len(sources["audience_segments_sources"]) == 1, "Expected 1 audience segment source"

        # Step 2: Enable sharing for this chat
        share_url = f"/v1/chats/users/{user_id}/chats/{chat_uuid}/share"
        share_response = await async_http_requester(
            "enable chat sharing",
            async_client.patch,
            share_url,
            json={"share": True},
        )

        share_code = share_response["share_code"]
        logger.info(f"Created share code: {share_code}")
        assert share_code is not None, "Share code was not generated"

        # Step 3: Retrieve the shared chat using share_code
        shared_chat_url = f"/v1/chats/shared/{share_code}"
        shared_chat_response = await async_http_requester(
            "get shared chat",
            async_client.get,
            shared_chat_url,
        )

        logger.info(f"Shared chat response: {json.dumps(shared_chat_response, indent=2)}")

        # Step 4: Verify audience segment sources are present in shared chat
        assert "messages" in shared_chat_response, "No messages in shared chat response"
        assert len(shared_chat_response["messages"]) > 0, "Expected at least one message in shared chat"

        # Find the assistant message (the one with sources)
        assistant_message = None
        for msg in shared_chat_response["messages"]:
            if msg["role"] == "assistant":
                assistant_message = msg
                break

        assert assistant_message is not None, "No assistant message found in shared chat"
        assert "sources" in assistant_message, "No sources field in assistant message"

        # Parse and verify the sources
        shared_sources = json.loads(assistant_message["sources"])
        assert "audience_segments_sources" in shared_sources, (
            "BUG CONFIRMED: audience_segments_sources missing from shared chat! "
            f"Sources: {shared_sources}"
        )

        # Verify the content matches
        assert len(shared_sources["audience_segments_sources"]) == 1, (
            f"Expected 1 audience segment source in shared chat, "
            f"got {len(shared_sources['audience_segments_sources'])}"
        )

        assert AUDIENCE_SEGMENT_1["uuid"] in shared_sources["audience_segments_sources"][0]["url"], (
            f"Expected audience segment UUID {AUDIENCE_SEGMENT_1['uuid']} in shared chat URL, "
            f"got {shared_sources['audience_segments_sources'][0]['url']}"
        )

        logger.info("✓ Audience segment references are correctly preserved in shared chat")
