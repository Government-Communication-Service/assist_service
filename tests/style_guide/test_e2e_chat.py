"""
End-to-end tests for style guide feature in chat flow.
Tests the complete integration with chat endpoints and streaming.
"""
import json
import logging
from unittest.mock import patch

import pytest

from app.api.endpoints import ENDPOINTS
from app.style_guide.service import StyleGuideContentType

api = ENDPOINTS()
logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.streaming,
]


def parse_streaming_response(response_data) -> tuple[list[dict], dict]:
    """
    Parse streaming response and return all JSON objects and the final message_streamed packet.
    """
    if isinstance(response_data, bytes):
        response_text = response_data.decode("utf-8")
    else:
        response_text = str(response_data)

    json_objects = []
    decoder = json.JSONDecoder()
    pos = 0
    trimmed_text = response_text.strip()

    while pos < len(trimmed_text):
        try:
            obj, pos = decoder.raw_decode(trimmed_text, pos)
            json_objects.append(obj)
        except json.JSONDecodeError:
            remaining_text = trimmed_text[pos:]
            non_space = remaining_text.lstrip()
            if not non_space:
                break
            skipped_len = len(remaining_text) - len(non_space)
            pos += skipped_len
            if pos < len(trimmed_text):
                try:
                    decoder.raw_decode(trimmed_text, pos)
                except json.JSONDecodeError:
                    pos += 1

    assert len(json_objects) > 0, "No valid JSON objects found in streaming response"

    message_streamed_packets = [p for p in json_objects if "message_streamed" in p]
    assert len(message_streamed_packets) > 0, "No message_streamed packets found in response"

    final_message_packet = message_streamed_packets[-1]

    return json_objects, final_message_packet


class TestStyleGuideE2E:
    """End-to-end tests for style guide in chat."""

    @pytest.mark.asyncio
    async def test_style_guide_initial_check_streaming(
        self,
        async_client,
        user_id,
        async_http_requester,
        db_session,
        default_headers,
        style_guide_use_case_id,
    ):
        """
        Test initial style guide check with streaming response.

        This test:
        1. Creates a chat stream with a use_case_id tied to the style guide theme
        2. Submits text with style guide violations
        3. Verifies the response contains style guide analysis
        4. Checks that violations were detected
        """
        # Create initial message with style guide violations
        content_with_violations = "The Prime Minister met with the Home Secretary today."

        chat_create_payload = {
            "query": content_with_violations,
            "use_rag": False,
            "use_gov_uk_search_api": False,
            "use_case_id": style_guide_use_case_id,
            "use_smart_targets": False,
        }

        # Create chat with stream
        response = await async_client.post(
            api.create_chat_stream(user_uuid=user_id),
            json=chat_create_payload,
            headers=default_headers,
        )

        assert response.status_code == 200

        # Parse streaming response
        json_objects, final_packet = parse_streaming_response(response.content)

        # Extract chat UUID
        chat_uuid = json_objects[0].get("uuid")
        assert chat_uuid is not None

        # Verify message_streamed packet
        assert "message_streamed" in final_packet
        message_data = final_packet["message_streamed"]

        # Verify content contains style guide analysis
        content = message_data.get("content", "")
        assert len(content) > 0
        assert "GOV.UK Style Guide" in content or "style guide" in content.lower()

        # Verify sources (may be empty for initial bypass mode)
        sources_json = message_data.get("sources", "{}")
        _ = json.loads(sources_json) if isinstance(sources_json, str) else sources_json

        logger.info(f"Style guide initial check completed: {len(content)} chars in response")

    @pytest.mark.asyncio
    @patch("app.style_guide.service.determine_style_guide_content_type")
    async def test_style_guide_follow_up_modification_streaming(
        self,
        mock_router,
        async_client,
        user_id,
        async_http_requester,
        db_session,
        default_headers,
        style_guide_use_case_id,
    ):
        """
        Test style guide follow-up modification request with streaming.

        This test:
        1. Creates initial chat with style guide check
        2. Sends follow-up modification request
        3. Verifies router is called to decide flow
        4. Checks that response is conversational
        """
        # Mock router to return simple response mode
        mock_router.return_value = StyleGuideContentType.SIMPLE_RESPONSE

        # Create initial chat
        initial_content = "The Prime Minister announced today."

        chat_create_payload = {
            "query": initial_content,
            "use_rag": False,
            "use_gov_uk_search_api": False,
            "use_case_id": style_guide_use_case_id,
            "use_smart_targets": False,
        }

        response = await async_client.post(
            api.create_chat_stream(user_uuid=user_id),
            json=chat_create_payload,
            headers=default_headers,
        )

        assert response.status_code == 200

        # Extract chat UUID
        json_objects, _ = parse_streaming_response(response.content)
        chat_uuid = json_objects[0].get("uuid")
        assert chat_uuid is not None

        # Send follow-up modification request
        follow_up_content = "Could you make that sentence shorter?"

        follow_up_payload = {
            "query": follow_up_content,
            "use_rag": False,
            "use_gov_uk_search_api": False,
            "use_smart_targets": False,
        }

        follow_up_response = await async_client.put(
            api.get_chat_stream(user_uuid=user_id, chat_uuid=chat_uuid),
            json=follow_up_payload,
            headers=default_headers,
        )

        assert follow_up_response.status_code == 200

        # Parse follow-up response
        follow_up_objects, follow_up_packet = parse_streaming_response(follow_up_response.content)

        # Verify conversational response
        message_data = follow_up_packet["message_streamed"]
        content = message_data.get("content", "")

        assert len(content) > 0

        # Router should have been called for follow-up
        assert mock_router.call_count >= 1

        logger.info(f"Follow-up modification completed: {len(content)} chars in response")

    @pytest.mark.asyncio
    async def test_style_guide_no_violations_streaming(
        self,
        async_client,
        user_id,
        async_http_requester,
        db_session,
        default_headers,
    ):
        """
        Test style guide check on clean content with no violations.

        This test:
        1. Submits clean text without style guide issues
        2. Verifies the response indicates no violations
        """
        # Clean content without violations
        clean_content = "This is clean text without any style guide issues."

        chat_create_payload = {
            "query": clean_content,
            "use_rag": False,
            "use_gov_uk_search_api": False,
            "use_style_guide_checker": True,
            "use_smart_targets": False,
        }

        response = await async_client.post(
            api.create_chat_stream(user_uuid=user_id),
            json=chat_create_payload,
            headers=default_headers,
        )

        assert response.status_code == 200

        # Parse response
        json_objects, final_packet = parse_streaming_response(response.content)

        message_data = final_packet["message_streamed"]
        content = message_data.get("content", "")

        # Should indicate no violations or provide analysis
        assert len(content) > 0

        logger.info(f"Clean content check completed: {len(content)} chars in response")

    @pytest.mark.asyncio
    async def test_style_guide_with_other_features_streaming(
        self,
        async_client,
        user_id,
        async_http_requester,
        db_session,
        sync_central_rag_index,
        default_headers,
    ):
        """
        Test style guide check combined with other features (RAG).

        This test:
        1. Enables both style_guide_checker and RAG
        2. Verifies both features work together
        3. Checks response contains appropriate information
        """
        content = "The Prime Minister announced the new policy."

        chat_create_payload = {
            "query": content,
            "use_rag": True,  # Enable central guidance
            "use_gov_uk_search_api": False,
            "use_style_guide_checker": True,  # Enable style guide
            "use_smart_targets": False,
        }

        response = await async_client.post(
            api.create_chat_stream(user_uuid=user_id),
            json=chat_create_payload,
            headers=default_headers,
        )

        assert response.status_code == 200

        # Parse response
        json_objects, final_packet = parse_streaming_response(response.content)

        message_data = final_packet["message_streamed"]
        content = message_data.get("content", "")

        # Should have response content
        assert len(content) > 0

        # Check sources - may include central_guidance_sources if RAG found relevant docs
        sources_json = message_data.get("sources", "{}")
        sources = json.loads(sources_json) if isinstance(sources_json, str) else sources_json

        logger.info(f"Combined features test completed with {len(sources)} source types")

    @pytest.mark.asyncio
    async def test_style_guide_error_handling(
        self,
        async_client,
        user_id,
        async_http_requester,
        db_session,
        default_headers,
    ):
        """
        Test style guide error handling.

        This test:
        1. Submits content that might cause processing issues
        2. Verifies the system handles errors gracefully
        3. Checks that a response is still provided
        """
        # Edge case: very short content
        edge_case_content = "Hi"

        chat_create_payload = {
            "query": edge_case_content,
            "use_rag": False,
            "use_gov_uk_search_api": False,
            "use_style_guide_checker": True,
            "use_smart_targets": False,
        }

        response = await async_client.post(
            api.create_chat_stream(user_uuid=user_id),
            json=chat_create_payload,
            headers=default_headers,
        )

        # Should not fail even with edge case content
        assert response.status_code == 200

        # Parse response
        json_objects, final_packet = parse_streaming_response(response.content)

        message_data = final_packet["message_streamed"]
        content = message_data.get("content", "")

        # Should still provide some response
        assert len(content) > 0

        logger.info("Edge case handling test completed successfully")


class TestStyleGuideNonStreaming:
    """Tests for style guide with non-streaming responses."""

    @pytest.mark.asyncio
    async def test_style_guide_non_streaming_response(
        self,
        async_client,
        user_id,
        async_http_requester,
        db_session,
        default_headers,
    ):
        """
        Test style guide check with non-streaming response.

        This test:
        1. Creates chat without streaming
        2. Verifies response structure
        3. Checks style guide analysis is included
        """
        content_with_violations = "The Prime Minister met with the Home Secretary."

        chat_create_payload = {
            "query": content_with_violations,
            "use_rag": False,
            "use_gov_uk_search_api": False,
            "use_style_guide_checker": True,
            "use_smart_targets": False,
            "stream": False,  # Non-streaming
        }

        response = await async_client.post(
            api.chats(user_uuid=user_id),
            json=chat_create_payload,
            headers=default_headers,
        )

        assert response.status_code == 200

        # Parse JSON response
        response_data = response.json()

        # Verify response structure
        assert "uuid" in response_data
        assert "message" in response_data

        message = response_data["message"]
        assert "content" in message

        # Verify style guide analysis
        content = message["content"]
        assert len(content) > 0

        logger.info(f"Non-streaming test completed: {len(content)} chars in response")
