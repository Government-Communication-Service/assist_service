# ruff: noqa: E501

import json
import logging
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.future import select

from app.api.endpoints import ENDPOINTS
from app.chat.schemas import CentralGuidanceSource, SmartTargetsSource, Sources
from app.database.models import Chat, Message

api = ENDPOINTS()
logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.streaming,
]


def extract_chat_uuid_from_stream(json_objects: list) -> str:
    """Extract chat UUID from streaming response packets."""
    for obj in json_objects:
        if "uuid" in obj:
            return obj["uuid"]
    raise ValueError("No chat UUID found in streaming response")


def parse_streaming_response(response_data) -> tuple[list[dict], dict]:
    """
    Parse streaming response and return all JSON objects and the final message_streamed packet.

    Uses JSONDecoder.raw_decode to properly handle nested JSON structures.

    Returns:
        tuple: (all_json_objects, final_message_streamed_packet)
    """
    # Convert bytes to string if needed
    if isinstance(response_data, bytes):
        response_text = response_data.decode("utf-8")
    else:
        response_text = str(response_data)

    # Parse concatenated JSON objects using JSONDecoder
    json_objects = []
    decoder = json.JSONDecoder()
    pos = 0
    trimmed_text = response_text.strip()

    while pos < len(trimmed_text):
        try:
            obj, pos = decoder.raw_decode(trimmed_text, pos)
            json_objects.append(obj)
        except json.JSONDecodeError:
            # Skip whitespace or invalid chars between objects
            remaining_text = trimmed_text[pos:]
            non_space = remaining_text.lstrip()
            if not non_space:
                break  # End of string
            skipped_len = len(remaining_text) - len(non_space)
            pos += skipped_len
            # If after skipping whitespace we still can't decode, skip one char
            if pos < len(trimmed_text):
                try:
                    decoder.raw_decode(trimmed_text, pos)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping unexpected character '{trimmed_text[pos]}' at index {pos}")
                    pos += 1

    # Ensure we got at least one valid JSON object
    assert len(json_objects) > 0, "No valid JSON objects found in streaming response"

    logger.info(f"Parsed {len(json_objects)} JSON objects from stream")

    # Find the final message_streamed packet
    message_streamed_packets = [p for p in json_objects if "message_streamed" in p]
    assert len(message_streamed_packets) > 0, "No message_streamed packets found in response"

    final_message_packet = message_streamed_packets[-1]

    return json_objects, final_message_packet


@pytest.mark.asyncio
@patch("app.chat.service.SmartTargetsService")
async def test_smart_targets_with_central_guidance_streaming(
    mock_smart_targets_service,
    async_client,
    user_id,
    async_http_requester,
    db_session,
    sync_central_rag_index,
):
    """
    End-to-end test for Smart Targets with Central Guidance using streaming response.

    This test:
    1. Creates a chat stream with both use_rag=True and use_smart_targets=True
    2. Verifies the streaming response completes successfully
    3. Validates the 'sources' field in the streamed response has expected structure
    4. Checks the database to ensure the 'sources' column is populated correctly

    Note: SmartTargetsService is mocked because the Smart Targets API is not running in tests.
    """
    # Setup mock Smart Targets service response
    mock_instance = mock_smart_targets_service.return_value
    mock_instance.use_smart_targets_tool = AsyncMock(
        return_value=(
            # Context string (what the LLM sees)
            "Campaign Performance Statistics (campaign-recognition, unit=percentage), filters_applied={region: [London]; channel: [TV]}:\n"
            "- Number of campaigns: 45\n"
            "- Number of data points: 180\n"
            "- Lower quartile: 35.2 percentage\n"
            "- Median: 48.5\n"
            "- Upper quartile: 62.8 percentage\n"
            "- 95% confidence interval: 42.3 to 54.7 percentage",
            # Citations list
            [
                {
                    "docname": "Smart Targets dashboard: metric=campaign-recognition; filters=region: [London]; channel: [TV]",
                    "docurl": "https://smart-targets.example.com/dashboard/123",
                }
            ],
        )
    )

    logger.info(f"Creating Smart Targets + Central Guidance chat stream for user ID: {user_id}")

    # Create streaming endpoint
    endpoint = api.create_chat_stream(user_uuid=user_id)

    # Query designed to trigger both Smart Targets (campaign metrics) and Central Guidance (OASIS)
    query = (
        "What are typical campaign recognition scores in recent campaigns? "
        "Also, what is the GCS guidance on OASIS planning?"
    )

    # Make streaming request with both RAG and Smart Targets enabled
    response_data = await async_http_requester(
        "test_smart_targets_with_central_guidance_streaming",
        async_client.post,
        endpoint,
        response_type="text",
        response_content_type="text/event-stream; charset=utf-8",
        json={"query": query, "use_rag": True, "use_smart_targets": True},
    )

    # Verify response is not None
    assert response_data is not None, "Streaming response should not be None"

    # Parse the streaming response
    all_json_objects, final_message_packet = parse_streaming_response(response_data)

    logger.info(f"Parsed {len(all_json_objects)} JSON packets from stream")

    # Extract chat UUID for database lookup
    chat_uuid = extract_chat_uuid_from_stream(all_json_objects)
    logger.info(f"Extracted chat UUID: {chat_uuid}")

    # Verify message_streamed structure
    assert "message_streamed" in final_message_packet, "Final packet should contain message_streamed"
    message_streamed = final_message_packet["message_streamed"]

    # Verify basic message fields
    assert "uuid" in message_streamed, "message_streamed should have uuid"
    assert "role" in message_streamed, "message_streamed should have role"
    assert message_streamed["role"] == "assistant", "message role should be assistant"
    assert "content" in message_streamed, "message_streamed should have content"
    assert message_streamed["content"] != "", "message content should not be empty"

    # Verify sources field exists in streaming response
    assert "sources" in message_streamed, "message_streamed should have sources field"
    assert message_streamed["sources"] != "", "sources field should not be empty"

    logger.info(f"Sources field in streaming response: {message_streamed['sources']}")

    # Parse and validate sources structure in streaming response
    stream_sources = Sources.model_validate_json(message_streamed["sources"])

    # Verify Smart Targets sources are present
    assert stream_sources.smart_targets_sources is not None, "Smart Targets sources should be present"
    assert len(stream_sources.smart_targets_sources) >= 1, "Should have at least 1 Smart Targets source"

    # Verify Central Guidance sources are present
    assert stream_sources.central_guidance_sources is not None, "Central Guidance sources should be present"
    assert len(stream_sources.central_guidance_sources) >= 1, "Should have at least 1 Central Guidance source"

    # Validate Smart Targets source structure
    for source in stream_sources.smart_targets_sources:
        assert isinstance(source, SmartTargetsSource), "Should be SmartTargetsSource type"
        assert source.pretty_name != "", "Smart Targets source should have pretty_name"
        assert source.url != "", "Smart Targets source should have url"
        logger.info(f"Smart Targets source: {source.pretty_name} - {source.url}")

    # Validate Central Guidance source structure
    for source in stream_sources.central_guidance_sources:
        assert isinstance(source, CentralGuidanceSource), "Should be CentralGuidanceSource type"
        assert source.pretty_name != "", "Central Guidance source should have pretty_name"
        assert source.url != "", "Central Guidance source should have url"
        logger.info(f"Central Guidance source: {source.pretty_name} - {source.url}")

    # --- Database Verification ---
    logger.info("Verifying database persistence of sources...")

    # Get chat from database
    execute = await db_session.execute(select(Chat).filter(Chat.uuid == chat_uuid))
    chat_model = execute.scalar_one()
    logger.info(f"Found chat in database with ID: {chat_model.id}")

    # Note: use_smart_targets flag on Chat model may not be set during streaming endpoint
    # The important part is that the sources field is correctly populated
    assert chat_model.use_smart_targets, f"Chat.use_smart_targets flag should be True: {chat_model.use_smart_targets=}"
    logger.info(f"{chat_model.use_smart_targets=}")

    # Get assistant message from database
    execute = await db_session.execute(
        select(Message)
        .filter(Message.chat_id == chat_model.id, Message.role == "assistant")
        .order_by(Message.created_at.desc())
    )
    assistant_message = execute.scalar_one()
    logger.info(f"Found assistant message in database with UUID: {assistant_message.uuid}")

    # Verify sources column is populated
    assert assistant_message.sources is not None, "Assistant message sources column should not be None"
    assert assistant_message.sources != "", "Assistant message sources column should not be empty"

    logger.info(f"Sources from database: {assistant_message.sources}")

    # Parse and validate database sources
    db_sources = Sources.model_validate_json(assistant_message.sources)

    # Verify Smart Targets sources in database
    assert db_sources.smart_targets_sources is not None, "DB should have Smart Targets sources"
    assert len(db_sources.smart_targets_sources) >= 1, "DB should have at least 1 Smart Targets source"

    # Verify Central Guidance sources in database
    assert db_sources.central_guidance_sources is not None, "DB should have Central Guidance sources"
    assert len(db_sources.central_guidance_sources) >= 1, "DB should have at least 1 Central Guidance source"

    # Verify database sources match streaming response sources
    assert len(db_sources.smart_targets_sources) == len(stream_sources.smart_targets_sources), (
        "DB and stream should have same number of Smart Targets sources"
    )
    assert len(db_sources.central_guidance_sources) == len(stream_sources.central_guidance_sources), (
        "DB and stream should have same number of Central Guidance sources"
    )

    # Verify actual content matches
    for i, db_source in enumerate(db_sources.smart_targets_sources):
        stream_source = stream_sources.smart_targets_sources[i]
        assert db_source.pretty_name == stream_source.pretty_name, "Smart Targets source names should match"
        assert db_source.url == stream_source.url, "Smart Targets source URLs should match"

    for i, db_source in enumerate(db_sources.central_guidance_sources):
        stream_source = stream_sources.central_guidance_sources[i]
        assert db_source.pretty_name == stream_source.pretty_name, "Central Guidance source names should match"
        assert db_source.url == stream_source.url, "Central Guidance source URLs should match"

    # --- GET Endpoint Verification ---
    logger.info("Verifying sources returned by GET endpoints...")

    # Test 1: Get chat using get_chat_item endpoint
    logger.info("Testing GET /v1/chats/users/{user_id}/chats/{chat_uuid} endpoint")
    get_chat_url = api.get_chat_item(user_id, chat_uuid)
    get_chat_response = await async_http_requester(
        "test_get_chat_item_sources",
        async_client.get,
        get_chat_url,
    )

    # Verify response structure
    assert "messages" in get_chat_response, "GET chat response should contain messages"
    assert len(get_chat_response["messages"]) >= 2, "Should have at least user and assistant messages"

    # Find the assistant message in the response
    get_chat_assistant_msg = None
    for msg in get_chat_response["messages"]:
        if msg["role"] == "assistant":
            get_chat_assistant_msg = msg
            break

    assert get_chat_assistant_msg is not None, "Should find assistant message in GET chat response"
    logger.info(f"Found assistant message in GET chat response with UUID: {get_chat_assistant_msg['uuid']}")

    # Verify sources field exists in GET response
    assert "sources" in get_chat_assistant_msg, "Assistant message should have sources field in GET response"
    assert get_chat_assistant_msg["sources"] != "", "Sources should not be empty in GET response"

    logger.info(f"Sources from get_chat_item endpoint: {get_chat_assistant_msg['sources']}")

    # Parse and validate sources from GET endpoint
    get_chat_sources = Sources.model_validate_json(get_chat_assistant_msg["sources"])

    # Verify Smart Targets sources in GET response
    assert get_chat_sources.smart_targets_sources is not None, "GET response should have Smart Targets sources"
    assert len(get_chat_sources.smart_targets_sources) >= 1, "GET response should have at least 1 Smart Targets source"

    # Verify Central Guidance sources in GET response
    assert get_chat_sources.central_guidance_sources is not None, "GET response should have Central Guidance sources"
    assert len(get_chat_sources.central_guidance_sources) >= 1, (
        "GET response should have at least 1 Central Guidance source"
    )

    # Verify GET response sources match streaming response sources
    assert len(get_chat_sources.smart_targets_sources) == len(stream_sources.smart_targets_sources), (
        "GET and stream should have same number of Smart Targets sources"
    )
    assert len(get_chat_sources.central_guidance_sources) == len(stream_sources.central_guidance_sources), (
        "GET and stream should have same number of Central Guidance sources"
    )

    # Verify actual content matches streaming response
    for i, get_source in enumerate(get_chat_sources.smart_targets_sources):
        stream_source = stream_sources.smart_targets_sources[i]
        assert get_source.pretty_name == stream_source.pretty_name, "GET Smart Targets source names should match stream"
        assert get_source.url == stream_source.url, "GET Smart Targets source URLs should match stream"

    for i, get_source in enumerate(get_chat_sources.central_guidance_sources):
        stream_source = stream_sources.central_guidance_sources[i]
        assert get_source.pretty_name == stream_source.pretty_name, (
            "GET Central Guidance source names should match stream"
        )
        assert get_source.url == stream_source.url, "GET Central Guidance source URLs should match stream"

    logger.info("✓ get_chat_item endpoint sources verified successfully")

    # Test 2: Get chat using get_chat_messages endpoint
    logger.info("Testing GET /v1/chats/users/{user_id}/chats/{chat_uuid}/messages endpoint")
    get_messages_url = api.get_chat_messages(user_id, chat_uuid)
    get_messages_response = await async_http_requester(
        "test_get_chat_messages_sources",
        async_client.get,
        get_messages_url,
    )

    # Verify response structure
    assert "messages" in get_messages_response, "GET messages response should contain messages"
    assert len(get_messages_response["messages"]) >= 2, "Should have at least user and assistant messages"

    # Find the assistant message in the response
    get_messages_assistant_msg = None
    for msg in get_messages_response["messages"]:
        if msg["role"] == "assistant":
            get_messages_assistant_msg = msg
            break

    assert get_messages_assistant_msg is not None, "Should find assistant message in GET messages response"
    logger.info(f"Found assistant message in GET messages response with UUID: {get_messages_assistant_msg['uuid']}")

    # Verify sources field exists
    assert "sources" in get_messages_assistant_msg, "Assistant message should have sources field in messages response"
    assert get_messages_assistant_msg["sources"] != "", "Sources should not be empty in messages response"

    logger.info(f"Sources from get_chat_messages endpoint: {get_messages_assistant_msg['sources']}")

    # Parse and validate sources from GET messages endpoint
    get_messages_sources = Sources.model_validate_json(get_messages_assistant_msg["sources"])

    # Verify Smart Targets sources
    assert get_messages_sources.smart_targets_sources is not None, "GET messages should have Smart Targets sources"
    assert len(get_messages_sources.smart_targets_sources) >= 1, (
        "GET messages should have at least 1 Smart Targets source"
    )

    # Verify Central Guidance sources
    assert get_messages_sources.central_guidance_sources is not None, (
        "GET messages should have Central Guidance sources"
    )
    assert len(get_messages_sources.central_guidance_sources) >= 1, (
        "GET messages should have at least 1 Central Guidance source"
    )

    # Verify both GET endpoints return identical sources
    assert len(get_messages_sources.smart_targets_sources) == len(get_chat_sources.smart_targets_sources), (
        "Both GET endpoints should have same number of Smart Targets sources"
    )
    assert len(get_messages_sources.central_guidance_sources) == len(get_chat_sources.central_guidance_sources), (
        "Both GET endpoints should have same number of Central Guidance sources"
    )

    # Verify content matches between both GET endpoints
    for i, msg_source in enumerate(get_messages_sources.smart_targets_sources):
        chat_source = get_chat_sources.smart_targets_sources[i]
        assert msg_source.pretty_name == chat_source.pretty_name, (
            "Smart Targets source names should match between GET endpoints"
        )
        assert msg_source.url == chat_source.url, "Smart Targets source URLs should match between GET endpoints"

    for i, msg_source in enumerate(get_messages_sources.central_guidance_sources):
        chat_source = get_chat_sources.central_guidance_sources[i]
        assert msg_source.pretty_name == chat_source.pretty_name, (
            "Central Guidance source names should match between GET endpoints"
        )
        assert msg_source.url == chat_source.url, "Central Guidance source URLs should match between GET endpoints"
