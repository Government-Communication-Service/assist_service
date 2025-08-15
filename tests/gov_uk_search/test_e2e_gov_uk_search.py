import logging

import pytest

# from pydantic import ValidationError
from app.api.endpoints import ENDPOINTS

api = ENDPOINTS()


logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_stream_with_gov_uk_search_api_response(default_headers, user_id, async_client, async_http_requester):
    """
    This tests uses Search Gov UK
    Test the chat stream response returns an event stream response when the user creates a chat stream.
    Since FastAPI test client does not support true streaming / http response chunking,
    here only content type is checked.
    """
    from app.api.endpoints import ENDPOINTS

    logger.debug(f"Creating chat stream for user ID: {user_id}")

    api_class = ENDPOINTS()
    endpoint = api_class.create_chat_stream(user_uuid=user_id)
    non_rag_url = "https://www.gov.uk"
    response_data = await async_http_requester(
        "test_stream_response",
        async_client.post,
        endpoint,
        response_type="text",
        response_content_type="text/event-stream; charset=utf-8",
        json={
            "query": "What is the current guidance on contempt of court?",
            "use_gov_uk_search_api": True,
            "enable_web_browsing": False,
        },
    )

    print(f"--stream response_data: {response_data}")
    assert response_data is not None
    assert non_rag_url in str(response_data), (
        "Expected some documents to be returned from GOV.UK Search when asking about contempt of court, received none"
    )
