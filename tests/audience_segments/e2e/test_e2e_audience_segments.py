import json
from logging import getLogger

from app.api.endpoints import ENDPOINTS
from tests.audience_segments.conftest import AUDIENCE_SEGMENT_1

logger = getLogger(__name__)


class TestE2EAudienceSegments:
    async def test_chat_with_audience_segments(
        self,
        user_id,
        async_client,
        async_http_requester,
        mock_get_audience_segment_1_response,
        mock_llm_response,
    ):
        """
        Checks that a message that references audience segments works.
        """
        api = ENDPOINTS()
        url = api.chats(user_uuid=user_id)
        response = await async_http_requester(
            "chat_endpoint",
            async_client.post,
            url,
            json={
                "query": "Write 3 words about each audience segment.",
                "audience_segment_uuids": [AUDIENCE_SEGMENT_1["uuid"]],
            },
        )
        logger.info(f"{response=}")
        sources = json.loads(response["message"]["sources"])
        assert sources, "No 'sources' field was returned by the LLM when using the audience segments feature"
        audience_segments_sources = sources["audience_segments_sources"]
        assert audience_segments_sources, (
            "No 'audience_segments_sources' field was returned by the LLM when using the audience segments feature"
        )
        assert len(audience_segments_sources) == 1, (
            f"Expected 1 entry in 'audience_segments_sources', got {len(audience_segments_sources)}"
        )
        assert AUDIENCE_SEGMENT_1["uuid"] in audience_segments_sources[0]["url"], (
            f"Expected audience_segment_uuid {AUDIENCE_SEGMENT_1['uuid']} to be in url, "
            f"got {audience_segments_sources[0]['url']}"
        )
