from logging import getLogger

from app.audience_segments.clients import AudienceSegmentsClient
from app.audience_segments.schemas import AudienceSegmentSchema
from tests.audience_segments.conftest import AUDIENCE_SEGMENT_1

logger = getLogger(__name__)


class TestAudienceSegmentsClient:
    async def test_get_all_audience_segments(self, mock_get_audience_segments_response):
        audience_segments = await AudienceSegmentsClient.get_all_audience_segments()
        assert audience_segments
        for segment in audience_segments:
            assert isinstance(segment, AudienceSegmentSchema)

    async def test_client(self, mock_get_audience_segment_1_response):
        audience_segment = await AudienceSegmentsClient.get_audience_segment(AUDIENCE_SEGMENT_1["uuid"])
        assert audience_segment
        assert isinstance(audience_segment, AudienceSegmentSchema)
