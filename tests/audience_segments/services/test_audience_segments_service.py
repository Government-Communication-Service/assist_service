from uuid import UUID

from app.audience_segments.schemas import AudienceSegmentSchema
from app.audience_segments.services import AudienceSegmentsService
from tests.audience_segments.conftest import AUDIENCE_SEGMENT_1, AUDIENCE_SEGMENT_2


class TestAudienceSegmentsService:
    async def test_get_all_audience_segments(self, mock_get_audience_segments_response):
        audience_segments = await AudienceSegmentsService.get_all_audience_segments()
        assert audience_segments
        for segment in audience_segments:
            assert isinstance(segment, AudienceSegmentSchema)

    async def test_use_audience_segments(
        self, mock_get_audience_segment_1_response, mock_get_audience_segment_2_response
    ):
        uuids = [
            UUID(AUDIENCE_SEGMENT_1["uuid"]),
            UUID(AUDIENCE_SEGMENT_2["uuid"]),
        ]
        audience_segments = await AudienceSegmentsService.use_audience_segments(uuids)
        assert audience_segments
        assert len(audience_segments) == len(uuids)
        for segment in audience_segments:
            assert isinstance(segment, AudienceSegmentSchema)
