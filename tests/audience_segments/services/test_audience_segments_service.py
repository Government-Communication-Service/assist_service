from app.audience_segments.schemas import AudienceSegmentSchema
from app.audience_segments.services import AudienceSegmentsService


class TestAudienceSegmentsService:
    async def test_get_all_audience_segments(self, mock_get_audience_segments_response):
        audience_segments = await AudienceSegmentsService.get_all_audience_segments()
        assert audience_segments
        for segment in audience_segments:
            assert isinstance(segment, AudienceSegmentSchema)
