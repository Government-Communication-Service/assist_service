from app.audience_segments.schemas import AudienceSegmentSchema
from tests.audience_segments.conftest import AUDIENCE_SEGMENT_1


def test_segment_without_connect_url_is_valid():
    segment = AudienceSegmentSchema(**{**AUDIENCE_SEGMENT_1, "connect_url": None})

    assert segment.connect_url is None
