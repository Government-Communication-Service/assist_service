from httpx import AsyncClient

from app.audience_segments.constants import AUDIENCE_SEGMENT_ENDPOINT
from app.audience_segments.schemas import AudienceSegmentSchema
from app.config import GCS_DATA_API_URL


class AudienceSegmentsClient:
    URL = f"{GCS_DATA_API_URL}{AUDIENCE_SEGMENT_ENDPOINT}"

    @classmethod
    async def get_all_audience_segments(cls) -> list[AudienceSegmentSchema]:
        async with AsyncClient(timeout=10.0) as client:
            response = await client.get(cls.URL)
            response.raise_for_status()
            data = response.json()
            segments = data["audience_segments"]
            return [AudienceSegmentSchema(**segment) for segment in segments]

    @classmethod
    async def get_audience_segment(cls, audience_segment_uuid) -> AudienceSegmentSchema:
        async with AsyncClient(timeout=10.0) as client:
            url = f"{cls.URL}/{audience_segment_uuid}"
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            segment = data["audience_segment"]
            return AudienceSegmentSchema(**segment)
