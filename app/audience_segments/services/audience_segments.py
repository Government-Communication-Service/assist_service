import asyncio
from uuid import UUID

from app.audience_segments.clients import AudienceSegmentsClient
from app.audience_segments.schemas import AudienceSegmentSchema


class AudienceSegmentsService:
    @staticmethod
    async def get_all_audience_segments() -> list[AudienceSegmentSchema]:
        return await AudienceSegmentsClient.get_all_audience_segments()

    @staticmethod
    async def use_audience_segments(audience_segment_uuids: list[UUID]) -> list[AudienceSegmentSchema]:
        tasks = [
            AudienceSegmentsClient.get_audience_segment(audience_segment_uuid)
            for audience_segment_uuid in audience_segment_uuids
        ]
        return await asyncio.gather(*tasks)
