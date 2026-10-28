import time
from logging import getLogger

import httpx

from app.audience_segments.clients import AudienceSegmentsClient
from app.audience_segments.schemas import AudienceSegmentSchema
from app.config import AUDIENCE_SEGMENTS_LIBRARY_CACHE_SECONDS, AUDIENCE_SEGMENTS_LIBRARY_RETRY_BACKOFF_SECONDS

logger = getLogger(__name__)


class AudienceSegmentsService:
    _library: list[AudienceSegmentSchema] | None = None
    _library_fetched_at: float = 0.0
    # monotonic time before which a failed refresh isn't retried
    _library_retry_after: float = 0.0

    @staticmethod
    async def get_all_audience_segments() -> list[AudienceSegmentSchema]:
        return await AudienceSegmentsClient.get_all_audience_segments()

    @classmethod
    async def get_segment_library(cls) -> list[AudienceSegmentSchema]:
        """All segments sorted by pretty name, cached in-process for AUDIENCE_SEGMENTS_LIBRARY_CACHE_SECONDS.

        If a refresh fails, the stale library is served and the fetch isn't retried for
        AUDIENCE_SEGMENTS_LIBRARY_RETRY_BACKOFF_SECONDS.

        Raises:
            Exception: The library couldn't be fetched and nothing is cached. During the backoff
                this is an httpx.HTTPError raised without calling the API.
        """
        now = time.monotonic()
        if cls._library is not None and now - cls._library_fetched_at <= AUDIENCE_SEGMENTS_LIBRARY_CACHE_SECONDS:
            return cls._library
        if now < cls._library_retry_after:
            if cls._library is not None:
                return cls._library
            raise httpx.HTTPError("Audience segment library unavailable; backing off after a failed fetch")
        try:
            segments = await AudienceSegmentsClient.get_all_audience_segments()
        except Exception as e:
            cls._library_retry_after = time.monotonic() + AUDIENCE_SEGMENTS_LIBRARY_RETRY_BACKOFF_SECONDS
            if cls._library is None:
                raise
            logger.warning(
                f"Failed to refresh the audience segment library; serving the stale copy and retrying in "
                f"{AUDIENCE_SEGMENTS_LIBRARY_RETRY_BACKOFF_SECONDS}s. Error: {e!r}"
            )
            return cls._library
        cls._library = sorted(segments, key=lambda segment: segment.pretty_name)
        cls._library_fetched_at = time.monotonic()
        cls._library_retry_after = 0.0
        return cls._library

    @classmethod
    async def find_segments(cls, names: list[str]) -> tuple[list[AudienceSegmentSchema], list[str]]:
        """Look up segments by pretty name or name, ignoring case.

        Returns:
            The segments found, in request order without duplicates, and the names not found.
        """
        by_name = {}
        for segment in await cls.get_segment_library():
            by_name[segment.pretty_name.casefold()] = segment
            by_name[segment.name.casefold()] = segment
        found = []
        missing = []
        for name in names:
            segment = by_name.get(name.strip().casefold())
            if segment is None:
                missing.append(name)
            elif segment not in found:
                found.append(segment)
        return found, missing
