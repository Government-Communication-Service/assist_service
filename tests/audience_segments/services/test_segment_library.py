"""Tests for the cached segment library, name lookup, and the get_audience_segments executor."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.audience_segments.schemas import AudienceSegmentSchema
from app.audience_segments.services import AudienceSegmentsService
from app.audience_segments.tools import GET_AUDIENCE_SEGMENTS
from app.chat.tools import TOOL_REGISTRY
from tests.audience_segments.conftest import AUDIENCE_SEGMENT_1, AUDIENCE_SEGMENT_2

SEGMENTS = [AudienceSegmentSchema(**AUDIENCE_SEGMENT_2), AudienceSegmentSchema(**AUDIENCE_SEGMENT_1)]


@pytest.fixture(autouse=True)
def library():
    AudienceSegmentsService._library = None
    AudienceSegmentsService._library_retry_after = 0.0
    with patch(
        "app.audience_segments.services.audience_segments.AudienceSegmentsClient.get_all_audience_segments",
        AsyncMock(return_value=SEGMENTS),
    ) as fetch:
        yield fetch
    AudienceSegmentsService._library = None
    AudienceSegmentsService._library_retry_after = 0.0


@pytest.fixture
def clock():
    """A controllable time.monotonic for the service module."""
    now = SimpleNamespace(value=1000.0)
    with patch("app.audience_segments.services.audience_segments.time.monotonic", lambda: now.value):
        yield now


async def test_library_sorted_and_cached(library):
    first = await AudienceSegmentsService.get_segment_library()
    second = await AudienceSegmentsService.get_segment_library()

    assert [s.pretty_name for s in first] == ["Test Segment 1", "Test Segment 2"]
    assert second is first
    library.assert_awaited_once()


async def test_failed_refresh_serves_stale_library_and_backs_off(library, clock, caplog):
    first = await AudienceSegmentsService.get_segment_library()
    library.side_effect = httpx.ConnectTimeout("down")
    clock.value += 601  # past the 600 s cache

    with caplog.at_level("WARNING"):
        stale = await AudienceSegmentsService.get_segment_library()

    assert stale is first
    assert "serving the stale copy" in caplog.text
    assert library.await_count == 2

    # Within the backoff, the API isn't called again.
    clock.value += 59
    assert await AudienceSegmentsService.get_segment_library() is first
    assert library.await_count == 2

    # After the backoff, a successful retry refreshes the library.
    library.side_effect = None
    clock.value += 2
    await AudienceSegmentsService.get_segment_library()
    assert library.await_count == 3
    assert AudienceSegmentsService._library_retry_after == 0.0


async def test_failed_fetch_with_nothing_cached_raises_and_backs_off(library, clock):
    library.side_effect = httpx.ConnectTimeout("down")

    with pytest.raises(httpx.ConnectTimeout):
        await AudienceSegmentsService.get_segment_library()
    with pytest.raises(httpx.HTTPError):
        await AudienceSegmentsService.get_segment_library()
    assert library.await_count == 1

    library.side_effect = None
    clock.value += 61
    assert len(await AudienceSegmentsService.get_segment_library()) == 2
    assert library.await_count == 2


async def test_find_segments_by_pretty_name_or_name_ignoring_case():
    found, missing = await AudienceSegmentsService.find_segments(
        ["test segment 2", "TEST-SEGMENT-1", "Nope", "Test Segment 2"]
    )

    assert [s.name for s in found] == ["test-segment-2", "test-segment-1"]
    assert missing == ["Nope"]


async def test_executor_returns_insights_and_sources():
    tool = TOOL_REGISTRY["get_audience_segments"]
    args = GET_AUDIENCE_SEGMENTS.parse({"segment_names": ["Test Segment 1", "Unknown"]})

    result = await tool.executor(SimpleNamespace(), args, SimpleNamespace(), None)

    assert "<segment-insights segment-name:Test Segment 1>" in result.content
    assert "Unknown segment(s): Unknown" in result.content
    assert [s.pretty_name for s in result.sources.audience_segments_sources] == ["Test Segment 1"]
