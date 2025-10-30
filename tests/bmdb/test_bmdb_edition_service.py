"""Unit tests for BmdbEditionService.

Tests the get_latest_edition() method with mocked dependencies.
"""

from datetime import date
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.bmdb.exceptions import GetBenchmarkDatabaseEditionError
from app.bmdb.schemas import LatestEdition
from app.bmdb.services.bmdb_edition_service import BmdbEditionService

pytestmark = [
    pytest.mark.unit,
]


@pytest.fixture
def sample_latest_edition():
    """Fixture providing a sample LatestEdition object for testing."""
    return LatestEdition(
        version_number=1,
        date_received=date(2025, 1, 15),
        latest_campaign_end_date=date(2024, 12, 31),
        earliest_campaign_end_date=date(2024, 1, 1),
        n_campaigns=100,
        min_media_spend=5000.0,
        max_media_spend=1000000.0,
    )


@pytest.mark.asyncio
@patch("app.bmdb.services.bmdb_edition_service.BmdbClient.get_latest_edition")
async def test_get_latest_edition_success(mock_get_latest_edition, sample_latest_edition):
    """Test successful retrieval of latest edition information.

    Verifies that:
    - The service method calls BmdbClient.get_latest_edition()
    - The returned LatestEdition object has the correct data
    - All fields are correctly passed through
    """
    # Arrange: Mock the BmdbClient to return our sample data
    mock_get_latest_edition.return_value = sample_latest_edition

    # Act: Call the service method
    result = await BmdbEditionService.get_latest_edition()

    # Assert: Verify the result matches expected data
    assert result == sample_latest_edition
    assert result.version_number == 1
    assert result.date_received == date(2025, 1, 15)
    assert result.latest_campaign_end_date == date(2024, 12, 31)
    assert result.earliest_campaign_end_date == date(2024, 1, 1)
    assert result.n_campaigns == 100
    assert result.min_media_spend == 5000.0
    assert result.max_media_spend == 1000000.0

    # Verify the client method was called once
    mock_get_latest_edition.assert_called_once()


@pytest.mark.asyncio
@patch("app.bmdb.services.bmdb_edition_service.BmdbClient.get_latest_edition")
async def test_get_latest_edition_raises_error(mock_get_latest_edition):
    """Test that exceptions from BmdbClient are wrapped in GetBenchmarkDatabaseEditionError.

    Verifies that:
    - When BmdbClient raises an exception, the service wraps it
    - The wrapped exception is of the correct type
    - The error message is descriptive
    - The original exception is chained (accessible via __cause__)
    """
    # Arrange: Mock the BmdbClient to raise an HTTP error
    original_error = httpx.HTTPStatusError("500 Server Error", request=AsyncMock(), response=AsyncMock())
    mock_get_latest_edition.side_effect = original_error

    # Act & Assert: Verify the exception is wrapped correctly
    with pytest.raises(GetBenchmarkDatabaseEditionError) as exc_info:
        await BmdbEditionService.get_latest_edition()

    # Verify error message
    assert str(exc_info.value) == ("Failed to get information about the latest edition of the Benchmark Database")

    # Verify the original exception is chained
    assert exc_info.value.__cause__ == original_error

    # Verify the client method was called
    mock_get_latest_edition.assert_called_once()
