from unittest.mock import AsyncMock, Mock

import pytest

AUDIENCE_SEGMENT_1 = {
    "uuid": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "name": "test-segment-1",
    "pretty_name": "Test Segment 1",
    "insights_markdown": "- This is a test segment\n- There is a lot of information",
    "connect_url": "test.url.com/audience-insights?audience_segment_uuid=3fa85f64-5717-4562-b3fc-2c963f66afa6",
}

AUDIENCE_SEGMENT_2 = {
    "uuid": "8908ac76-e093-4d27-9629-7aa9607c0ddc",
    "name": "test-segment-2",
    "pretty_name": "Test Segment 2",
    "insights_markdown": "- This is another test segment\n- There is also a lot of information",
    "connect_url": "test.url.com/audience-insights?audience_segment_uuid=8908ac76-e093-4d27-9629-7aa9607c0ddc",
}


SUCCESS_DATA_GET_AUDIENCE_SEGMENTS = {
    "message": "success",
    "audience_segments": [AUDIENCE_SEGMENT_1, AUDIENCE_SEGMENT_2],
}

SUCCESS_DATA_GET_AUDIENCE_SEGMENT_1 = {
    "message": "success",
    "audience_segment": AUDIENCE_SEGMENT_1,
}

SUCCESS_DATA_GET_AUDIENCE_SEGMENT_2 = {
    "message": "success",
    "audience_segment": AUDIENCE_SEGMENT_2,
}


def _mock_client(mocker, return_value):
    mock_client = Mock()
    mock_response = Mock()
    mock_response.json = Mock(return_value=return_value)
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_async_client = AsyncMock()
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=None)
    return mocker.patch("app.audience_segments.clients.audience_segments.AsyncClient", return_value=mock_async_client)


@pytest.fixture
def mock_get_audience_segments_response(mocker):
    return _mock_client(mocker, return_value=SUCCESS_DATA_GET_AUDIENCE_SEGMENTS)


@pytest.fixture
def mock_get_audience_segment_1_response(mocker):
    return _mock_client(mocker, return_value=SUCCESS_DATA_GET_AUDIENCE_SEGMENT_1)


@pytest.fixture
def mock_get_audience_segment_2_response(mocker):
    return _mock_client(mocker, return_value=SUCCESS_DATA_GET_AUDIENCE_SEGMENT_2)
