import logging
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api.endpoints import ENDPOINTS
from app.logs.bugsnag_logger import BugsnagLogger

api = ENDPOINTS()

pytestmark = [
    pytest.mark.general,
]


@pytest.fixture(scope="session")
def app():
    from app.main import app

    return app


@pytest.fixture
def test_client(app):
    return TestClient(app)


@pytest.fixture(scope="session")
def bugsnag_logger():
    bugsnag_logger = BugsnagLogger()
    print("bugsnag_logger: ", bugsnag_logger)
    return bugsnag_logger


async def test_not_found_exception_handler(test_client, default_headers):
    # Test with a non-existent endpoint
    response = test_client.get("/non-existent-endpoint", headers=default_headers)
    print(f"response---: {response}")

    assert response.status_code == 404


# @pytest.mark.asyncio
# async def test_catch_all_exception_handler(bugsnag_logger, default_headers):
#     # Convert headers dict to list of tuples as expected by ASGI scope
#     headers = [[key.encode(), value.encode()] for key, value in default_headers.items()]
#     request = Request(scope={"type": "http", "headers": headers})
#     exc = Exception("Test exception")

#     with patch("bugsnag.notify") as mock_notify:
#         response = await bugsnag_logger._catch_all_exception_handler(request, exc)

#     assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
#     assert response.body == b'{"message":"Internal server error"}'
#     mock_notify.assert_called_once_with(exc)


# @pytest.mark.asyncio
# async def test_validation_exception_handler(bugsnag_logger, default_headers):
#     # Convert headers dict to list of tuples as expected by ASGI scope
#     headers = [[key.encode(), value.encode()] for key, value in default_headers.items()]
#     request = Request(scope={"type": "http", "headers": headers})
#     exc = RequestValidationError(
#         errors=[{"loc": ("body", "field"), "msg": "field required", "type": "value_error.missing"}],
#     )

#     with patch("bugsnag.notify") as mock_notify:
#         response = await bugsnag_logger._validation_exception_handler(request, exc)

#     assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
#     assert "detail" in response.body.decode()
#     mock_notify.assert_called_once_with(exc)


def test_setup_bugsnag(app):
    with patch("bugsnag.handlers.BugsnagHandler.emit"):
        logger = logging.getLogger()
        logger.error("Test error")


def test_bugsnag_logger_initialization_without_api_key():
    with pytest.raises(Exception, match="BUGSNAG_API_KEY is required"):
        with patch.dict(os.environ, {"BUGSNAG_API_KEY": "", "BUGSNAG_RELEASE_STAGE": "test"}):
            BugsnagLogger()


def test_bugsnag_logger_initialization_without_release_stage():
    with pytest.raises(Exception, match="BUGSNAG_RELEASE_STAGE is required"):
        with patch.dict(os.environ, {"BUGSNAG_RELEASE_STAGE": ""}):
            BugsnagLogger()


def test_bugsnag_logger_disabled():
    with patch.dict(os.environ, {"BUGSNAG_RELEASE_STAGE": "test", "DISABLE_BUGSNAG_LOGGING": "True"}):
        logger = BugsnagLogger()
        assert not logger.BUGSNAG_ENABLED
