import logging
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


def test_setup_bugsnag(app):
    with patch("bugsnag.handlers.BugsnagHandler.emit"):
        logger = logging.getLogger()
        logger.error("Test error")


def test_bugsnag_logger_initialization_without_api_key():
    with pytest.raises(Exception, match="BUGSNAG_API_KEY is required"):
        with patch("app.logs.BUGSNAG_API_KEY", None):
            with patch("app.logs.BUGSNAG_RELEASE_STAGE", "test"):
                BugsnagLogger()


def test_bugsnag_logger_initialization_without_release_stage():
    with pytest.raises(Exception, match="BUGSNAG_RELEASE_STAGE is required"):
        with patch("app.logs.BUGSNAG_RELEASE_STAGE", None):
            BugsnagLogger()


def test_bugsnag_logger_disabled():
    with patch("app.logs.BUGSNAG_RELEASE_STAGE", "test"):
        with patch("app.logs.BUGSNAG_API_KEY", "test-key"):
            with patch("app.logs.DISABLE_BUGSNAG_LOGGING", True):
                logger = BugsnagLogger()
                assert not logger.BUGSNAG_ENABLED
