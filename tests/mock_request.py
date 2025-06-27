import pytest

from app.logs.logs_handler import logger


def fail_test(message, exception=None):
    """Utility function to log errors and fail the pytest."""
    if exception:
        message += f": {exception}"
    logger.error(message)
    pytest.fail(message)
