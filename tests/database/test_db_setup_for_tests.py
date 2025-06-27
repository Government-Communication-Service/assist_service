import logging
import os

logger = logging.getLogger(__name__)


def test_db_setup_for_tests(test_db):
    logger.debug(f"TEST_POSTGRES_DB set to {test_db}")
    logger.debug(f"POSTGRES_DB set to {os.getenv('POSTGRES_DB')}")
    if os.getenv("TEST_POSTGRES_DB") and os.getenv("POSTGRES_DB"):
        assert os.getenv("TEST_POSTGRES_DB") == test_db
        assert os.getenv("POSTGRES_DB") == test_db
