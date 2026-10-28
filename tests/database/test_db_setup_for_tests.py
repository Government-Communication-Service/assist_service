import logging
import os

from app.config import settings

logger = logging.getLogger(__name__)


def test_db_setup_for_tests(test_db):
    logger.debug(f"TEST_POSTGRES_DB set to {test_db}")
    logger.debug(f"POSTGRES_DB set to {settings.postgres_db}")
    if os.getenv("TEST_POSTGRES_DB") and settings.postgres_db:
        assert os.getenv("TEST_POSTGRES_DB") == test_db
        assert settings.postgres_db == test_db
