import logging

from app.config import settings

logger = logging.getLogger(__name__)


def database_url() -> str:
    pw = settings.postgres_password.get_secret_value()
    logger.debug("Using sync database url")
    return f"postgresql+psycopg2://{settings.postgres_user}:{pw}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"


def async_database_url() -> str:
    pw = settings.postgres_password.get_secret_value()
    logger.debug("Using async database url")
    return f"postgresql+asyncpg://{settings.postgres_user}:{pw}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
