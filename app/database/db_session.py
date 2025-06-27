import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.table import async_db_session

logger = logging.getLogger()


async def get_db_session() -> AsyncSession:
    """
    This function yields an asynchronous session.
    It ensures that a new session is created for each request and properly
    managed within an asynchronous context.

    Yields:
        AsyncSession: An asynchronous session for database operations.

    Example:
        async def read_items(db: AsyncSession = Depends(get_db_session)):

    """
    async with async_db_session() as session:
        yield session
