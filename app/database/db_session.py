import logging
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.table import async_db_session

logger = logging.getLogger()


async def get_db_session() -> AsyncSession:
    """
    This function yields an asynchronous session.
    It ensures that a new session is created for each request and properly
    managed within an asynchronous context.

    Don't depend on this function directly: use the `DbSession` alias below.

    Yields:
        AsyncSession: An asynchronous session for database operations.

    Example:
        async def read_items(db_session: DbSession):

    """
    async with async_db_session() as session:
        yield session


# Function scope closes (and commits) the session when the endpoint function returns, so:
# - writes are committed before the client gets its 2xx;
# - a StreamingResponse doesn't hold the connection open while the body streams.
# Use this alias at every site: the dependency cache key includes the scope, so mixing it with a
# plain Depends(get_db_session) in one request would give two separate sessions.
DbSession = Annotated[AsyncSession, Depends(get_db_session, scope="function")]
