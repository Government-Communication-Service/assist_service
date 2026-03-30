import asyncio
import logging

from app.database.table import async_db_session
from app.document_upload.service import clean_expired_documents

logger = logging.getLogger(__name__)

SLEEP_TIME = 3600  # 1 hour in seconds


async def _delete_expired_files():
    """
    Delete expired files from the database and OpenSearch.
    """
    async with async_db_session() as db_session:
        await clean_expired_documents(db_session)
        await db_session.commit()


async def schedule_expired_files_deletion():
    """
    Schedules the periodic execution of the expired file deletion process.
    The process runs every hour and checks if there are expired documents to delete from database and opensearch
    """
    while True:
        try:
            logger.info("Running scheduled expired documents deletion process")
            await _delete_expired_files()
        except Exception as e:
            logger.exception("An error occurred during expired documents deletion: %s", e)
        logger.info("Sleeping for %s seconds before the next document deletion run.", SLEEP_TIME)
        await asyncio.sleep(SLEEP_TIME)
