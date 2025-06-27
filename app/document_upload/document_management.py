import asyncio
import logging

from app.database.db_operations import DbOperations
from app.database.table import async_db_session
from app.document_upload.constants import PERSONAL_DOCUMENTS_INDEX_NAME
from app.opensearch.service import AsyncOpenSearchOperations

logger = logging.getLogger(__name__)

SLEEP_TIME = 3600  # 1 hour in seconds


async def _delete_expired_files():
    """
    Delete expired files from the database and OpenSearch.
    """
    async with async_db_session() as db_session:
        opensearch_ids = await DbOperations.delete_expired_documents(db_session)
        logger.info("Marked %s expired document chunk(s) as deleted.", len(opensearch_ids))
        if opensearch_ids:
            await AsyncOpenSearchOperations.delete_document_chunks(PERSONAL_DOCUMENTS_INDEX_NAME, opensearch_ids)
            logger.info("Successfully deleted %s document chunk(s) from OpenSearch.", len(opensearch_ids))
        else:
            logger.info("No opensearch ids found for deletion from opensearch.")


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
