import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints import ENDPOINTS
from app.auth.verify_service import verify_auth_token
from app.database.db_session import get_db_session
from app.document_upload.schemas import DocumentCleanupResponse
from app.document_upload.service import clean_expired_documents

router = APIRouter()

logger = logging.getLogger(__name__)


@router.delete(
    path=ENDPOINTS.DOCUMENT_CLEANUP_EXPIRED,
    dependencies=[Depends(verify_auth_token)],
    response_model=DocumentCleanupResponse,
)
async def delete_expired_documents(db_session: AsyncSession = Depends(get_db_session)):
    """
    Remove expired documents from the database and OpenSearch for data protection compliance.

    This endpoint cleans up document chunks that have expired based on their expiry date.
    Authentication token is required to call this endpoint.

    Returns:
        DocumentCleanupResponse: Response containing the number of documents cleaned and success status.
    """
    logger.info("Calling delete expired documents")

    try:
        cleanup_data = await clean_expired_documents(db_session)
        deleted = cleanup_data["deleted_count"]
        failed = cleanup_data.get("failed_count", 0)

        if deleted == 0 and failed > 0:
            message = f"OpenSearch deletion failed for {failed} expired document(s)"
            logger.error(message)
            raise HTTPException(status_code=500, detail=message)

        return DocumentCleanupResponse(
            message=f"Cleanup complete. Deleted: {deleted}, Failed: {failed}",
            deleted_count=deleted
        )
    except Exception as ex:
        detail_message = f"Failed to delete expired documents: {str(ex)}"
        logger.exception(detail_message)
        raise HTTPException(status_code=500, detail=detail_message) from ex
