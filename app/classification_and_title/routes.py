# ruff: noqa: B008
from fastapi import APIRouter, Depends

from app.api.endpoints import ENDPOINTS
from app.auth.verify_service import verify_auth_token
from app.classification_and_title.schemas import (
    BulkSyncResult,
    ClassificationInput,
    ClassificationResponse,
    ClassificationsResponse,
)
from app.classification_and_title.service import bulk_sync_classifications, get_classifications
from app.database.table import async_db_session

router = APIRouter()


@router.get(
    path=ENDPOINTS.CLASSIFICATIONS,
    dependencies=[Depends(verify_auth_token)],
)
async def list_classifications() -> ClassificationsResponse:
    async with async_db_session() as db_session:
        rows = await get_classifications(db_session)
    return ClassificationsResponse(classifications=[ClassificationResponse(**row) for row in rows])


@router.post(
    path=ENDPOINTS.CLASSIFICATIONS_BULK,
    dependencies=[Depends(verify_auth_token)],
)
async def bulk_upload_classifications(classifications: list[ClassificationInput]) -> BulkSyncResult:
    """Sync the canonical classification list.

    - New titles are created.
    - Existing titles with a changed description are updated.
    - Titles absent from the list are soft-deleted (deprecated).
    """
    async with async_db_session() as db_session:
        result = await bulk_sync_classifications(db_session, classifications)
    return BulkSyncResult(**result)
