import logging
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select, update

from app.api.endpoints import ENDPOINTS
from app.database.models import Document, DocumentChunk, DocumentUserMapping
from app.document_upload.schemas import DocumentCleanupResponse

logger = logging.getLogger(__name__)
api = ENDPOINTS()

pytestmark = [
    pytest.mark.document_upload,
    pytest.mark.unit,
]


@pytest.mark.asyncio
async def test_delete_expired_documents_endpoint_success(
    async_client, user_id, async_http_requester, db_session_provider, db_session, file_uploader
):
    """
    Tests the endpoint for deleting expired documents.

    Asserts:
        The user document mapping, document and document chunks are marked as deleted.
    """
    file_path = "tests/resources/random-topics.pdf"
    response = await file_uploader(file_path, "test_delete_document_endpoint")
    document_uuid = response["document_uuid"]

    async with db_session_provider() as db_session1:
        result = await db_session1.execute(
            select(Document).filter(Document.uuid == document_uuid, Document.deleted_at.is_(None))
        )
        document = result.scalars().first()

        # mark document as expired manually
        await db_session1.execute(
            update(DocumentUserMapping)
            .where(DocumentUserMapping.document_id == document.id)
            .values(expired_at=(datetime.now() - timedelta(days=180)))
        )

    # Call the cleanup endpoint
    cleanup_url = api.build_url(api.DOCUMENT_CLEANUP_EXPIRED)
    cleanup_response = await async_http_requester(
        "test_delete_expired_documents_endpoint_success",
        async_client.delete,
        cleanup_url,
    )

    # Verify response
    response_obj = DocumentCleanupResponse(**cleanup_response)
    assert response_obj.status == "success"
    assert response_obj.deleted_count == 5  # 5 chunks from the PDF

    # assert document mapping
    result = await db_session.execute(
        select(DocumentUserMapping).filter(
            DocumentUserMapping.document_id == document.id, DocumentUserMapping.deleted_at.is_(None)
        )
    )
    document_mapping = result.scalars().all()
    assert document_mapping == []

    # assert document mark as deleted
    result = await db_session.execute(
        select(Document).filter(Document.id == document.id, Document.deleted_at.is_(None))
    )
    document_record = result.scalars().first()
    assert document_record is None

    # assert document chunks marked as deleted
    result = await db_session.execute(
        select(DocumentChunk).filter(DocumentChunk.document_id == document.id, DocumentChunk.deleted_at.is_(None))
    )
    document_chunks = result.scalars().all()
    assert document_chunks == []


@pytest.mark.asyncio
async def test_delete_expired_documents_endpoint_when_no_expired_documents(
    async_client, async_http_requester
):
    """
    Tests document deletion endpoint when there aren't any expired documents.

    Asserts:
        No documents are deleted and response indicates zero deletions.
    """
    # Call the cleanup endpoint
    cleanup_url = api.build_url(api.DOCUMENT_CLEANUP_EXPIRED)
    cleanup_response = await async_http_requester(
        "test_delete_expired_documents_endpoint_when_no_expired_documents",
        async_client.delete,
        cleanup_url,
    )

    # Verify response
    response_obj = DocumentCleanupResponse(**cleanup_response)
    assert response_obj.status == "success"
    assert response_obj.deleted_count == 0
