import logging
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select, update

from app.database.models import Document, DocumentChunk, DocumentUserMapping
from app.document_upload.document_management import _delete_expired_files

logger = logging.getLogger()


@pytest.mark.asyncio
async def test_delete_expired_documents_success(
    async_client, user_id, async_http_requester, db_session_provider, db_session, file_uploader, auth_session, caplog
):
    """
    Tests marking expired documents as deleted, when the expiry time is over today's date, when
    there is an expired document at least.

    Asserts:
        The user document mapping, document and document chunks are marked as deleted.
    """

    file_path = "tests/resources/random-topics.pdf"
    response = await file_uploader(file_path, "test_delete_document")
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

    # trigger document deletion process
    await _delete_expired_files()

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

    # assert logs
    assert "Marked 5 expired document chunk(s) as deleted." in caplog.text
    assert "Successfully deleted 5 document chunk(s) from OpenSearch." in caplog.text


@pytest.mark.asyncio
async def test_delete_expired_documents_when_no_expired_documents_found(
    async_client, user_id, async_http_requester, db_session_provider, db_session, file_uploader, auth_session, caplog
):
    """
    Tests document deletion process, when there aren't any expired documents

    Asserts:
        No document, chunks and opensearch documents are deleted.
    """
    # trigger document deletion process
    await _delete_expired_files()

    # assert logs
    assert "Marked 0 expired document chunk(s) as deleted." in caplog.text
    assert "No opensearch ids found for deletion from opensearch." in caplog.text
