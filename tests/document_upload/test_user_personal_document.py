import logging
from datetime import datetime
from typing import Dict, List
from unittest.mock import patch
from uuid import UUID

import anyio
import pytest
from sqlalchemy.future import select

from app.api.endpoints import ENDPOINTS
from app.database.models import Document, DocumentChunk

logger = logging.getLogger()


def assert_user_docs(docs: List[Dict]):
    for doc in docs:
        assert doc["name"] is not None
        assert UUID(doc["uuid"]) is not None
        assert isinstance(datetime.fromisoformat(doc.get("created_at")), datetime)
        assert isinstance(datetime.fromisoformat(doc.get("expired_at")), datetime)


def assert_central_docs(docs: List[Dict]):
    for doc in docs:
        assert doc.get("name") is not None
        assert UUID(doc.get("uuid")) is not None
        assert isinstance(datetime.fromisoformat(doc.get("created_at")), datetime)
        assert doc.get("expired_at") is None


async def assert_uploaded_file(db_session, document_uuid):
    execute = await db_session.execute(select(Document).filter_by(uuid=document_uuid))
    document = execute.scalar()
    assert document is not None
    execute = await db_session.execute(select(DocumentChunk).filter_by(document_id=document.id))
    chunks = execute.scalars().all()
    assert chunks is not None


async def upload_file(api_endpoint, async_client, async_http_requester, doc, response_code: int = 200) -> Dict:
    async with await anyio.open_file(doc, "rb") as f:
        return await async_http_requester(
            f"uploading file: {doc}",
            async_client.post,
            api_endpoint,
            response_code=response_code,
            files={"file": (doc, await f.read())},
        )


@pytest.mark.parametrize(
    "test_scenario, file_path, expected_status_code, expected_error",
    [
        ("test_uploading_word_document", "tests/resources/random-topics.docx", 200, None),
        ("test_uploading_pdf_document", "tests/resources/random-topics.pdf", 200, None),
        ("test_uploading_csv_document", "tests/resources/username.csv", 200, None),
        ("test_uploading_text_document", "tests/resources/sample3.txt", 200, None),
        ("test_uploading_powerpoint_document", "tests/resources/samplepptx.pptx", 200, None),
        ("test_uploading_odt_document", "tests/resources/random-topics.odt", 200, None),
        ("test_uploading_word_doc_document", "tests/resources/sample_doc_document.doc", 200, None),
        (
            "test_unknown_file_format",
            "tests/resources/sample_binary_file.bin",
            400,
            {
                "error_code": "FILE_FORMAT_NOT_SUPPORTED",
                "supported_formats": [
                    ".txt",
                    ".pdf",
                    ".docx",
                    ".csv",
                    ".pptx",
                    ".ppt",
                    ".odt",
                    ".doc",
                    ".xlsx",
                    ".xls",
                    ".html",
                    ".htm",
                ],
                "status": "failed",
                "status_message": "Unsupported file format: .bin",
            },
        ),
        (
            "test_ocr_format",
            "tests/resources/Non-text-searchable.pdf",
            400,
            {
                "error_code": "DOCUMENTS_REQUIRING_OCR_NOT_SUPPORTED",
                "status": "failed",
                "status_message": "This document does not contain any text."
                "It may contain scanned text or images of text, but Assist cannot process these. "
                "Please upload a document that contains the information in text format.",
            },
        ),
        (
            "test_no_text_document",
            "tests/resources/no-text-document.docx",
            400,
            {
                "error_code": "NO_TEXT_CONTENT_ERROR",
                "status": "failed",
                "status_message": "There is not usable text content in tests/resources/no-text-document.docx file",
            },
        ),
        (
            "invalid_word_document",
            "tests/resources/invalid_word_document.docx",
            400,
            {
                "error_code": "UNSUPPORTED_WORD_DOCUMENT_VERSION",
                "status": "failed",
                "status_message": "The file uploaded is either not a word document, "
                "or was generated with an older Word version,"
                "Please use latest Word version or upload the document in PDF format",
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_uploading_documents(
    test_scenario,
    file_path,
    expected_status_code,
    expected_error,
    async_client,
    user_id,
    async_http_requester,
    db_session,
):
    """
    Tests uploading a Word document.

    Asserts:
        The document and its associated chunks are successfully added to the database.
    """
    logger.info("Running test_uploading_documents, test scenario %s", test_scenario)
    api = ENDPOINTS()
    api_endpoint = api.user_documents(user_id)
    response = await upload_file(
        api_endpoint, async_client, async_http_requester, file_path, response_code=expected_status_code
    )

    if expected_status_code == 200:
        # get document_id from response
        document_uuid = response["document_uuid"]
        # check document table is populated
        await assert_uploaded_file(db_session, document_uuid)
    else:
        assert expected_error == response


async def test_list_user_documents(async_client, user_id, async_http_requester):
    """
    Tests list user documents api. It should return a dictionary containing user_documents and central_documents
    list of documents with name, uuid, created_at and expired_at fields.

    Asserts:
        user_documents field contains documents uploaded by the user.
        central_documents field contains documents available to all users.

    """
    docs_to_upload = ["tests/resources/random-topics.docx", "tests/resources/random-topics.pdf"]
    api = ENDPOINTS()
    api_endpoint = api.user_documents(user_id)
    for doc in docs_to_upload:
        await upload_file(api_endpoint, async_client, async_http_requester, doc)

    response = await async_http_requester(
        "list_user_documents",
        async_client.get,
        api_endpoint,
    )

    assert response["central_documents"]

    user_documents = response["user_documents"]
    assert len(user_documents) == 2
    assert_user_docs(user_documents)


@pytest.mark.asyncio
@patch("app.document_upload.personal_document_parser.PersonalDocumentParser._PROCESSING_TIME_IN_SECS", new=0.01)
async def test_timeout_for_processing_large_files(async_client, user_id, async_http_requester):
    """
    Uploads a large file and verifies that the API responds with http 400 status and error code below
    when the file processing exceeds the allowed timeout period.

    It changes processing timeout value to 0.01 temporarily to reduce test execution time.

    """
    doc_to_upload = "tests/resources/random-topics.pdf"
    api = ENDPOINTS()
    api_endpoint = api.user_documents(user_id)

    expected_response = {
        "error_code": "FILE_PROCESSING_TIMEOUT_ERROR",
        "status": "failed",
        "status_message": "Uploading document timed out, please try again",
    }
    result = await upload_file(api_endpoint, async_client, async_http_requester, doc_to_upload, response_code=400)
    assert expected_response == result
