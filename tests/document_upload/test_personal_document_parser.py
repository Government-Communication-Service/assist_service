import asyncio
import io

import anyio
import pytest

from app.database.models import AuthSession, User
from app.document_upload.personal_document_parser import FileInfo, PersonalDocumentParser


async def test_timeout_uploading_large_files():
    """
    Checks timeout error is raised when a large file is processed.
    """
    doc_parser = PersonalDocumentParser()
    # adjust processing time for test
    doc_parser._PROCESSING_TIME_IN_SECS = 0.01
    session_request = AuthSession(id=1, user_id=1)
    user = User(id=1)
    file_path = "tests/resources/DNA_Topics_UK.docx"
    async with await anyio.open_file(file_path, "rb") as f:
        content = await f.read()
        file_info = FileInfo("DNA_Topics_UK.docx", io.BytesIO(content))
        with pytest.raises(asyncio.TimeoutError):
            await doc_parser.process_document(file_info, auth_session=session_request, user=user)
