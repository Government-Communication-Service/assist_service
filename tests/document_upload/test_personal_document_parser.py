import asyncio
import io

import anyio
import pytest

from app.database.models import AuthSession, User
from app.document_upload.personal_document_parser import FileFormatError, FileInfo, PersonalDocumentParser


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


def test_parse_empty_csv_returns_no_elements():
    """Test that an empty .csv file produces no elements."""
    parser = PersonalDocumentParser()
    with open("tests/resources/empty.csv", "rb") as f:
        file_info = FileInfo(filename="empty.csv", content=io.BytesIO(f.read()))
        elements = parser._parse_file_content(file_info)

        assert len(elements) == 0


def test_parse_csv_file_returns_elements():
    """Test that .csv files are parsed and return elements."""
    parser = PersonalDocumentParser()
    with open("tests/resources/username.csv", "rb") as f:
        file_info = FileInfo(filename="username.csv", content=io.BytesIO(f.read()))
        elements = parser._parse_file_content(file_info)

        assert len(elements) > 0
        assert all(hasattr(elem, "text") for elem in elements)

        all_text = " ".join([elem.text for elem in elements if hasattr(elem, "text") and elem.text])
        assert "Rachel" in all_text
        assert "Laura" in all_text
        assert "Craig" in all_text
        assert "Jamie" in all_text


def test_parse_xlsx_file_returns_elements():
    """Test that .xlsx files are parsed and return elements."""
    parser = PersonalDocumentParser()
    with open("tests/resources/user_details.xlsx", "rb") as f:
        file_info = FileInfo(filename="user_details.xlsx", content=io.BytesIO(f.read()))
        elements = parser._parse_file_content(file_info)

        assert len(elements) > 0
        assert all(hasattr(elem, "text") for elem in elements)

        # Verify actual department data from the Excel file is extracted
        all_text = " ".join([elem.text for elem in elements if hasattr(elem, "text") and elem.text])
        assert "Marketing" in all_text
        assert "Engineering" in all_text
        assert "Sales" in all_text
        assert "Human Resources" in all_text
        assert "Product" in all_text


def test_parse_xls_file_raises_file_format_error():
    """Test that .xls files raise FileFormatError as the format is not supported."""
    parser = PersonalDocumentParser()
    file_info = FileInfo(filename="user_details.xls", content=io.BytesIO(b""))
    with pytest.raises(FileFormatError) as exc_info:
        parser._parse_file_content(file_info)
    assert exc_info.value.file_format == ".xls"
