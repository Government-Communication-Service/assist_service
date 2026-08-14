import asyncio
import io
from unittest.mock import patch

import anyio
import pytest
from pandas.errors import ParserError
from sqlalchemy import delete, select
from unstructured.documents.elements import ElementMetadata, Table, Text

from app.database.models import AuthSession, Document, DocumentChunk, DocumentUserMapping, User
from app.document_upload.constants import PERSONAL_DOCUMENTS_INDEX_NAME
from app.document_upload.personal_document_parser import (
    DocumentParsingError,
    FileFormatError,
    FileInfo,
    PersonalDocumentParser,
)
from app.opensearch.service import AsyncOpenSearchOperations


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


def test_sanitize_text_preserves_newlines_tabs():
    """
    Regression test: the sanitize regex must strip control characters but preserve
    \\n/\\t/\\r, otherwise a Markdown table (which relies on newlines for row
    separation) would be silently collapsed into one unreadable line.
    """
    parser = PersonalDocumentParser()
    text = "line1\nline2\tcol\rline3\x00\x1b"

    result = parser._sanitize_text(text)

    assert result == "line1\nline2\tcol\rline3"


@pytest.mark.parametrize(
    "file_extension, patch_target, side_effect",
    [
        (".xlsx", "app.document_upload.personal_document_parser.partition_xlsx", IndexError("list index out of range")),
        (
            ".csv",
            "app.document_upload.personal_document_parser.partition",
            UnicodeDecodeError("utf-8", b"\x93", 0, 1, "invalid start byte"),
        ),
        (
            ".csv",
            "app.document_upload.personal_document_parser.partition",
            ParserError("Error tokenizing data. C error: Expected 1 fields in line 2, saw 2"),
        ),
    ],
)
def test_parse_malformed_file_raises_document_parsing_error(file_extension, patch_target, side_effect):
    """
    Regression test for three real production tracebacks (openpyxl IndexError on a
    corrupt stylesheet, CSV UnicodeDecodeError, CSV ParserError) that used to escape
    _parse_file_content unhandled and become noisy 500s. They must now be wrapped in
    DocumentParsingError so the endpoint can return a clean 4xx.
    """
    parser = PersonalDocumentParser()
    file_info = FileInfo(filename=f"malformed{file_extension}", content=io.BytesIO(b""))

    with patch(patch_target, side_effect=side_effect):
        with pytest.raises(DocumentParsingError) as exc_info:
            parser._parse_file_content(file_info)

    assert exc_info.value.cause is side_effect


@pytest.mark.parametrize(
    "csv_bytes",
    [
        b"name\nalice\nbob\n",  # single-column: csv.Sniffer can't detect a delimiter by design
        b"a\tb\n1\t2\n",  # tab-delimited
    ],
)
def test_parse_csv_with_no_detectable_delimiter_does_not_raise(csv_bytes):
    """
    Regression test: unstructured's partition_csv falls back to pd.read_csv(sep=None)
    whenever csv.Sniffer can't detect a delimiter (single-column/tab-delimited input).
    pandas 3.x's C parser raises a bare TypeError for sep=None instead of falling back
    to the python engine like pre-3.0 did, which used to escape this method as an
    unhandled 500. requirements.txt now pins pandas<3.0 to fix this at the dependency
    level - this test guards against a future bump reintroducing it.
    """
    parser = PersonalDocumentParser()
    file_info = FileInfo(filename="odd.csv", content=io.BytesIO(csv_bytes))

    elements = parser._parse_file_content(file_info)

    assert len(elements) > 0


def test_parse_ragged_csv_raises_document_parsing_error():
    """
    With pandas pinned <3.0 (see test_parse_csv_with_no_detectable_delimiter_does_not_raise),
    a CSV with inconsistent row lengths now surfaces as a genuine pandas.errors.ParserError
    (via the python-engine fallback) instead of an uncaught TypeError - so it's cleanly
    caught by the existing ParserError handling rather than escaping as a raw 500.
    """
    parser = PersonalDocumentParser()
    file_info = FileInfo(filename="ragged.csv", content=io.BytesIO(b"name,age\nalice,30,extra,more\nbob,40\n"))

    with pytest.raises(DocumentParsingError) as exc_info:
        parser._parse_file_content(file_info)

    assert isinstance(exc_info.value.cause, ParserError)


def test_parse_pdf_indexerror_is_not_wrapped_as_document_parsing_error():
    """
    Regression test: the IndexError catch in _parse_file_content is scoped to the
    .xlsx branch only (it targets openpyxl's stylesheet-parsing bug). An IndexError
    from any other partitioner must propagate unchanged, not be misreported to the
    user as a corrupted-file 400.
    """
    parser = PersonalDocumentParser()
    file_info = FileInfo(filename="doc.pdf", content=io.BytesIO(b""))

    with patch(
        "app.document_upload.personal_document_parser.partition_pdf",
        side_effect=IndexError("unrelated bug"),
    ):
        with pytest.raises(IndexError):
            parser._parse_file_content(file_info)


@pytest.mark.asyncio
async def test_process_document_batches_large_chunk_inserts(user, auth_session, db_session_provider):
    """
    A document that produces more chunks than fit in a single INSERT (asyncpg caps bind
    parameters at 32767, i.e. ~6553 rows for DocumentChunk's 5 columns) must still upload
    successfully now that the insert is batched. Regression test for a complex/many-sheet
    xlsx workbook producing enough chunks to exceed that limit.
    """
    parser = PersonalDocumentParser()
    num_chunks = 7000
    elements = [Text(f"chunk content {i}") for i in range(num_chunks)]
    file_info = FileInfo(filename="large_workbook.xlsx", content=io.BytesIO(b""))

    document = None
    opensearch_ids: list[str] = []
    try:
        with patch.object(parser, "_parse_file_content", return_value=elements):
            document = await parser.process_document(file_info, auth_session=auth_session, user=user)

        async with db_session_provider() as db_session:
            result = await db_session.execute(select(DocumentChunk).filter(DocumentChunk.document_id == document.id))
            chunks = result.scalars().all()
            opensearch_ids = [chunk.id_opensearch for chunk in chunks]

        assert len(chunks) == num_chunks
    finally:
        if document is not None:
            async with db_session_provider() as db_session:
                await db_session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
                await db_session.execute(
                    delete(DocumentUserMapping).where(DocumentUserMapping.document_id == document.id)
                )
                await db_session.execute(delete(Document).where(Document.id == document.id))
                await db_session.commit()
            if opensearch_ids:
                await AsyncOpenSearchOperations.delete_document_chunks(PERSONAL_DOCUMENTS_INDEX_NAME, opensearch_ids)


@pytest.mark.asyncio
async def test_process_document_splits_oversized_table_into_multiple_chunks(user, auth_session, db_session_provider):
    """
    A single Table element whose text_as_html renders to a Markdown table bigger than
    MAX_TABLE_CHUNK_CHARS must be split into multiple DocumentChunk rows, each carrying
    the header row, rather than becoming one giant unbounded chunk (the LLM-context
    inefficiency this fix addresses).
    """
    from app.config import MAX_TABLE_CHUNK_CHARS

    header_html = "<tr><td>col1</td><td>col2</td></tr>"
    rows_html = "".join(f"<tr><td>row{i}</td><td>val{i}</td></tr>" for i in range(200))
    table_html = f"<table>{header_html}{rows_html}</table>"
    element = Table(text="ignored", metadata=ElementMetadata(text_as_html=table_html))

    parser = PersonalDocumentParser()
    file_info = FileInfo(filename="large_table.xlsx", content=io.BytesIO(b""))

    document = None
    opensearch_ids: list[str] = []
    try:
        with patch.object(parser, "_parse_file_content", return_value=[element]):
            document = await parser.process_document(file_info, auth_session=auth_session, user=user)

        async with db_session_provider() as db_session:
            result = await db_session.execute(select(DocumentChunk).filter(DocumentChunk.document_id == document.id))
            chunks = result.scalars().all()
            opensearch_ids = [chunk.id_opensearch for chunk in chunks]

        assert len(chunks) > 1
        for chunk in chunks:
            assert "col1" in chunk.content
            assert "col2" in chunk.content
            assert len(chunk.content) <= MAX_TABLE_CHUNK_CHARS
    finally:
        if document is not None:
            async with db_session_provider() as db_session:
                await db_session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
                await db_session.execute(
                    delete(DocumentUserMapping).where(DocumentUserMapping.document_id == document.id)
                )
                await db_session.execute(delete(Document).where(Document.id == document.id))
                await db_session.commit()
            if opensearch_ids:
                await AsyncOpenSearchOperations.delete_document_chunks(PERSONAL_DOCUMENTS_INDEX_NAME, opensearch_ids)
