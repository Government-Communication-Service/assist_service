import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import BinaryIO, List

from sqlalchemy import insert
from sqlalchemy.future import select
from unstructured.documents.elements import (
    Element,
    Text,
)
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.ppt import partition_ppt
from unstructured.partition.pptx import partition_pptx

from app.auth.session_request import SessionRequest
from app.database.models import Document, DocumentChunk, DocumentUserMapping, SearchIndex, User
from app.database.table import async_db_session
from app.document_upload.constants import PERSONAL_DOCUMENTS_INDEX_NAME
from app.opensearch.schemas import OpenSearchRecord
from app.opensearch.service import AsyncOpenSearchOperations

logger = logging.getLogger()


class NoTextContentError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class FileFormatError(Exception):
    def __init__(self, msg: str, file_format: str, supported_formats: List[str]):
        super().__init__(msg)
        self.file_format = file_format
        self.supported_formats = supported_formats


@dataclass
class FileInfo:
    filename: str
    content: BinaryIO


_SUPPORTED_TYPES = [".txt", ".pdf", ".docx", ".csv", ".pptx", ".ppt", ".odt", ".doc", ".xlsx", ".xls", ".html", ".htm"]

_UTF8_INVALID_CHARS_PATTERN = re.compile(
    r"[\x80-\x9F]|"  # Control characters
    r"[\x00-\x1F]"  # Control characters
)


class PersonalDocumentParser:
    """
    Parses file document provided and saves it in the database and opensearch personal_document_uploads index.
    """

    _PROCESSING_TIME_IN_SECS = 118
    """
    The maximum time in seconds for processing uploaded file,
    set lower than network timeout value, which is 120 seconds.
    """

    _PARTITION_DEFAULT_PARAMETERS = {
        "chunking_strategy": "by_title",
        "max_characters": 1500,
        "new_after_n_chars": 800,
        "overlap": 200,
    }
    """ see https://docs.unstructured.io/open-source/core-functionality/chunking for parameters """

    def _parse_file_content(self, file_info: FileInfo) -> List[Element]:
        file_extension = os.path.splitext(file_info.filename)[-1].lower()
        file_extension = file_extension.lower()

        if file_extension not in _SUPPORTED_TYPES:
            raise FileFormatError(
                f"Unsupported file format: {file_extension}",
                file_format=file_extension,
                supported_formats=_SUPPORTED_TYPES,
            )
        # there's a bug in partition function that it fails auto detect some pptx, pdf files.

        if file_extension == ".pdf":
            return partition_pdf(
                file=file_info.content, metadata_filename=file_info.filename, **self._PARTITION_DEFAULT_PARAMETERS
            )
        if file_extension == ".pptx":
            return partition_pptx(
                file=file_info.content, metadata_filename=file_info.filename, **self._PARTITION_DEFAULT_PARAMETERS
            )
        if file_extension == ".ppt":
            return partition_ppt(
                file=file_info.content, metadata_filename=file_info.filename, **self._PARTITION_DEFAULT_PARAMETERS
            )
        return partition(
            file=file_info.content, metadata_filename=file_info.filename, **self._PARTITION_DEFAULT_PARAMETERS
        )

    def _sanitize_text(self, text: str):
        """
        Removes non re
        """
        # text = text.replace('\x00', '')
        return _UTF8_INVALID_CHARS_PATTERN.sub("", text)

    async def process_document(self, file: FileInfo, auth_session: SessionRequest, user: User) -> Document:
        """
        Parses the provided file into its constituent elements (chunks),
        saves both the document metadata and its content chunks to the database, and
        indexes the chunks in OpenSearch.

        Args:
            file (FileInfo): An object containing file metadata such as filename, file extension, and content.
            auth_session (SessionRequest): The auth session that initiated processing the document.
            user (User): The user who uploaded the document, including their user ID.

        Returns:
            Document: The Document that's been saved in the database.

        Raises:
            UnsupportedFileFormatError: if the document isn't supported for chunking.
            DocumentOperationError: If Opensearch bulk insert operation fails.

        Process Flow:
            - Parses the document into elements (chunks)
            - Creates a new document entry in the database.
            - Saves each chunk with a reference to the document and the search index in OpenSearch.
            - Commits the database transaction, saving both the document and its associated chunks.

        Example:
            file_info = FileInfo(filename="file.pdf",  content=b"binary content")
            user = User(id=123, name="John Doe")
            description = "This is a sample document"
            document_id = await process_document(file_info, user, description)

        """

        # generate chunks in thread pool so that asyncio loop isn't blocked.
        processing_fn = asyncio.to_thread(self._parse_file_content, file)
        elements: List[Element] = await asyncio.wait_for(processing_fn, self._PROCESSING_TIME_IN_SECS)

        async with async_db_session() as db_session:
            # save records in a single transaction

            pdu_index_stmt = select(SearchIndex).filter(
                SearchIndex.name == PERSONAL_DOCUMENTS_INDEX_NAME, SearchIndex.deleted_at.is_(None)
            )
            pdu_index_result = await db_session.execute(pdu_index_stmt)
            pdu_index = pdu_index_result.scalar_one()
            filename = self._sanitize_text(file.filename)

            execute = await db_session.execute(insert(Document).values(name=filename).returning(Document))
            new_document = execute.scalar_one()

            await db_session.execute(
                insert(DocumentUserMapping)
                .values(document_id=new_document.id, user_id=user.id, auth_session_id=auth_session.id)
                .returning(DocumentUserMapping)
            )

            # prepare for bulk operations
            documents = []
            for element in elements:
                # Check what content is being passed
                content = element.text if isinstance(element, Text) else str(element)
                content = self._sanitize_text(content)
                if not content:
                    continue
                category = self._sanitize_text(element.category)
                opensearch_record = OpenSearchRecord(
                    new_document.name, new_document.url, category, content, new_document.uuid
                )
                opensearch_doc = opensearch_record.to_opensearch_dict()
                documents.append(opensearch_doc)

            # check if there are document chunks to index, passing empty documents to opensearch causes error.
            if not documents:
                logger.info("No text chunks extracted from document %s", file.filename)
                raise NoTextContentError(f"There is not usable text content in {file.filename} file")

            # save docs in opensearch in bulk.
            response = await AsyncOpenSearchOperations.index_document_chunks(PERSONAL_DOCUMENTS_INDEX_NAME, documents)

            # insert into db in bulk
            items_ = response["items"]
            document_chunks = [
                {
                    "search_index_id": pdu_index.id,
                    "document_id": new_document.id,
                    "name": documents[idx]["chunk_name"],
                    "content": documents[idx]["chunk_content"],
                    "id_opensearch": doc["index"]["_id"],
                }
                for idx, doc in enumerate(items_)
            ]
            await db_session.execute(insert(DocumentChunk).values(document_chunks))

            return new_document
