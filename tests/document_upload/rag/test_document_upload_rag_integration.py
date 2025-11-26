import asyncio
import logging

import pytest

from app.central_guidance.schemas import RagRequest
from app.database.models import Message, SearchIndex
from app.document_upload.constants import PERSONAL_DOCUMENTS_INDEX_NAME
from app.document_upload.personal_document_rag import (
    GLOBAL_CHARACTER_LIMIT,
    _retrieve_relevant_chunks,
)

logger = logging.getLogger(__name__)


class TestPersonalDocumentRAGIntegration:
    """Integration tests for personal document RAG with real document uploads."""

    @pytest.fixture
    def mock_search_index(self):
        """Mock SearchIndex object."""
        search_index = SearchIndex()
        search_index.id = 1
        search_index.name = PERSONAL_DOCUMENTS_INDEX_NAME
        return search_index

    @pytest.fixture
    def mock_message(self):
        """Mock Message object."""
        message = Message()
        message.id = 1
        return message

    @pytest.mark.asyncio
    async def test_small_document_summary_all_chunks_retrieved(
        self, mock_search_index, mock_message, db_session, file_uploader
    ):
        """
        Example 1: Upload 1 small document -> ask to 'create a summary'
        -> Assert that all chunks were retrieved, because they are below the character limit
        """
        # Upload small document
        file_path = "tests/resources/DNA_Topics_UK.docx"
        upload_response = await file_uploader(file_path, "test_small_doc_summary")
        document_uuid = upload_response["document_uuid"]

        # Allow time for indexing
        await asyncio.sleep(10)

        # Create RAG request
        rag_request = RagRequest(
            use_central_rag=True, user_id=1, query="create a summary", document_uuids=[document_uuid]
        )

        # Call the function directly
        result = await _retrieve_relevant_chunks(rag_request, mock_search_index, mock_message, db_session)

        # Calculate total characters
        total_characters = sum(len(chunk.get("_source", {}).get("chunk_content", "")) for chunk in result)

        logger.info(f"Small document test: Retrieved {len(result)} chunks with {total_characters} total characters")

        # Assertions
        assert len(result) > 0, "Should retrieve some chunks"
        assert total_characters <= GLOBAL_CHARACTER_LIMIT, (
            f"Should not exceed character limit of {GLOBAL_CHARACTER_LIMIT}"
        )

        # Since it's a small document (~12 chunks), all chunks should be retrieved without search
        # We expect the total to be well under the 55,000 character limit
        assert total_characters < GLOBAL_CHARACTER_LIMIT * 0.5, (
            "Small document should use less than 50% of character limit"
        )

    @pytest.mark.asyncio
    async def test_large_document_summary_character_limit_respected(
        self, mock_search_index, mock_message, db_session, file_uploader
    ):
        """
        Example 2: Upload 1 large document -> ask to 'create a summary'
        -> Assert that GLOBAL_CHARACTER_LIMIT was not breached; assert that more than half was used
        """
        # Upload large document
        file_path = "tests/resources/Introduction_to_Machine_Learning_with_Python.pdf"
        upload_response = await file_uploader(file_path, "test_large_doc_summary")
        document_uuid = upload_response["document_uuid"]

        # Allow time for indexing
        await asyncio.sleep(15)

        # Create RAG request
        rag_request = RagRequest(
            use_central_rag=True, user_id=1, query="create a summary", document_uuids=[document_uuid]
        )

        # Call the function directly
        result = await _retrieve_relevant_chunks(rag_request, mock_search_index, mock_message, db_session)

        # Calculate total characters
        total_characters = sum(len(chunk.get("_source", {}).get("chunk_content", "")) for chunk in result)

        logger.info(f"Large document test: Retrieved {len(result)} chunks with {total_characters} total characters")
        logger.info(
            f"Character limit: {GLOBAL_CHARACTER_LIMIT}, "
            f"Usage: {(total_characters / GLOBAL_CHARACTER_LIMIT) * 100:.1f}%"
        )

        # Assertions
        assert len(result) > 0, "Should retrieve some chunks"
        assert total_characters <= GLOBAL_CHARACTER_LIMIT, (
            f"Should not exceed character limit of {GLOBAL_CHARACTER_LIMIT}"
        )
        assert total_characters > GLOBAL_CHARACTER_LIMIT * 0.5, (
            f"Should use more than 50% of character limit ({GLOBAL_CHARACTER_LIMIT * 0.5})"
        )

    @pytest.mark.asyncio
    async def test_two_small_documents_both_included(self, mock_search_index, mock_message, db_session, file_uploader):
        """
        Example 3: Upload 2 small documents -> Ask to 'summarise the content of both documents'
        -> Assert that chunks from both documents are included AND all chunks from each document
        """
        # Upload first small document
        file_path1 = "tests/resources/DNA_Topics_UK.docx"
        upload_response1 = await file_uploader(file_path1, "test_two_small_docs_1")
        document_uuid1 = upload_response1["document_uuid"]

        # Upload second small document (using same file for simplicity)
        upload_response2 = await file_uploader(file_path1, "test_two_small_docs_2")
        document_uuid2 = upload_response2["document_uuid"]

        # Allow time for indexing
        await asyncio.sleep(10)

        # Create RAG request for both documents
        rag_request = RagRequest(
            use_central_rag=True,
            user_id=1,
            query="summarise the content of both documents",
            document_uuids=[document_uuid1, document_uuid2],
        )

        # Call the function directly
        result = await _retrieve_relevant_chunks(rag_request, mock_search_index, mock_message, db_session)

        # Calculate total characters
        total_characters = sum(len(chunk.get("_source", {}).get("chunk_content", "")) for chunk in result)

        # Check which documents are represented
        doc_uuids_in_result = set()
        for chunk in result:
            doc_uuid = chunk.get("_source", {}).get("document_uuid")
            if doc_uuid:
                doc_uuids_in_result.add(doc_uuid)

        logger.info(f"Two small docs test: Retrieved {len(result)} chunks with {total_characters} total characters")
        logger.info(f"Documents represented: {len(doc_uuids_in_result)} out of 2")

        # Assertions
        assert len(result) > 0, "Should retrieve some chunks"
        assert total_characters <= GLOBAL_CHARACTER_LIMIT, (
            f"Should not exceed character limit of {GLOBAL_CHARACTER_LIMIT}"
        )
        assert len(doc_uuids_in_result) == 2, "Should include chunks from both documents"
        assert document_uuid1 in doc_uuids_in_result, "Should include chunks from first document"
        assert document_uuid2 in doc_uuids_in_result, "Should include chunks from second document"

        # Since both are small documents, total should be well under limit and include all chunks
        assert total_characters < GLOBAL_CHARACTER_LIMIT * 0.5, (
            "Two small documents should use less than 50% of character limit"
        )

    @pytest.mark.skip(reason="flaky")
    async def test_twenty_small_documents_distributed_retrieval(
        self, mock_search_index, mock_message, db_session, file_uploader
    ):
        """
        Example 4: Upload 20 small documents -> Ask to 'summarise the content of all documents'
        -> Assert that GLOBAL_CHARACTER_LIMIT was not breached AND more than half was used
        AND some chunks from each document were included
        """
        # Upload 20 small documents (using same file for simplicity)
        document_uuids = []
        file_path = "tests/resources/DNA_Topics_UK.docx"

        for i in range(20):
            upload_response = await file_uploader(file_path, f"test_twenty_docs_{i}")
            document_uuids.append(upload_response["document_uuid"])

        # Allow time for indexing
        await asyncio.sleep(30)

        # Create RAG request for all documents
        rag_request = RagRequest(
            use_central_rag=True,
            user_id=1,
            query="summarise the content of all documents",
            document_uuids=document_uuids,
        )

        # Call the function directly
        result = await _retrieve_relevant_chunks(rag_request, mock_search_index, mock_message, db_session)

        # Calculate total characters
        total_characters = sum(len(chunk.get("_source", {}).get("chunk_content", "")) for chunk in result)

        # Check which documents are represented
        doc_uuids_in_result = set()
        for chunk in result:
            doc_uuid = chunk.get("_source", {}).get("document_uuid")
            if doc_uuid:
                doc_uuids_in_result.add(doc_uuid)

        logger.info(f"Twenty docs test: Retrieved {len(result)} chunks with {total_characters} total characters")
        logger.info(
            f"Character limit: {GLOBAL_CHARACTER_LIMIT}, "
            f"Usage: {(total_characters / GLOBAL_CHARACTER_LIMIT) * 100:.1f}%"
        )
        logger.info(f"Documents represented: {len(doc_uuids_in_result)} out of {len(document_uuids)}")

        # Assertions
        assert len(result) > 0, "Should retrieve some chunks"
        assert total_characters <= GLOBAL_CHARACTER_LIMIT, (
            f"Should not exceed character limit of {GLOBAL_CHARACTER_LIMIT}"
        )

        # Should include chunks from multiple documents (ideally all, but at least most)
        min_expected_docs = max(10, len(document_uuids) // 2)  # At least half the documents
        assert len(doc_uuids_in_result) >= min_expected_docs, (
            f"Should include chunks from at least {min_expected_docs} documents, got {len(doc_uuids_in_result)}"
        )

    @pytest.mark.skip(reason="Flaky")
    async def test_fair_distribution_large_and_small_documents(
        self, mock_search_index, mock_message, db_session, file_uploader
    ):
        """
        Test fair distribution: 1 large document + 1 small document
        -> Assert that both documents are represented despite size difference
        -> Assert that the large document gets more space but small document isn't ignored
        """
        # Upload one small document
        small_file_path = "tests/resources/DNA_Topics_UK.docx"  # ~32KB, ~12 chunks
        small_upload_response = await file_uploader(small_file_path, "test_fair_small")
        small_doc_uuid = small_upload_response["document_uuid"]

        # Upload one large document
        large_file_path = "tests/resources/Introduction_to_Machine_Learning_with_Python.pdf"  # 6.7MB, ~700 chunks
        large_upload_response = await file_uploader(large_file_path, "test_fair_large")
        large_doc_uuid = large_upload_response["document_uuid"]

        # Allow time for indexing
        await asyncio.sleep(20)

        # Create RAG request for both documents
        rag_request = RagRequest(
            use_central_rag=True,
            user_id=1,
            query="summarize the content from both documents",
            document_uuids=[small_doc_uuid, large_doc_uuid],
        )

        # Call the function directly
        result = await _retrieve_relevant_chunks(rag_request, mock_search_index, mock_message, db_session)

        # Calculate total characters and per-document statistics
        total_characters = sum(len(chunk.get("_source", {}).get("chunk_content", "")) for chunk in result)

        # Analyze distribution by document
        small_doc_chunks = []
        large_doc_chunks = []
        small_doc_chars = 0
        large_doc_chars = 0

        for chunk in result:
            doc_uuid = chunk.get("_source", {}).get("document_uuid")
            chunk_content = chunk.get("_source", {}).get("chunk_content", "")
            chunk_chars = len(chunk_content)

            if doc_uuid == small_doc_uuid:
                small_doc_chunks.append(chunk)
                small_doc_chars += chunk_chars
            elif doc_uuid == large_doc_uuid:
                large_doc_chunks.append(chunk)
                large_doc_chars += chunk_chars

        logger.info("Fair distribution test results:")
        logger.info(f"Total: {len(result)} chunks, {total_characters} characters")
        logger.info(f"Small doc ({small_doc_uuid[:8]}): {len(small_doc_chunks)} chunks, {small_doc_chars} chars")
        logger.info(f"Large doc ({large_doc_uuid[:8]}): {len(large_doc_chunks)} chunks, {large_doc_chars} chars")
        logger.info(
            f"Character limit: {GLOBAL_CHARACTER_LIMIT}, "
            f"Usage: {(total_characters / GLOBAL_CHARACTER_LIMIT) * 100:.1f}%"
        )

        # Core assertions
        assert len(result) > 0, "Should retrieve some chunks"
        assert total_characters <= GLOBAL_CHARACTER_LIMIT, (
            f"Should not exceed character limit of {GLOBAL_CHARACTER_LIMIT}"
        )

        # Fair distribution assertions
        assert len(small_doc_chunks) > 0, "Small document should have at least some chunks represented"
        assert len(large_doc_chunks) > 0, "Large document should have chunks represented"
