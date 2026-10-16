import asyncio

import pytest

from app.document_upload.personal_document_rag import (
    UPLOADED_DOCUMENTS_MAX_RESULT_CHARS,
    select_chunks_for_tool,
)


class TestPersonalDocumentRAGIntegration:
    """Integration tests for personal document search with real document uploads."""

    @pytest.mark.asyncio
    async def test_large_document_search_respects_character_limit(self, db_session, file_uploader):
        """A search over a large document returns chunks up to, and never over, the character limit."""
        upload_response = await file_uploader(
            "tests/resources/Introduction_to_Machine_Learning_with_Python.pdf", "test_large_doc_search"
        )
        document_uuid = upload_response["document_uuid"]
        await asyncio.sleep(15)

        result, _ = await select_chunks_for_tool(
            ["machine learning", "model training data"], [document_uuid], exclude_ids=set()
        )

        total_characters = sum(chunk.char_count for chunk in result)
        assert len(result) > 0
        assert total_characters <= UPLOADED_DOCUMENTS_MAX_RESULT_CHARS
        assert total_characters > UPLOADED_DOCUMENTS_MAX_RESULT_CHARS * 0.5
        assert [c.score for c in result] == sorted((c.score for c in result), reverse=True)

    @pytest.mark.asyncio
    async def test_search_only_returns_chunks_from_the_documents_asked_for(self, db_session, file_uploader):
        """Chunks of a document that isn't in the search scope are never returned."""
        first = (await file_uploader("tests/resources/DNA_Topics_UK.docx", "test_scope_1"))["document_uuid"]
        second = (await file_uploader("tests/resources/DNA_Topics_UK.docx", "test_scope_2"))["document_uuid"]
        await asyncio.sleep(10)

        both, _ = await select_chunks_for_tool(["DNA"], [first, second], exclude_ids=set())
        only_first, _ = await select_chunks_for_tool(["DNA"], [first], exclude_ids=set())

        assert len(both) > 0
        assert 0 < len(only_first) < len(both)
