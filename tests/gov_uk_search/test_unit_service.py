from unittest.mock import patch

import pytest

from app.gov_uk_search.schemas import DocumentBlacklistStatus, NonRagDocument
from app.gov_uk_search.service import (
    process_documents,
)

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# process_documents
# ---------------------------------------------------------------------------


class TestProcessDocuments:
    @pytest.mark.asyncio
    async def test_truncates_oversized_document_body(self):
        oversized_body = "x" * 50000
        document = NonRagDocument(
            url="https://www.gov.uk/some-guide",
            title="Some guide",
            body=oversized_body,
            status=DocumentBlacklistStatus.OK,
        )

        with patch("app.gov_uk_search.service.GOV_UK_SEARCH_MAX_DOCUMENT_CHARS", 100):
            wrapped = await process_documents([document])

        assert "x" * 100 in wrapped
        assert "x" * 101 not in wrapped
        assert "[49,900 characters truncated. Call get_gov_uk_page to retrieve the full page.]" in wrapped

    @pytest.mark.asyncio
    async def test_leaves_short_document_body_untouched(self):
        document = NonRagDocument(
            url="https://www.gov.uk/some-guide",
            title="Some guide",
            body="short content",
            status=DocumentBlacklistStatus.OK,
        )

        wrapped = await process_documents([document])

        assert "<document-body>\nshort content\n</document-body>" in wrapped
        assert "truncated" not in wrapped


# ---------------------------------------------------------------------------
# assess_if_next_message_should_use_gov_uk_search
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# get_search_queries
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# assess_document_relevancy
# ---------------------------------------------------------------------------
