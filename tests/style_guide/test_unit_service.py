"""Unit tests for reading an attached document and the tool's arguments."""

import logging
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from app.style_guide.service import _get_document_text
from app.style_guide.tools import CheckStyleGuideArgs

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.unit,
]


class TestDocumentText:
    """Tests for _get_document_text."""

    @pytest.mark.asyncio
    async def test_retrieves_document_text(self):
        mock_document = Mock()
        mock_document.id = 1
        mock_document.name = "test_doc.pdf"
        mock_chunk1 = Mock()
        mock_chunk1.content = "First chunk content"
        mock_chunk2 = Mock()
        mock_chunk2.content = "Second chunk content"

        db_session = AsyncMock()
        db_session.scalar = AsyncMock(return_value=mock_document)
        chunks_result = Mock()
        chunks_result.all.return_value = [mock_chunk1, mock_chunk2]
        db_session.scalars = AsyncMock(return_value=chunks_result)

        name, text = await _get_document_text("test-uuid-123", user_id=1, db_session=db_session)

        assert name == "test_doc.pdf"
        assert text == "First chunk content\nSecond chunk content\n"

    @pytest.mark.asyncio
    async def test_user_access_denied(self):
        """A document the user can't access returns None."""
        db_session = AsyncMock()
        db_session.scalar = AsyncMock(return_value=None)

        result = await _get_document_text("test-uuid-123", user_id=1, db_session=db_session)

        assert result is None
        db_session.scalars.assert_not_called()

    @pytest.mark.asyncio
    async def test_database_errors_propagate(self):
        """A database error is raised for the agent loop to report."""
        db_session = AsyncMock()
        db_session.scalar = AsyncMock(side_effect=RuntimeError("db down"))

        with pytest.raises(RuntimeError, match="db down"):
            await _get_document_text("u1", user_id=1, db_session=db_session)


class TestCheckStyleGuideArgs:
    """The tool takes exactly one of text or document_id."""

    def test_text_only(self):
        assert CheckStyleGuideArgs(text="Draft.").text == "Draft."

    def test_document_only_with_context(self):
        args = CheckStyleGuideArgs(document_id="u1", context="A page for teachers")
        assert (args.document_id, args.context) == ("u1", "A page for teachers")

    def test_thoroughness(self):
        assert CheckStyleGuideArgs(text="Draft.", thoroughness="lite").thoroughness == "lite"
        assert CheckStyleGuideArgs(text="Draft.").thoroughness is None
        with pytest.raises(ValidationError):
            CheckStyleGuideArgs(text="Draft.", thoroughness="exhaustive")

    @pytest.mark.parametrize("kwargs", [{}, {"text": "Draft.", "document_id": "u1"}, {"context": "only context"}])
    def test_needs_exactly_one_source(self, kwargs):
        with pytest.raises(ValidationError, match="exactly one"):
            CheckStyleGuideArgs(**kwargs)
