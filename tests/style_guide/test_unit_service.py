"""
Unit tests for style guide service functions.
Tests document text retrieval and formatting findings for the chat model.
"""

import logging
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from app.style_guide.routes import StyleGuideCheckRequest
from app.style_guide.routes import check_style_guide as check_style_guide_route
from app.style_guide.service import (
    MAX_EXAMPLES_PER_ISSUE,
    RULE_GUIDANCE,
    _format_issue,
    _get_document_text,
    describe_violation,
)
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


class TestFormatIssue:
    """Tests for _format_issue, which turns one violation into data for the chat model."""

    def test_pattern_violation(self):
        violation = {
            "rule_id": "rule_002",
            "rule_title": "A level",
            "rule_details": "No hyphen. Lower case level.",
            "rule_broken": "A-level",
            "match_count": 2,
            "sentences": ["She has three A-levels.", "A-level results are out."],
        }

        result = _format_issue(violation)

        assert "<rule>A level</rule>" in result
        assert "<guidance>No hyphen. Lower case level.</guidance>" in result
        assert "<detected-by>pattern match</detected-by>" in result
        assert "<problem>A-level</problem>" in result
        assert "<count>2</count>" in result
        assert "<found-in>She has three A-levels.</found-in>" in result
        assert "<found-in>A-level results are out.</found-in>" in result

    def test_case_sensitive_violation_includes_correct_form(self):
        violation = {
            "rule_id": "rule_100",
            "rule_title": "Brexit",
            "rule_broken": "Incorrect casing (expected: 'Brexit')",
            "correct_string": "Brexit",
            "match_count": 1,
            "sentences": ["After brexit, things changed."],
        }

        assert "<correct-form>Brexit</correct-form>" in _format_issue(violation)

    def test_llm_violation_takes_guidance_from_rules(self):
        """LLM violations don't carry the rule details, so the guidance is looked up by rule id."""
        violation = {
            "rule_id": "rule_002",
            "rule_title": "A level",
            "validation_type": "llm",
            "violation_reason": "Hyphenated",
            "occurrences": ["A-level"],
        }

        result = _format_issue(violation)

        assert "<detected-by>model review</detected-by>" in result
        assert "<reason>Hyphenated</reason>" in result
        assert "<guidance>No hyphen. Lower case level.</guidance>" in result
        assert "<found-in>A-level</found-in>" in result

    def test_examples_are_capped(self):
        sentences = [f"Sentence {i}." for i in range(MAX_EXAMPLES_PER_ISSUE + 3)]
        violation = {"rule_id": "rule_x", "rule_title": "X", "sentences": sentences, "match_count": len(sentences)}

        result = _format_issue(violation)

        assert result.count("<found-in>") == MAX_EXAMPLES_PER_ISSUE
        assert "<more>3 more not listed</more>" in result
        assert f"<count>{len(sentences)}</count>" in result


class TestCheckStyleGuideArgs:
    """The tool takes exactly one of text or document_id."""

    def test_text_only(self):
        assert CheckStyleGuideArgs(text="Draft.").text == "Draft."

    def test_document_only_with_context(self):
        args = CheckStyleGuideArgs(document_id="u1", context="A page for teachers")
        assert (args.document_id, args.context) == ("u1", "A page for teachers")

    @pytest.mark.parametrize("kwargs", [{}, {"text": "Draft.", "document_id": "u1"}, {"context": "only context"}])
    def test_needs_exactly_one_source(self, kwargs):
        with pytest.raises(ValidationError, match="exactly one"):
            CheckStyleGuideArgs(**kwargs)


class TestDescribeViolation:
    """describe_violation gives pattern and model violations one shape."""

    def test_pattern_violation(self):
        violation = {
            "rule_id": "rule_100",
            "rule_title": "Brexit",
            "rule_broken": "Incorrect casing (expected: 'Brexit')",
            "correct_string": "Brexit",
            "match_count": 2,
            "sentences": ["After brexit, things changed."],
        }

        assert describe_violation(violation) == {
            "rule_id": "rule_100",
            "rule_title": "Brexit",
            "guidance": RULE_GUIDANCE.get("rule_100"),
            "detected_by": "pattern match",
            "problem": "Incorrect casing (expected: 'Brexit')",
            "correct_form": "Brexit",
            "reason": None,
            "count": 2,
            "examples": ["After brexit, things changed."],
        }

    def test_model_violation(self):
        violation = {
            "rule_id": "rule_002",
            "rule_title": "A level",
            "validation_type": "llm",
            "violation_reason": "Hyphenated",
            "occurrences": ["A-level", "A-levels"],
        }

        described = describe_violation(violation)

        assert described["detected_by"] == "model review"
        assert described["reason"] == "Hyphenated"
        assert described["guidance"] == "No hyphen. Lower case level."
        assert (described["count"], described["examples"]) == (2, ["A-level", "A-levels"])
        assert described["problem"] is None


class TestStyleGuideCheckRoute:
    """The /v1/style-guide/check route returns the violations the checker finds."""

    @pytest.mark.asyncio
    async def test_returns_violation_details(self):
        request = StyleGuideCheckRequest(content="She has three A-levels and an A star.", skip_llm=True)

        response = await check_style_guide_route(request)

        assert response.violation_count == len(response.violations) > 0
        a_star = next(v for v in response.violations if v.problem == "a star")
        assert a_star.detected_by == "pattern match"
        assert a_star.count == 1
        assert a_star.examples == ["She has three A-levels and an A star."]
        assert a_star.guidance
