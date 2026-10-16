"""
Integration tests for style guide service.
Tests the check_style_guide_for_tool / analysis flow with mocked components.
"""

import logging
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from app.bedrock.call_log import LlmCallContext
from app.style_guide.service import (
    _run_llm_validation_on_chunks,
    analyse_content_for_style_guide,
    check_style_guide_for_tool,
)

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.integration,
]

CONTEXT = LlmCallContext(purpose="style_guide_validation", chat_id=7, message_id=9)


async def check_style_guide(
    text: str | None = None,
    document_id: str | None = None,
    checker_context: str | None = None,
    document_uuids: list | None = None,
) -> tuple[str, bool]:
    """Call the tool's service entry point with sensible defaults."""
    return await check_style_guide_for_tool(
        text=text,
        document_id=document_id,
        checker_context=checker_context,
        document_uuids=document_uuids or [],
        user_id=1,
        context=CONTEXT,
    )


class TestStyleGuideAnalysis:
    """The findings returned to the chat model."""

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_case_sensitive_rules", return_value=[])
    @patch("app.style_guide.service.check_case_insensitive_rules", return_value=[])
    @patch("app.style_guide.service.check_llm_validation_rules")
    async def test_no_violations_says_so(self, mock_llm_rules, *_):
        mock_llm_rules.return_value = []

        content, failed = await check_style_guide("This is clean text.")

        assert not failed
        assert content.startswith("<style-guide-check>")
        assert "<checked>the text passed in</checked>" in content
        assert "No issues found" in content
        assert "<issue>" not in content

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_case_sensitive_rules", return_value=[])
    @patch("app.style_guide.service.check_llm_validation_rules")
    async def test_lists_issues_without_instructions_or_rewrite(self, mock_llm_rules, _):
        """Findings come back as data: no formatting orders for the chat model and no corrected version."""
        mock_llm_rules.return_value = [
            {
                "rule_id": "rule_002",
                "rule_title": "A level",
                "validation_type": "llm",
                "violation_reason": "Hyphenated",
                "occurrences": ["A-levels"],
            }
        ]

        content, failed = await check_style_guide("She has three A-levels and an A star.")

        assert not failed
        assert content.count("<issue>") >= 2  # the LLM finding plus the deterministic "a star" rule
        assert "<reason>Hyphenated</reason>" in content
        assert "MUST" not in content
        assert "corrected" not in content.lower()

    @pytest.mark.asyncio
    @patch("app.style_guide.service.MAX_ISSUES", 3)
    @patch("app.style_guide.service.check_case_sensitive_rules", return_value=[])
    @patch("app.style_guide.service.check_llm_validation_rules", return_value=[])
    @patch("app.style_guide.service.check_case_insensitive_rules")
    async def test_caps_issues_keeping_the_most_frequent(self, mock_insensitive, *_):
        mock_insensitive.return_value = [
            {"rule_id": f"r{i}", "rule_title": f"Rule {i}", "match_count": i, "sentences": [f"S{i}."]}
            for i in range(1, 6)
        ]

        content, failed = await check_style_guide("Some text.")

        assert not failed
        assert "<result>5 issues found.</result>" in content
        assert content.count("<issue>") == 3
        assert "<rule>Rule 5</rule>" in content and "<rule>Rule 3</rule>" in content
        assert "<rule>Rule 2</rule>" not in content
        assert "The other 2 were left out" in content

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_case_sensitive_rules", return_value=[])
    @patch("app.style_guide.service.check_llm_validation_rules", return_value=[])
    async def test_guidance_given_once_per_rule(self, *_):
        """Words to avoid gives an issue per word matched; its long guidance should appear only once."""
        content, failed = await check_style_guide("We will deliver and leverage this.")

        assert not failed
        words_to_avoid = content.count("<rule>Words to avoid</rule>")
        assert words_to_avoid >= 2
        assert content.count("<guidance>") == content.count("<issue>") - words_to_avoid + 1

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_case_insensitive_rules")
    async def test_errors_are_raised_for_the_loop_to_handle(self, mock_check_rules):
        """Errors propagate; the agent loop turns them into a generic is_error tool_result."""
        mock_check_rules.side_effect = Exception("Test error")

        with pytest.raises(Exception, match="Test error"):
            await check_style_guide("Test content")

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_llm_validation_rules")
    async def test_llm_validation_failure_is_raised(self, mock_llm_rules):
        """A failed validation call fails the tool, rather than reporting a partial check as complete."""
        mock_llm_rules.side_effect = RuntimeError("Bedrock unavailable")

        with pytest.raises(RuntimeError, match="Bedrock unavailable"):
            await check_style_guide("Some text.")


class TestCheckerContextAndLogging:
    """The validation calls get the assistant's context and the chat's logging context."""

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_llm_validation_rules")
    async def test_call_context_is_passed_to_validation(self, mock_llm_rules):
        mock_llm_rules.return_value = []

        await check_style_guide("Some text.")

        assert mock_llm_rules.call_args.kwargs["context"] is CONTEXT

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_llm_validation_rules")
    async def test_assistant_context_reaches_the_prompts(self, mock_llm_rules):
        mock_llm_rules.return_value = []

        await check_style_guide("TEXT BEING CHECKED", checker_context="A page for teachers")

        checker_context = mock_llm_rules.call_args.kwargs["checker_context"]
        assert "A page for teachers" in checker_context
        assert "TEXT BEING CHECKED" not in checker_context

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_llm_validation_rules")
    async def test_no_assistant_context_means_no_context_block(self, mock_llm_rules):
        mock_llm_rules.return_value = []

        await check_style_guide("Some text.")

        assert mock_llm_rules.call_args.kwargs["checker_context"] == ""


# ---------------------------------------------------------------------------
# Document size handling
# ---------------------------------------------------------------------------


class TestStyleGuideDocumentSizeHandling:
    """Tests for the max-document-size guard and chunked processing paths."""

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_llm_validation_rules")
    async def test_oversized_document_returns_error(self, mock_llm_rules):
        """A document over STYLE_GUIDE_MAX_DOCUMENT_CHARS returns an error naming the limit, not a check."""
        from app.config import STYLE_GUIDE_MAX_DOCUMENT_CHARS

        oversized = "x" * (STYLE_GUIDE_MAX_DOCUMENT_CHARS + 500)

        content, failed = await check_style_guide(oversized)

        mock_llm_rules.assert_not_called()
        assert failed
        assert "<error>" in content
        assert f"{STYLE_GUIDE_MAX_DOCUMENT_CHARS:,}" in content

    @pytest.mark.asyncio
    @patch("app.style_guide.service.STYLE_GUIDE_MAX_DOCUMENT_CHARS", 500)
    @patch("app.style_guide.service.STYLE_GUIDE_MAX_CHUNK_CHARS", 200)
    @patch("app.style_guide.service._run_llm_validation_on_chunks")
    @patch("app.style_guide.service.check_llm_validation_rules")
    async def test_large_document_uses_chunked_llm_path(self, mock_llm_rules, mock_chunk_llm):
        """Documents over MAX_CHUNK_CHARS (but under MAX_DOCUMENT_CHARS) use chunked LLM path."""
        mock_chunk_llm.return_value = []

        # ~300 chars: above the patched MAX_CHUNK_CHARS (200) but below MAX_DOCUMENT_CHARS (500)
        content, failed = await analyse_content_for_style_guide(
            "This is a sentence here. " * 12, "the text", checker_context="CONV", context=CONTEXT
        )

        mock_chunk_llm.assert_called_once()
        assert mock_chunk_llm.call_args.args[2:] == ("CONV", CONTEXT)
        mock_llm_rules.assert_not_called()
        assert not failed

    @pytest.mark.asyncio
    @patch("app.style_guide.service.STYLE_GUIDE_MAX_DOCUMENT_CHARS", 5000)
    @patch("app.style_guide.service.STYLE_GUIDE_MAX_CHUNK_CHARS", 1000)
    @patch("app.style_guide.service.check_llm_validation_rules")
    async def test_small_document_uses_normal_path(self, mock_llm_rules):
        """Documents under MAX_CHUNK_CHARS use the normal non-chunked LLM path."""
        mock_llm_rules.return_value = []

        await analyse_content_for_style_guide("Short text.", "the text")

        mock_llm_rules.assert_called_once()

    @pytest.mark.asyncio
    async def test_whitespace_only_content_returns_error(self):
        content, failed = await check_style_guide("   \n\t  ")

        assert failed
        assert "<error>There was no text to check.</error>" in content


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


class TestRunLlmValidationOnChunks:
    """Tests for the _run_llm_validation_on_chunks helper."""

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_llm_validation_rules")
    async def test_calls_llm_once_per_chunk_with_context(self, mock_llm_rules):
        mock_llm_rules.return_value = []
        chunks = ["chunk one.", "chunk two.", "chunk three."]
        await _run_llm_validation_on_chunks(chunks, llm_model="test-model", checker_context="CONV", context=CONTEXT)
        assert mock_llm_rules.call_count == 3
        assert all(call.kwargs["checker_context"] == "CONV" for call in mock_llm_rules.call_args_list)
        assert all(call.kwargs["context"] is CONTEXT for call in mock_llm_rules.call_args_list)

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_llm_validation_rules")
    async def test_merges_same_rule_across_chunks(self, mock_llm_rules):
        """A rule triggered in multiple chunks appears once, with the occurrences from every chunk."""
        mock_llm_rules.side_effect = [
            [{"rule_id": "rule_001", "rule_title": "Test Rule", "occurrences": ["first"]}],
            [{"rule_id": "rule_001", "rule_title": "Test Rule", "occurrences": ["first", "second"]}],
        ]
        result = await _run_llm_validation_on_chunks(["chunk one.", "chunk two."], llm_model="test-model")
        assert len(result) == 1
        assert result[0]["occurrences"] == ["first", "second"]

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_llm_validation_rules")
    async def test_combines_different_rules_from_chunks(self, mock_llm_rules):
        """Different rules found in different chunks all appear in the output."""
        mock_llm_rules.side_effect = [
            [{"rule_id": "rule_001", "rule_title": "Rule 1"}],
            [{"rule_id": "rule_002", "rule_title": "Rule 2"}],
        ]
        result = await _run_llm_validation_on_chunks(["chunk one.", "chunk two."], llm_model="test-model")
        assert {v["rule_id"] for v in result} == {"rule_001", "rule_002"}

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_llm_validation_rules")
    async def test_empty_chunks_returns_empty_list(self, mock_llm_rules):
        result = await _run_llm_validation_on_chunks([], llm_model="test-model")
        assert result == []
        mock_llm_rules.assert_not_called()


class TestCheckStyleGuideForToolSources:
    """What gets checked: the text given, or one attached document."""

    @pytest.mark.asyncio
    @patch("app.style_guide.service.analyse_content_for_style_guide")
    async def test_text_is_checked(self, mock_analyse):
        mock_analyse.return_value = ("<style-guide-check/>", False)

        await check_style_guide("Draft text.")

        assert mock_analyse.call_args.args[:2] == ("Draft text.", "the text passed in")

    @pytest.mark.asyncio
    @patch("app.style_guide.service.analyse_content_for_style_guide")
    @patch("app.style_guide.service._get_document_text")
    async def test_document_is_checked(self, mock_get_text, mock_analyse):
        mock_get_text.return_value = ("a.docx", "Body text")
        mock_analyse.return_value = ("<style-guide-check/>", False)
        read_session = AsyncMock()

        @asynccontextmanager
        async def short_session():
            yield read_session

        with patch("app.style_guide.service.async_db_session", short_session):
            await check_style_guide(document_id="U1", document_uuids=["u1", "u2"])

        mock_get_text.assert_awaited_once_with("u1", 1, read_session)
        assert mock_analyse.call_args.args[:2] == ("Body text", "the attached document a.docx")

    @pytest.mark.asyncio
    @patch("app.style_guide.service._get_document_text")
    async def test_document_read_session_is_closed_before_the_validation_calls(self, mock_get_text):
        events = []
        mock_get_text.return_value = ("a.docx", "Body text")

        @asynccontextmanager
        async def short_session():
            events.append("read_open")
            yield AsyncMock()
            events.append("read_closed")

        async def analyse(*args, **kwargs):
            events.append("validation")
            return "<style-guide-check/>", False

        with (
            patch("app.style_guide.service.async_db_session", short_session),
            patch("app.style_guide.service.analyse_content_for_style_guide", side_effect=analyse),
        ):
            await check_style_guide(document_id="u1", document_uuids=["u1"])

        assert events == ["read_open", "read_closed", "validation"]

    @pytest.mark.asyncio
    @patch("app.style_guide.service._get_document_text")
    async def test_unattached_document_returns_error(self, mock_get_text):
        content, failed = await check_style_guide(document_id="other", document_uuids=["u1"])

        mock_get_text.assert_not_called()
        assert failed
        assert "Unknown document id: other" in content
        assert "u1" in content

    @pytest.mark.asyncio
    @patch("app.style_guide.service._get_document_text")
    async def test_unreadable_document_returns_error(self, mock_get_text):
        mock_get_text.return_value = None

        content, failed = await check_style_guide(document_id="u1", document_uuids=["u1"])

        assert failed
        assert "could not be read" in content

    @pytest.mark.asyncio
    @patch("app.style_guide.service.analyse_content_for_style_guide")
    @patch("app.style_guide.service._get_document_text")
    async def test_document_without_text_returns_error(self, mock_get_text, mock_analyse):
        mock_get_text.return_value = ("a.docx", "")

        content, failed = await check_style_guide(document_id="u1", document_uuids=["u1"])

        mock_analyse.assert_not_called()
        assert failed
        assert "no readable text" in content
