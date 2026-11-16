"""Integration tests for the check_style_guide tool's service entry point, with the engine mocked."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

from app.bedrock.call_log import LlmCallContext
from app.config import STYLE_GUIDE_MAX_DOCUMENT_CHARS, StyleGuideThoroughness
from app.style_guide.engine import CheckResult, Finding, Source
from app.style_guide.rules import RULES, rules_for
from app.style_guide.service import check_style_guide_for_tool

pytestmark = [
    pytest.mark.integration,
]

CONTEXT = LlmCallContext(purpose="style_guide_validation", chat_id=7, message_id=9)


async def check_style_guide(
    text: str | None = None,
    document_id: str | None = None,
    checker_context: str | None = None,
    document_uuids: list | None = None,
    thoroughness: StyleGuideThoroughness | None = None,
) -> tuple[str, bool]:
    """Call the tool's service entry point with sensible defaults."""
    return await check_style_guide_for_tool(
        text=text,
        document_id=document_id,
        checker_context=checker_context,
        document_uuids=document_uuids or [],
        user_id=1,
        context=CONTEXT,
        thoroughness=thoroughness,
    )


class TestAnalyseContent:
    """What the service does around the engine."""

    @pytest.mark.asyncio
    @patch("app.style_guide.service.run_check", new_callable=AsyncMock)
    async def test_contexts_reach_the_engine(self, mock_run):
        mock_run.return_value = CheckResult()

        await check_style_guide("Some text.", checker_context="A page for teachers")

        mock_run.assert_awaited_once_with(
            "Some text.", "A page for teachers", CONTEXT, rules=rules_for(StyleGuideThoroughness.balanced)
        )

    @pytest.mark.asyncio
    @patch("app.style_guide.service.run_check", new_callable=AsyncMock)
    async def test_default_thoroughness_skips_the_lowest_priority_rules(self, mock_run):
        mock_run.return_value = CheckResult()

        content, _ = await check_style_guide("Some text.")

        skipped = len(RULES) - len(rules_for(StyleGuideThoroughness.balanced))
        assert skipped > 0
        assert f"<thoroughness>balanced: {skipped} lower-priority rules" in content

    @pytest.mark.asyncio
    @patch("app.style_guide.service.run_check", new_callable=AsyncMock)
    async def test_thorough_runs_every_rule(self, mock_run):
        mock_run.return_value = CheckResult()

        content, _ = await check_style_guide("Some text.", thoroughness=StyleGuideThoroughness.thorough)

        assert mock_run.await_args.kwargs["rules"] == RULES
        assert "<thoroughness>thorough: all rules were run.</thoroughness>" in content

    @pytest.mark.asyncio
    @patch("app.style_guide.service.run_check", new_callable=AsyncMock)
    async def test_findings_are_formatted(self, mock_run):
        mock_run.return_value = CheckResult(findings=[Finding("rule_099", 4, 9, "can't", Source.PATTERN)])

        content, failed = await check_style_guide("You can't do that.")

        assert not failed
        assert content.startswith("<style-guide-check>")
        assert "<checked>the text passed in</checked>" in content
        assert "<rule>contractions</rule>" in content

    @pytest.mark.asyncio
    @patch("app.style_guide.service.run_check", new_callable=AsyncMock)
    async def test_oversized_text_returns_error(self, mock_run):
        content, failed = await check_style_guide("x" * (STYLE_GUIDE_MAX_DOCUMENT_CHARS + 1))

        mock_run.assert_not_called()
        assert failed
        assert f"{STYLE_GUIDE_MAX_DOCUMENT_CHARS:,}" in content

    @pytest.mark.asyncio
    @patch("app.style_guide.service.run_check", new_callable=AsyncMock)
    async def test_whitespace_only_text_returns_error(self, mock_run):
        content, failed = await check_style_guide("   \n\t  ")

        mock_run.assert_not_called()
        assert failed
        assert "<error>There was no text to check.</error>" in content

    @pytest.mark.asyncio
    @patch("app.style_guide.service.run_check", new_callable=AsyncMock)
    async def test_engine_failure_returns_error(self, mock_run):
        mock_run.side_effect = RuntimeError("model missing")

        content, failed = await check_style_guide("Some text.")

        assert failed
        assert "could not be completed" in content


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
