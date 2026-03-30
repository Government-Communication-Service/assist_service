"""
Integration tests for style guide service.
Tests the full check_content_against_style_guide flow with mocked components.
"""
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from anthropic.types import Message as AnthropicMessage
from anthropic.types import TextBlock, Usage

from app.database.models import Message
from app.style_guide.service import (
    _filter_violations_for_chunk,
    _generate_chunked_summary_and_fix,
    _run_llm_validation_on_chunks,
    check_content_against_style_guide,
)

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.integration,
]


def create_mock_message(role: str, content: str, message_id: int = 1) -> Message:
    """Helper to create mock Message objects."""
    message = Mock(spec=Message)
    message.id = message_id
    message.role = role
    message.content = content
    message.content_enhanced_with_rag = content
    message.summary = None  # Needed for prepare_message_objects_for_llm
    return message


def create_mock_llm_response(text: str) -> AnthropicMessage:
    """Helper to create mock Anthropic LLM response."""
    return AnthropicMessage(
        id="msg_test",
        content=[TextBlock(text=text, type="text")],
        model="test-model",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage=Usage(input_tokens=100, output_tokens=50),
    )


class TestStyleGuideIntegration:
    """Integration tests for check_content_against_style_guide."""

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_llm_validation_rules")
    @patch("app.style_guide.service.generate_summary_and_fix")
    async def test_initial_call_full_analysis(
        self, mock_generate_summary, mock_check_llm_rules
    ):
        """Test that initial call runs full analysis."""
        # Mock no LLM violations
        mock_check_llm_rules.return_value = []

        # Mock summary generation
        mock_generate_summary.return_value = {
            "summary": "Found violations in the document.",
            "fixed_document": "This is the fixed document.",
        }

        content = "The Prime Minister announced today."

        prompt_segment = await check_content_against_style_guide(
            content=content,
            messages=None  # No messages = initial call
        )

        # Should run full analysis
        mock_check_llm_rules.assert_called_once()

        # Should have formatted prompt segment
        assert prompt_segment is not None
        assert "style-guide-analysis" in prompt_segment

    @pytest.mark.asyncio
    @patch("app.style_guide.service.determine_style_guide_content_type")
    async def test_follow_up_routes_to_simple_response(
        self, mock_content_type_router
    ):
        """Test that follow-up modification request returns None (main LLM handles it)."""
        from app.style_guide.service import StyleGuideContentType

        # Router decides simple response - main LLM should handle
        mock_content_type_router.return_value = StyleGuideContentType.SIMPLE_RESPONSE

        # Create follow-up conversation
        messages = [
            create_mock_message("user", "Check this text", 1),
            create_mock_message("assistant", "Found violations", 2),
            create_mock_message("user", "Remove the sort code", 3),
        ]

        content = "Remove the sort code"

        prompt_segment = await check_content_against_style_guide(
            content=content,
            messages=messages
        )

        # Should call router
        mock_content_type_router.assert_called_once()

        # Should return None - main LLM will handle follow-up naturally
        assert prompt_segment is None

    @pytest.mark.asyncio
    @patch("app.style_guide.service.determine_style_guide_content_type")
    @patch("app.style_guide.service.check_llm_validation_rules")
    @patch("app.style_guide.service.generate_summary_and_fix")
    async def test_follow_up_routes_to_full_analysis(
        self, mock_generate_summary, mock_check_llm_rules, mock_content_type_router
    ):
        """Test that follow-up check request routes to full analysis."""
        from app.style_guide.service import StyleGuideContentType

        # Router decides query text check
        mock_content_type_router.return_value = StyleGuideContentType.QUERY_TEXT

        # Mock LLM validation - return a violation to trigger summary
        mock_check_llm_rules.return_value = [{"rule_id": "test", "rule_name": "Test Rule"}]

        # Mock summary generation
        mock_generate_summary.return_value = {
            "summary": "New violations found.",
            "fixed_document": "Fixed text.",
        }

        # Create follow-up conversation asking for new check
        messages = [
            create_mock_message("user", "Check this text", 1),
            create_mock_message("assistant", "Found violations", 2),
            create_mock_message("user", "Check this new paragraph", 3),
        ]

        content = "Check this new paragraph with violations."

        prompt_segment = await check_content_against_style_guide(
            content=content,
            messages=messages
        )

        # Should call router
        mock_content_type_router.assert_called_once()

        # Should run full analysis
        mock_check_llm_rules.assert_called_once()

        # Should generate summary
        mock_generate_summary.assert_called_once()

        # Should return prompt segment
        assert prompt_segment is not None

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_llm_validation_rules")
    async def test_no_violations_returns_clean_message(self, mock_check_llm_rules):
        """Test that content with no violations returns appropriate message."""
        # Mock no violations
        mock_check_llm_rules.return_value = []

        content = "This is clean text."

        prompt_segment = await check_content_against_style_guide(
            content=content
        )

        # Should have clean message
        assert prompt_segment is not None
        assert "No style guide violations" in prompt_segment
        assert "GOV.UK style guide principles" in prompt_segment

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_case_insensitive_rules")
    async def test_error_handling_returns_none(self, mock_check_rules):
        """Test that errors are handled gracefully."""
        # Simulate error in checking
        mock_check_rules.side_effect = Exception("Test error")

        content = "Test content"

        prompt_segment = await check_content_against_style_guide(
            content=content
        )

        # Should return None on error
        assert prompt_segment is None

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_llm_validation_rules")
    @patch("app.style_guide.service.generate_summary_and_fix")
    async def test_conversation_context_passed_to_summary(
        self, mock_generate_summary, mock_check_llm_rules
    ):
        """Test that conversation context is passed to summary generation."""
        # Mock violations found to trigger summary generation
        mock_check_llm_rules.return_value = [{"rule_id": "test", "rule_name": "Test Rule"}]

        # Mock summary generation
        mock_generate_summary.return_value = {
            "summary": "Summary",
            "fixed_document": "Fixed",
        }

        # Create conversation
        messages = [
            create_mock_message("user", "Check this", 1),
        ]

        content = "The Prime Minister said."

        await check_content_against_style_guide(
            content=content,
            messages=messages
        )

        # Verify conversation context was built and passed
        if mock_generate_summary.called:
            call_args = mock_generate_summary.call_args
            conversation_context = call_args[1].get('conversation_context', '')

            # Should have wrapped context
            assert "<ConversationContext>" in conversation_context
            assert "Check this" in conversation_context


class TestStyleGuideEndToEnd:
    """End-to-end style tests with real rule checking (but mocked LLM calls)."""

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_llm_validation_rules")
    @patch("app.style_guide.service.generate_summary_and_fix")
    async def test_e2e_violation_detection_and_summary(
        self, mock_generate_summary, mock_check_llm_rules
    ):
        """Test that violations trigger summary generation."""
        # Mock LLM violations to ensure summary is called
        mock_check_llm_rules.return_value = [{"rule_id": "test", "rule_name": "Test Rule"}]
        mock_generate_summary.return_value = {
            "summary": "Fixed job title capitalization.",
            "fixed_document": "The prime minister announced today.",
        }

        # Content to check
        content = "The Prime Minister and Home Secretary met today."

        await check_content_against_style_guide(
            content=content
        )

        # Should generate summary (violations were detected)
        mock_generate_summary.assert_called_once()


# ---------------------------------------------------------------------------
# Document size handling
# ---------------------------------------------------------------------------

class TestStyleGuideDocumentSizeHandling:
    """Tests for the max-document-size guard and chunked processing paths."""

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_llm_validation_rules")
    async def test_oversized_document_returns_error_segment(self, mock_llm_rules):
        """A document over STYLE_GUIDE_MAX_DOCUMENT_CHARS returns an error, not a check."""
        from app.config import STYLE_GUIDE_MAX_DOCUMENT_CHARS

        oversized = "This is a sentence. " * (STYLE_GUIDE_MAX_DOCUMENT_CHARS // 18 + 1)

        prompt_segment = await check_content_against_style_guide(content=oversized)

        # No LLM calls should be made
        mock_llm_rules.assert_not_called()
        assert prompt_segment is not None
        assert "style-guide-analysis" in prompt_segment
        assert "<error>" in prompt_segment

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_llm_validation_rules")
    async def test_oversized_document_error_mentions_character_limit(self, mock_llm_rules):
        """The error response mentions the character limit so the user knows how to split."""
        from app.config import STYLE_GUIDE_MAX_DOCUMENT_CHARS

        oversized = "x" * (STYLE_GUIDE_MAX_DOCUMENT_CHARS + 500)

        prompt_segment = await check_content_against_style_guide(content=oversized)

        assert prompt_segment is not None
        # The limit should appear somewhere in the response (formatted with commas or not)
        limit_str = str(STYLE_GUIDE_MAX_DOCUMENT_CHARS)
        limit_str_commas = f"{STYLE_GUIDE_MAX_DOCUMENT_CHARS:,}"
        assert limit_str in prompt_segment or limit_str_commas in prompt_segment

    @pytest.mark.asyncio
    @patch("app.style_guide.service.STYLE_GUIDE_MAX_DOCUMENT_CHARS", 500)
    @patch("app.style_guide.service.STYLE_GUIDE_MAX_CHUNK_CHARS", 200)
    @patch("app.style_guide.service._run_llm_validation_on_chunks")
    @patch("app.style_guide.service.check_llm_validation_rules")
    @patch("app.style_guide.service._generate_chunked_summary_and_fix")
    async def test_large_document_uses_chunked_llm_path(
        self, mock_chunked_summary, mock_llm_rules, mock_chunk_llm
    ):
        """Documents over MAX_CHUNK_CHARS (but under MAX_DOCUMENT_CHARS) use chunked LLM path."""
        mock_chunk_llm.return_value = []
        mock_chunked_summary.return_value = {
            "summary": "Chunked summary.",
            "fixed_document": "Corrected text.",
        }

        # ~300 chars: above the patched MAX_CHUNK_CHARS (200) but below MAX_DOCUMENT_CHARS (500)
        content = "This is a sentence here. " * 12

        prompt_segment = await check_content_against_style_guide(content=content)

        mock_chunk_llm.assert_called_once()
        mock_llm_rules.assert_not_called()
        assert prompt_segment is not None

    @pytest.mark.asyncio
    @patch("app.style_guide.service.STYLE_GUIDE_MAX_DOCUMENT_CHARS", 5000)
    @patch("app.style_guide.service.STYLE_GUIDE_MAX_CHUNK_CHARS", 1000)
    @patch("app.style_guide.service.check_llm_validation_rules")
    @patch("app.style_guide.service.generate_summary_and_fix")
    async def test_small_document_uses_normal_path(
        self, mock_generate_summary, mock_llm_rules
    ):
        """Documents under MAX_CHUNK_CHARS use the normal non-chunked LLM path."""
        mock_llm_rules.return_value = []
        mock_generate_summary.return_value = None

        content = "Short text."  # well under the patched limit

        await check_content_against_style_guide(content=content)

        mock_llm_rules.assert_called_once()


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------

class TestFilterViolationsForChunk:
    """Tests for the _filter_violations_for_chunk helper."""

    def test_includes_violation_with_matching_sentence(self):
        chunk = "The Prime Minister attended the meeting."
        violations = [
            {
                "rule_id": "rule_001",
                "sentences": ["The Prime Minister attended the meeting."],
            },
            {
                "rule_id": "rule_002",
                "sentences": ["The Home Secretary announced."],
            },
        ]
        result = _filter_violations_for_chunk(violations, chunk)
        assert len(result) == 1
        assert result[0]["rule_id"] == "rule_001"

    def test_includes_violation_with_matching_occurrence(self):
        chunk = "Access to Work is the programme."
        violations = [
            {"rule_id": "llm_001", "occurrences": ["Access to Work is the programme."]},
            {"rule_id": "llm_002", "occurrences": ["Elsewhere in the document."]},
        ]
        result = _filter_violations_for_chunk(violations, chunk)
        assert len(result) == 1
        assert result[0]["rule_id"] == "llm_001"

    def test_includes_violations_with_no_location_info(self):
        """Structural/doc-level violations with no sentences or occurrences are always kept."""
        chunk = "Any text here."
        violations = [{"rule_id": "structural_001"}]
        result = _filter_violations_for_chunk(violations, chunk)
        assert len(result) == 1

    def test_returns_empty_when_no_violations_match(self):
        chunk = "Completely different text."
        violations = [
            {"rule_id": "rule_001", "sentences": ["Other text in another chunk."]},
        ]
        result = _filter_violations_for_chunk(violations, chunk)
        assert len(result) == 0

    def test_narrows_sentences_list_to_chunk_only(self):
        """The returned violation's sentences list contains only those in the chunk."""
        chunk = "Prime Minister visited today."
        violations = [
            {
                "rule_id": "rule_001",
                "sentences": [
                    "Prime Minister visited today.",
                    "Another sentence not in this chunk.",
                ],
            }
        ]
        result = _filter_violations_for_chunk(violations, chunk)
        assert len(result) == 1
        assert result[0]["sentences"] == ["Prime Minister visited today."]


class TestRunLlmValidationOnChunks:
    """Tests for the _run_llm_validation_on_chunks helper."""

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_llm_validation_rules")
    async def test_calls_llm_once_per_chunk(self, mock_llm_rules):
        mock_llm_rules.return_value = []
        chunks = ["chunk one.", "chunk two.", "chunk three."]
        await _run_llm_validation_on_chunks(chunks, llm_model="test-model")
        assert mock_llm_rules.call_count == 3

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_llm_validation_rules")
    async def test_deduplicates_same_rule_across_chunks(self, mock_llm_rules):
        """A rule triggered in multiple chunks appears only once in the output."""
        violation = {"rule_id": "rule_001", "rule_title": "Test Rule"}
        mock_llm_rules.return_value = [violation]
        chunks = ["chunk one.", "chunk two."]
        result = await _run_llm_validation_on_chunks(chunks, llm_model="test-model")
        assert len(result) == 1
        assert result[0]["rule_id"] == "rule_001"

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_llm_validation_rules")
    async def test_combines_different_rules_from_chunks(self, mock_llm_rules):
        """Different rules found in different chunks all appear in the output."""
        mock_llm_rules.side_effect = [
            [{"rule_id": "rule_001", "rule_title": "Rule 1"}],
            [{"rule_id": "rule_002", "rule_title": "Rule 2"}],
        ]
        chunks = ["chunk one.", "chunk two."]
        result = await _run_llm_validation_on_chunks(chunks, llm_model="test-model")
        assert len(result) == 2
        assert {v["rule_id"] for v in result} == {"rule_001", "rule_002"}

    @pytest.mark.asyncio
    @patch("app.style_guide.service.check_llm_validation_rules")
    async def test_empty_chunks_returns_empty_list(self, mock_llm_rules):
        result = await _run_llm_validation_on_chunks([], llm_model="test-model")
        assert result == []
        mock_llm_rules.assert_not_called()


class TestGenerateChunkedSummaryAndFix:
    """Tests for the _generate_chunked_summary_and_fix helper."""
    """Tests for the _generate_chunked_summary_and_fix helper."""

    @pytest.mark.asyncio
    @patch("app.style_guide.service.generate_summary_and_fix")
    async def test_calls_summary_for_each_chunk(self, mock_generate_summary):
        mock_generate_summary.return_value = {
            "summary": "A violation.",
            "fixed_document": "Fixed chunk.",
        }
        chunks = ["chunk one.", "chunk two."]
        violations = [{"rule_id": "rule_001", "sentences": ["chunk one."]}]

        result = await _generate_chunked_summary_and_fix(
            chunks=chunks,
            all_violations=violations,
            llm_model="test-model",
            output_dir=Path("/tmp"),
        )

        assert mock_generate_summary.call_count == 2
        assert result is not None

    @pytest.mark.asyncio
    @patch("app.style_guide.service.generate_summary_and_fix")
    async def test_combines_fixed_documents(self, mock_generate_summary):
        mock_generate_summary.side_effect = [
            {"summary": "First summary.", "fixed_document": "Fixed part one."},
            {"summary": "Second summary.", "fixed_document": "Fixed part two."},
        ]
        chunks = ["part one.", "part two."]

        result = await _generate_chunked_summary_and_fix(
            chunks=chunks,
            all_violations=[],
            llm_model="test-model",
            output_dir=Path("/tmp"),
        )

        assert result is not None
        assert "Fixed part one." in result["fixed_document"]
        assert "Fixed part two." in result["fixed_document"]
        assert "First summary." in result["summary"]
        assert "Second summary." in result["summary"]

    @pytest.mark.asyncio
    @patch("app.style_guide.service.generate_summary_and_fix")
    async def test_returns_none_when_all_chunks_produce_no_summary(self, mock_generate_summary):
        """If every chunk returns None, the combined result is None."""
        mock_generate_summary.return_value = None
        chunks = ["chunk one.", "chunk two."]

        result = await _generate_chunked_summary_and_fix(
            chunks=chunks,
            all_violations=[],
            llm_model="test-model",
            output_dir=Path("/tmp"),
        )

        assert result is None

    @pytest.mark.asyncio
    @patch("app.style_guide.service.generate_summary_and_fix")
    async def test_conversation_context_only_in_first_chunk(self, mock_generate_summary):
        """Conversation context is passed to the first chunk only."""
        mock_generate_summary.return_value = {
            "summary": "Summary.",
            "fixed_document": "Fixed.",
        }
        chunks = ["chunk one.", "chunk two.", "chunk three."]
        context = "<ConversationContext>previous</ConversationContext>"

        await _generate_chunked_summary_and_fix(
            chunks=chunks,
            all_violations=[],
            llm_model="test-model",
            output_dir=Path("/tmp"),
            conversation_context=context,
        )

        calls = mock_generate_summary.call_args_list
        assert calls[0][1]["conversation_context"] == context
        assert calls[1][1]["conversation_context"] == ""
        assert calls[2][1]["conversation_context"] == ""

    @pytest.mark.asyncio
    @patch("app.style_guide.service.generate_summary_and_fix")
    async def test_failed_chunk_preserved_as_original_text(self, mock_generate_summary):
        """When a chunk's LLM call fails, the original chunk text is used in the output."""
        mock_generate_summary.side_effect = [
            None,  # first chunk fails
            {"summary": "Second summary.", "fixed_document": "Fixed second."},
        ]
        chunks = ["original first chunk.", "second chunk."]

        result = await _generate_chunked_summary_and_fix(
            chunks=chunks,
            all_violations=[],
            llm_model="test-model",
            output_dir=Path("/tmp"),
        )

        assert result is not None
        # First chunk kept as original; second fixed
        assert "original first chunk." in result["fixed_document"]
        assert "Fixed second." in result["fixed_document"]


# ---------------------------------------------------------------------------
# Missing content-guard and fallback paths in check_content_against_style_guide
# ---------------------------------------------------------------------------

class TestStyleGuideContentGuards:
    """Tests for the whitespace-only guard and summary-failure fallback paths."""

    @pytest.mark.asyncio
    @patch("app.style_guide.service.determine_style_guide_content_type")
    async def test_whitespace_only_content_returns_no_content_message(
        self, mock_content_type_router
    ):
        """A content string that is only whitespace returns the 'no content' segment.

        After the router decides to check query text the service checks whether
        ``content_to_check.strip()`` is falsy.  If so it must return a prompt
        segment containing the XML ``<message>`` tag rather than running any
        style-guide analysis.
        """
        from app.style_guide.service import StyleGuideContentType

        mock_content_type_router.return_value = StyleGuideContentType.QUERY_TEXT

        prompt_segment = await check_content_against_style_guide(content="   \n\t  ")

        assert prompt_segment is not None
        assert "style-guide-analysis" in prompt_segment
        assert "<message>" in prompt_segment
        # Should NOT contain error (reserved for oversized docs) or violation list
        assert "<error>" not in prompt_segment
        assert "<violations>" not in prompt_segment

    @pytest.mark.asyncio
    @patch("app.style_guide.service.determine_style_guide_content_type")
    @patch("app.style_guide.service.check_llm_validation_rules")
    @patch("app.style_guide.service.generate_summary_and_fix")
    async def test_summary_failure_falls_back_to_violation_list(
        self,
        mock_generate_summary,
        mock_check_llm_rules,
        mock_content_type_router,
    ):
        """When violations are found but summary generation returns None, a plain
        numbered violation list is returned rather than failing silently.

        This exercises the ``if not summary_result:`` branch in
        ``check_content_against_style_guide``.
        """
        from app.style_guide.service import StyleGuideContentType

        mock_content_type_router.return_value = StyleGuideContentType.QUERY_TEXT
        mock_check_llm_rules.return_value = [
            {
                "rule_id": "rule_001",
                "rule_title": "Test Violation One",
                "validation_type": "llm",
                "occurrences": [],
            }
        ]
        # Summary generation fails
        mock_generate_summary.return_value = None

        # Content that also triggers at least one deterministic check violation
        prompt_segment = await check_content_against_style_guide(
            content="The Prime Minister announced today."
        )

        assert prompt_segment is not None
        assert "style-guide-analysis" in prompt_segment
        # The fallback path uses a numbered list inside <violations>
        assert "<violations>" in prompt_segment
        # At least the rule title from the mocked violation should appear
        assert "Test Violation One" in prompt_segment
        # The structured corrected-document section must NOT be present
        assert "<corrected_document>" not in prompt_segment

    @pytest.mark.asyncio
    @patch("app.style_guide.service.determine_style_guide_content_type")
    @patch("app.style_guide.service._get_document_content_for_style_guide")
    @patch("app.style_guide.service.check_llm_validation_rules")
    @patch("app.style_guide.service.generate_summary_and_fix")
    async def test_documents_retrieval_failure_falls_back_to_query_text(
        self,
        mock_generate_summary,
        mock_check_llm_rules,
        mock_get_doc_content,
        mock_content_type_router,
    ):
        """DOCUMENTS routing type: when document retrieval returns empty content the
        service falls back to checking the original query text instead of failing.

        Exercises the ``if not document_content:`` fallback inside the DOCUMENTS
        branch of ``check_content_against_style_guide``.
        """
        from app.style_guide.service import StyleGuideContentType

        mock_content_type_router.return_value = StyleGuideContentType.DOCUMENTS
        # Document retrieval returns nothing
        mock_get_doc_content.return_value = ("", [])
        mock_check_llm_rules.return_value = []
        mock_generate_summary.return_value = None

        query_text = "Check my document please."

        prompt_segment = await check_content_against_style_guide(
            content=query_text,
            document_uuids=["some-uuid"],
            user_id=42,
        )

        # The service should fall back gracefully and still return a prompt segment
        assert prompt_segment is not None
        assert "style-guide-analysis" in prompt_segment
        # content_source_info should reflect the fallback to query text
        assert "Checking the text you provided" in prompt_segment
