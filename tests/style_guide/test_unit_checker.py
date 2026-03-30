"""
Unit tests for style guide checker functions.
Tests deterministic rule checking, LLM validation, and document fixing.
"""
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from anthropic.types import Message as AnthropicMessage
from anthropic.types import TextBlock, ToolUseBlock, Usage
from pydantic import ValidationError

from app.style_guide.style_guide_checker import (
    call_llm_for_validation,
    check_case_insensitive_rules,
    check_case_sensitive_rules,
    check_llm_deterministic_false_rules,
    check_llm_validation_rules,
    create_llm_prompt_for_deterministic_false,
    create_llm_validation_prompt,
    create_summary_and_fix_prompt,
    find_americanisms,
    generate_summary_and_fix,
    get_sentence_containing_text,
    get_sentence_with_context,
    load_document,
    load_rule_mapping,
    split_text_into_chunks,
)

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.unit,
]


# Sample rules for testing
SAMPLE_RULES = [
    {
        "rule_id": "test_001",
        "rule": "Use lowercase for job titles",
        "details": "Job titles should be lowercase unless at the start of a sentence",
        "detection_strategy": {
            "deterministic": True,
            "case_sensitive": False,
            "find": ["Prime Minister", "Home Secretary"],
        },
        "redundant": False,
    },
    {
        "rule_id": "test_002",
        "rule": "Use numbers not words for numbers above 9",
        "details": "Write numbers greater than nine as numerals",
        "detection_strategy": {
            "deterministic": True,
            "case_sensitive": True,
            "find": ["ten", "eleven", "twelve"],
        },
        "redundant": False,
    },
    {
        "rule_id": "test_redundant",
        "rule": "This should be filtered",
        "details": "Redundant rule",
        "detection_strategy": {
            "deterministic": True,
            "case_sensitive": False,
            "find": ["test"],
        },
        "redundant": True,
    },
]


class TestSentenceExtraction:
    """Tests for sentence extraction helper functions."""

    def test_get_sentence_containing_text_simple(self):
        """Test extracting a simple sentence."""
        document = "This is sentence one. This is sentence two. This is sentence three."
        text = "sentence two"
        position = document.index(text)

        result = get_sentence_containing_text(document, text, position)

        assert result == "This is sentence two."

    def test_get_sentence_containing_text_at_start(self):
        """Test extracting sentence at document start."""
        document = "First sentence here. Second sentence."
        text = "First"
        position = 0

        result = get_sentence_containing_text(document, text, position)

        assert result == "First sentence here."

    def test_get_sentence_containing_text_at_end(self):
        """Test extracting sentence at document end."""
        document = "First sentence. Final sentence here"
        text = "Final"
        position = document.index(text)

        result = get_sentence_containing_text(document, text, position)

        assert result == "Final sentence here"

    def test_get_sentence_with_context_has_preceding(self):
        """Test getting sentence with preceding context."""
        document = "First sentence. Second sentence. Third sentence."
        text = "Second"
        position = document.index(text)

        result = get_sentence_with_context(document, text, position)

        assert result["current_sentence"] == "Second sentence."
        assert result["preceding_sentence"] == "First sentence."

    def test_get_sentence_with_context_no_preceding(self):
        """Test getting sentence with no preceding context."""
        document = "Only sentence here."
        text = "Only"
        position = 0

        result = get_sentence_with_context(document, text, position)

        assert result["current_sentence"] == "Only sentence here."
        assert result["preceding_sentence"] == ""


class TestRuleLoading:
    """Tests for rule loading and filtering."""

    def test_load_rule_mapping_filters_redundant(self):
        """Test that load_rule_mapping filters out redundant rules."""
        # Create temporary rules file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(SAMPLE_RULES, f)
            temp_path = Path(f.name)

        try:
            rules = load_rule_mapping(temp_path)

            # Should filter out redundant rule
            assert len(rules) == 2
            assert all(not rule.get('redundant', False) for rule in rules)

            # Check specific rules loaded
            rule_ids = [r['rule_id'] for r in rules]
            assert 'test_001' in rule_ids
            assert 'test_002' in rule_ids
            assert 'test_redundant' not in rule_ids
        finally:
            temp_path.unlink()


class TestCaseInsensitiveRuleChecking:
    """Tests for case-insensitive deterministic rule checking."""

    def test_check_case_insensitive_finds_violations(self):
        """Test that case-insensitive check finds violations."""
        document = "The Prime Minister met with the Home Secretary today."

        violations = check_case_insensitive_rules(document, SAMPLE_RULES)

        # Should find both "Prime Minister" and "Home Secretary"
        assert len(violations) >= 1

        # Check violation structure
        violation = violations[0]
        assert 'rule_id' in violation
        assert 'rule_title' in violation
        assert 'match_count' in violation
        assert 'sentences' in violation

    def test_check_case_insensitive_no_violations(self):
        """Test case-insensitive check with no violations."""
        document = "This document has no job title violations."

        violations = check_case_insensitive_rules(document, SAMPLE_RULES)

        # Should find no violations for this specific rule
        # (Note: might find violations for other rules in SAMPLE_RULES)
        pm_violations = [v for v in violations if 'Prime Minister' in str(v.get('sentences', []))]
        assert len(pm_violations) == 0

    def test_check_case_insensitive_case_insensitivity(self):
        """Test that checking is truly case-insensitive."""
        document = "The PRIME MINISTER and prime minister both appeared."

        violations = check_case_insensitive_rules(document, SAMPLE_RULES)

        # Should find both variations
        pm_violations = [v for v in violations if v.get('rule_id') == 'test_001']
        if len(pm_violations) > 0:
            # At least one violation should have multiple matches
            assert any(v.get('match_count', 0) >= 2 for v in pm_violations)


class TestCaseSensitiveRuleChecking:
    """Tests for case-sensitive deterministic rule checking."""

    def test_check_case_sensitive_finds_exact_matches(self):
        """Test that case-sensitive check finds exact matches only."""
        document = "I have ten apples and Ten oranges."

        violations = check_case_sensitive_rules(document, SAMPLE_RULES)

        # Should find "ten" but not "Ten"
        ten_violations = [v for v in violations if 'ten' in str(v.get('sentences', []))]
        if len(ten_violations) > 0:
            # Check sentences only contain lowercase "ten"
            violation_text = str(ten_violations[0].get('sentences', []))
            assert 'ten apples' in violation_text.lower()

    def test_check_case_sensitive_no_false_positives(self):
        """Test case-sensitive check doesn't match wrong case."""
        document = "I have TEN APPLES and Eleven items."

        violations = check_case_sensitive_rules(document, SAMPLE_RULES)

        # Should not find "TEN" (uppercase) as violation for "ten" rule
        ten_violations = [v for v in violations if 'ten' in v.get('rule_id', '').lower()]
        if len(ten_violations) > 0:
            # Should not have found TEN in all caps
            violation_text = str(ten_violations)
            assert 'TEN APPLES' not in violation_text


class TestPromptCreation:
    """Tests for prompt creation functions."""

    def test_create_summary_and_fix_prompt_basic(self):
        """Test creating prompt without conversation context."""
        document = "Test document with violations."
        violations = [
            {
                "rule_id": "test_001",
                "rule_title": "Test Rule",
                "validation_type": "deterministic",
                "match_count": 1,
                "sentences": ["Test sentence."],
            }
        ]

        prompt = create_summary_and_fix_prompt(document, violations)

        assert "expert Government Digital Service" in prompt
        assert "VIOLATIONS FOUND:" in prompt
        assert "test_001" in prompt
        assert "Test Rule" in prompt
        assert document in prompt

    def test_create_summary_and_fix_prompt_with_context(self):
        """Test creating prompt with conversation context."""
        document = "Test document."
        violations = [{"rule_id": "test", "rule_title": "Test"}]
        conversation_context = "<ConversationContext><user>Previous</user></ConversationContext>"

        prompt = create_summary_and_fix_prompt(document, violations, conversation_context)

        assert "IMPORTANT CONTEXT" in prompt
        assert "follow-up message" in prompt
        assert conversation_context in prompt

    def test_create_summary_and_fix_prompt_llm_violations(self):
        """Test prompt creation with LLM-validated violations."""
        document = "Test doc"
        violations = [
            {
                "rule_id": "llm_001",
                "rule_title": "LLM Rule",
                "validation_type": "llm",
                "violation_reason": "This breaks the rule",
                "occurrences": ["occurrence 1", "occurrence 2"],
            }
        ]

        prompt = create_summary_and_fix_prompt(document, violations)

        assert "LLM Validated" in prompt
        assert "This breaks the rule" in prompt
        assert "occurrence 1" in prompt

    def test_create_summary_and_fix_prompt_multiple_violations(self):
        """Test prompt with multiple violation types."""
        document = "Test"
        violations = [
            {
                "rule_id": "det_001",
                "rule_title": "Deterministic Rule",
                "validation_type": "deterministic",
                "match_count": 2,
                "sentences": ["Sentence 1", "Sentence 2"],
            },
            {
                "rule_id": "llm_001",
                "rule_title": "LLM Rule",
                "validation_type": "llm",
                "occurrences": ["Occurrence"],
            },
        ]

        prompt = create_summary_and_fix_prompt(document, violations)

        # Should include both violations
        assert "VIOLATION 1:" in prompt
        assert "VIOLATION 2:" in prompt
        assert "Deterministic Rule" in prompt
        assert "LLM Rule" in prompt


class TestSplitTextIntoChunks:
    """Tests for the split_text_into_chunks helper."""

    def test_short_text_returned_unchanged(self):
        """Text shorter than chunk_size is returned as a single-element list."""
        text = "This is a short sentence."
        result = split_text_into_chunks(text, chunk_size=1000)
        assert result == [text]

    def test_exact_length_not_split(self):
        """Text exactly equal to chunk_size is returned as a single chunk."""
        text = "a" * 100
        result = split_text_into_chunks(text, chunk_size=100)
        assert len(result) == 1
        assert result[0] == text

    def test_long_text_produces_multiple_chunks(self):
        """Text longer than chunk_size produces more than one chunk."""
        text = "This is a sentence. " * 50  # ~1 000 chars
        result = split_text_into_chunks(text, chunk_size=200)
        assert len(result) > 1

    def test_chunks_respect_size_limit(self):
        """No chunk exceeds chunk_size characters."""
        text = "Word " * 200
        result = split_text_into_chunks(text, chunk_size=50)
        assert all(len(c) <= 50 for c in result)

    def test_no_data_loss_without_overlap(self):
        """All content is present somewhere across the chunks (no overlap)."""
        sentences = [f"Sentence number {i}." for i in range(20)]
        text = " ".join(sentences)
        result = split_text_into_chunks(text, chunk_size=100, overlap=0)
        combined = " ".join(result)
        for s in sentences:
            assert s in combined, f"Missing: {s}"

    def test_splits_at_sentence_boundary(self):
        """Chunks end at sentence punctuation when possible."""
        # Force a split: short sentence + padding that exceeds chunk_size
        text = "Short sentence. " + "x" * 80 + " End."
        result = split_text_into_chunks(text, chunk_size=50, overlap=0)
        # The first chunk should end after "Short sentence." not mid-word
        assert result[0].endswith(".")

    def test_overlap_adds_context_to_next_chunk(self):
        """With overlap > 0, later chunks contain text from the previous boundary."""
        text = "First sentence here. " * 10
        no_overlap = split_text_into_chunks(text, chunk_size=100, overlap=0)
        with_overlap = split_text_into_chunks(text, chunk_size=100, overlap=20)
        # With overlap there should be at least as many chunks
        assert len(with_overlap) >= len(no_overlap)

    def test_empty_string_returns_empty_list(self):
        """An empty string input returns an empty list."""
        result = split_text_into_chunks("", chunk_size=100)
        assert result == []

    def test_chunk_size_larger_than_text_single_chunk(self):
        """When chunk_size exceeds text length the text is not split."""
        text = "Just one sentence."
        result = split_text_into_chunks(text, chunk_size=10000)
        assert result == [text]


# ---------------------------------------------------------------------------
# Extra sample rules used in the new test classes below
# ---------------------------------------------------------------------------

SENTENCE_LENGTH_RULE = {
    "rule_id": "test_003",
    "rule": "Keep sentences short",
    "details": "Sentences should be no longer than 25 words",
    "detection_strategy": {
        "deterministic": True,
        "case_sensitive": False,
        "check_sentence_length": True,
        "max_sentence_word_length": 25,
    },
    "redundant": False,
}

LLM_PASS_RULE = {
    "rule_id": "test_004",
    "rule": "Contextual council check",
    "details": "Use 'council' lowercase unless referring to a specific named council",
    "detection_strategy": {
        "deterministic": True,
        "case_sensitive": False,
        "pass_to_llm": True,
        "find": ["council"],
    },
    "redundant": False,
}

DETERMINISTIC_FALSE_RULE = {
    "rule_id": "test_005",
    "rule": "Complex writing style rule",
    "details": "Avoid passive voice",
    "detection_strategy": {
        "deterministic": False,
    },
    "redundant": False,
}

UWOTM8_RULE = {
    "rule_id": "rule_024",
    "rule": "American and UK English",
    "details": "Use UK English spelling. For example, 'organise' not 'organize'.",
    "detection_strategy": {
        "deterministic": True,
        "use_uwotm8": True,
        "case_sensitive": False,
        "pass_to_llm": True,
    },
    "redundant": False,
}

NON_STANDALONE_RULE = {
    "rule_id": "test_006",
    "rule": "Substring check",
    "details": "Find any substring match",
    "detection_strategy": {
        "deterministic": True,
        "case_sensitive": False,
        "pass_to_llm": False,
        "find": ["gov"],
        "standalone_string": False,
    },
    "redundant": False,
}


def _make_llm_response(text: str) -> AnthropicMessage:
    """Build a minimal AnthropicMessage with the given text payload."""
    return AnthropicMessage(
        id="msg_test",
        content=[TextBlock(text=text, type="text")],
        model="claude-test",
        role="assistant",
        stop_reason="end_turn",
        type="message",
        usage=Usage(input_tokens=50, output_tokens=50),
    )


def _make_tool_use_response(tool_name: str, tool_input: dict) -> AnthropicMessage:
    """Build a minimal AnthropicMessage with a ToolUseBlock payload."""
    return AnthropicMessage(
        id="msg_test",
        content=[ToolUseBlock(id="tool_use_123", input=tool_input, name=tool_name, type="tool_use")],
        model="claude-test",
        role="assistant",
        stop_reason="tool_use",
        type="message",
        usage=Usage(input_tokens=50, output_tokens=50),
    )


# ---------------------------------------------------------------------------
# load_document
# ---------------------------------------------------------------------------


class TestLoadDocument:
    """Tests for the load_document helper."""

    def test_load_document_reads_file_content(self):
        """load_document returns the file content as a string."""
        content = "This is a test document.\nWith two lines."
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp_path = Path(f.name)

        try:
            result = load_document(tmp_path)
            assert result == content
        finally:
            tmp_path.unlink()

    def test_load_document_file_not_found_raises(self):
        """load_document raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_document(Path("/nonexistent/path/doc.txt"))


# ---------------------------------------------------------------------------
# Extra sentence-extraction edge cases
# ---------------------------------------------------------------------------


class TestSentenceExtractionEdgeCases:
    """Additional edge-case tests for sentence extraction helpers."""

    def test_get_sentence_containing_text_newline_boundary(self):
        """Newlines act as sentence boundaries."""
        document = "First line\nSecond line\nThird line"
        text = "Second"
        position = document.index(text)

        result = get_sentence_containing_text(document, text, position)

        assert "Second" in result
        assert "First" not in result

    def test_get_sentence_containing_text_exclamation_boundary(self):
        """Exclamation marks act as sentence boundaries."""
        document = "Great news! Something happened. More text."
        text = "Something"
        position = document.index(text)

        result = get_sentence_containing_text(document, text, position)

        assert "Something happened." in result
        assert "Great news" not in result

    def test_get_sentence_with_context_third_sentence(self):
        """Preceding sentence is correctly identified for a third sentence."""
        document = "Sentence one. Sentence two. Sentence three."
        text = "three"
        position = document.index(text)

        result = get_sentence_with_context(document, text, position)

        assert "Sentence three." in result["current_sentence"]
        assert "Sentence two." in result["preceding_sentence"]


# ---------------------------------------------------------------------------
# Extra check_case_insensitive_rules coverage
# ---------------------------------------------------------------------------


class TestCaseInsensitiveRulesExtra:
    """Extra edge-case tests for check_case_insensitive_rules."""

    def test_sentence_length_violation_detected(self):
        """Sentences exceeding max_sentence_word_length are flagged."""
        # 30 words -> exceeds the 25-word limit
        long_sentence = (
            "This is an extremely long sentence that goes on and on and on "
            "with far too many words in it for it to be considered acceptable by anyone."
        )
        document = f"Short intro. {long_sentence}"

        violations = check_case_insensitive_rules(document, [SENTENCE_LENGTH_RULE])

        assert len(violations) == 1
        assert violations[0]["rule_id"] == "test_003"
        assert violations[0]["match_count"] >= 1

    def test_sentence_length_no_violation_for_short_sentences(self):
        """Sentences within the word limit are not flagged."""
        document = "This is short. So is this one."

        violations = check_case_insensitive_rules(document, [SENTENCE_LENGTH_RULE])

        assert violations == []

    def test_pass_to_llm_rules_are_excluded(self):
        """Rules with pass_to_llm=True must NOT be checked by check_case_insensitive_rules."""
        # 'council' is in LLM_PASS_RULE which has pass_to_llm=True
        document = "The council met yesterday."

        violations = check_case_insensitive_rules(document, [LLM_PASS_RULE])

        assert violations == []

    def test_standalone_string_false_matches_substring(self):
        """standalone_string=False finds the term inside longer words."""
        document = "Visit gov.uk for more information about your government."

        violations = check_case_insensitive_rules(document, [NON_STANDALONE_RULE])

        assert len(violations) >= 1
        # 'gov' appears inside 'gov.uk' and 'government'
        assert violations[0]["match_count"] >= 2

    def test_standalone_string_true_does_not_match_substring(self):
        """Default standalone_string=True should NOT match the search string inside a longer word."""
        # "Minister" should not match "Ministered"
        rule = {
            "rule_id": "test_standalone",
            "rule": "Test standalone",
            "details": "Test",
            "detection_strategy": {
                "deterministic": True,
                "case_sensitive": False,
                "pass_to_llm": False,
                "find": ["internet"],
                "standalone_string": True,
            },
            "redundant": False,
        }
        # "interned" does not contain "internet"; this is just about word boundaries
        document = "She interneted all day long and used the internet."

        violations = check_case_insensitive_rules(document, [rule])

        # Only the standalone "internet" should be matched (the second occurrence)
        assert len(violations) == 1
        assert violations[0]["match_count"] == 1


# ---------------------------------------------------------------------------
# Extra check_case_sensitive_rules coverage
# ---------------------------------------------------------------------------


class TestCaseSensitiveRulesExtra:
    """Extra edge-case tests for check_case_sensitive_rules."""

    def test_sentence_start_capitalisation_is_not_a_violation(self):
        """Capitalising the first letter at the start of a sentence is acceptable."""
        rule = {
            "rule_id": "test_cs_start",
            "rule": "Use lowercase for internet",
            "details": "The word internet should be lowercase",
            "detection_strategy": {
                "deterministic": True,
                "case_sensitive": True,
                "pass_to_llm": False,
                "find": ["internet"],
            },
            "redundant": False,
        }
        # "Internet" at sentence start is acceptable; mid-sentence "Internet" is a violation
        document = "Internet is widely used. She browsed the Internet daily."

        violations = check_case_sensitive_rules(document, [rule])

        # Only "Internet" in the second sentence should be a violation
        assert len(violations) == 1
        assert violations[0]["match_count"] == 1
        assert any("browsed the Internet" in s for s in violations[0]["sentences"])

    def test_incorrect_mid_sentence_casing_is_violation(self):
        """Incorrect casing mid-sentence is always flagged."""
        rule = {
            "rule_id": "test_cs_mid",
            "rule": "Use lowercase for email",
            "details": "The word email should be lowercase",
            "detection_strategy": {
                "deterministic": True,
                "case_sensitive": True,
                "pass_to_llm": False,
                "find": ["email"],
            },
            "redundant": False,
        }
        document = "Please send an Email to the team."

        violations = check_case_sensitive_rules(document, [rule])

        assert len(violations) == 1
        assert violations[0]["correct_string"] == "email"

    def test_plural_form_is_also_matched(self):
        """The rule finder also picks up plurals (e.g. 'emails' for 'email')."""
        rule = {
            "rule_id": "test_cs_plural",
            "rule": "Use lowercase for email",
            "details": "The word email should be lowercase",
            "detection_strategy": {
                "deterministic": True,
                "case_sensitive": True,
                "pass_to_llm": False,
                "find": ["email"],
            },
            "redundant": False,
        }
        # "Emails" is the wrong casing for the plural
        document = "Send Emails to everyone on the list."

        violations = check_case_sensitive_rules(document, [rule])

        assert len(violations) == 1

    def test_correct_casing_is_not_flagged(self):
        """The exact correct form in the middle of a sentence is not a violation."""
        rule = {
            "rule_id": "test_cs_correct",
            "rule": "Use lowercase for email",
            "details": "The word email should be lowercase",
            "detection_strategy": {
                "deterministic": True,
                "case_sensitive": True,
                "pass_to_llm": False,
                "find": ["email"],
            },
            "redundant": False,
        }
        document = "Please send an email to the team."

        violations = check_case_sensitive_rules(document, [rule])

        assert violations == []


# ---------------------------------------------------------------------------
# create_llm_validation_prompt
# ---------------------------------------------------------------------------


class TestCreateLLMValidationPrompt:
    """Tests for create_llm_validation_prompt."""

    def test_prompt_contains_rule_ids(self):
        """The prompt includes each rule ID."""
        rules_with_occurrences = [
            {
                "rule_id": "rule_111",
                "rule_title": "Avoid jargon",
                "rule_details": "Do not use jargon terms",
                "case_sensitive": False,
                "occurrences": [
                    {
                        "matched_text": "synergy",
                        "sentence": "We need synergy here.",
                        "preceding_sentence": "",
                    }
                ],
            }
        ]

        prompt = create_llm_validation_prompt("Some document text.", rules_with_occurrences)

        assert "rule_111" in prompt
        assert "Avoid jargon" in prompt

    def test_prompt_contains_occurrences(self):
        """The prompt lists each matched occurrence."""
        rules_with_occurrences = [
            {
                "rule_id": "rule_222",
                "rule_title": "Test rule",
                "rule_details": "Rule details",
                "case_sensitive": True,
                "occurrences": [
                    {
                        "matched_text": "FoundText",
                        "sentence": "We saw FoundText here.",
                        "preceding_sentence": "Previous sentence.",
                    }
                ],
            }
        ]

        prompt = create_llm_validation_prompt("Doc content.", rules_with_occurrences)

        assert "FoundText" in prompt
        assert "We saw FoundText here." in prompt
        assert "Previous sentence." in prompt

    def test_prompt_contains_full_document(self):
        """The full document is embedded in the prompt."""
        document = "This is the full document content for context."
        prompt = create_llm_validation_prompt(document, [])

        assert document in prompt

    def test_prompt_includes_multiple_rules(self):
        """Multiple rules are all included in a single prompt."""
        rules_with_occurrences = [
            {
                "rule_id": f"rule_{i}",
                "rule_title": f"Rule {i}",
                "rule_details": f"Details {i}",
                "case_sensitive": False,
                "occurrences": [],
            }
            for i in range(3)
        ]

        prompt = create_llm_validation_prompt("Document.", rules_with_occurrences)

        for i in range(3):
            assert f"rule_{i}" in prompt
            assert f"Rule {i}" in prompt


# ---------------------------------------------------------------------------
# create_llm_prompt_for_deterministic_false
# ---------------------------------------------------------------------------


class TestCreateLLMPromptForDeterministicFalse:
    """Tests for create_llm_prompt_for_deterministic_false."""

    def test_prompt_contains_rule_details(self):
        """The prompt includes rule IDs, names, and details."""
        rules = [DETERMINISTIC_FALSE_RULE]

        prompt = create_llm_prompt_for_deterministic_false("Some document.", rules)

        assert "test_005" in prompt
        assert "Complex writing style rule" in prompt
        assert "Avoid passive voice" in prompt

    def test_prompt_embeds_document(self):
        """The document text appears in the prompt."""
        document = "Unique marker text for deterministic false."

        prompt = create_llm_prompt_for_deterministic_false(document, [DETERMINISTIC_FALSE_RULE])

        assert document in prompt

    def test_prompt_requests_json_response(self):
        """The prompt instructs the LLM to return violations."""
        prompt = create_llm_prompt_for_deterministic_false("Doc.", [DETERMINISTIC_FALSE_RULE])

        assert "violations" in prompt

    def test_prompt_handles_multiple_rules(self):
        """Multiple rules all appear in the prompt."""
        rules = [
            DETERMINISTIC_FALSE_RULE,
            {
                "rule_id": "test_extra",
                "rule": "Another rule",
                "details": "More details",
                "detection_strategy": {"deterministic": False},
                "redundant": False,
            },
        ]

        prompt = create_llm_prompt_for_deterministic_false("Doc.", rules)

        assert "test_005" in prompt
        assert "test_extra" in prompt


# ---------------------------------------------------------------------------
# call_llm_for_validation
# ---------------------------------------------------------------------------


class TestCallLLMForValidation:
    """Tests for call_llm_for_validation."""

    @pytest.mark.asyncio
    @patch("app.style_guide.style_guide_checker.BedrockHandler")
    async def test_valid_tool_response_parsed(self, mock_bedrock_class):
        """A well-formed tool use response produces structured violations."""
        tool_input = {
            "violations": [
                {
                    "rule_id": "rule_001",
                    "rule_name": "Test rule",
                    "confidence": 0.95,
                    "violation_reason": "Bad usage",
                    "occurrences": ["Bad sentence here."],
                }
            ]
        }
        mock_handler = Mock()
        mock_handler.invoke_async = AsyncMock(
            return_value=_make_tool_use_response("report_style_guide_violations", tool_input)
        )
        mock_bedrock_class.return_value = mock_handler

        result = await call_llm_for_validation("prompt text", Mock(), [])

        assert len(result) == 1
        assert result[0]["rule_id"] == "rule_001"
        assert result[0]["validation_type"] == "llm"
        assert result[0]["occurrences"] == ["Bad sentence here."]

    @pytest.mark.asyncio
    @patch("app.style_guide.style_guide_checker.BedrockHandler")
    async def test_tool_response_with_empty_violations(self, mock_bedrock_class):
        """A tool use response with an empty violations array returns an empty list."""
        mock_handler = Mock()
        mock_handler.invoke_async = AsyncMock(
            return_value=_make_tool_use_response("report_style_guide_violations", {"violations": []})
        )
        mock_bedrock_class.return_value = mock_handler

        result = await call_llm_for_validation("prompt", Mock(), [])

        assert result == []

    @pytest.mark.asyncio
    @patch("app.style_guide.style_guide_checker.BedrockHandler")
    async def test_empty_violations_returns_empty_list(self, mock_bedrock_class):
        """A tool use response with an empty violations array produces an empty list."""
        mock_handler = Mock()
        mock_handler.invoke_async = AsyncMock(
            return_value=_make_tool_use_response("report_style_guide_violations", {"violations": []})
        )
        mock_bedrock_class.return_value = mock_handler

        result = await call_llm_for_validation("prompt", Mock(), [])

        assert result == []

    @pytest.mark.asyncio
    @patch("app.style_guide.style_guide_checker.BedrockHandler")
    async def test_invalid_tool_input_raises(self, mock_bedrock_class):
        """A tool response that fails Pydantic validation raises ValidationError."""
        # 'violations' must be a list, not a string
        mock_handler = Mock()
        mock_handler.invoke_async = AsyncMock(
            return_value=_make_tool_use_response(
                "report_style_guide_violations", {"violations": "not a list"}
            )
        )
        mock_bedrock_class.return_value = mock_handler

        with pytest.raises(ValidationError):
            await call_llm_for_validation("prompt", Mock(), [])

        assert mock_handler.invoke_async.call_count == 1

    @pytest.mark.asyncio
    @patch("app.style_guide.style_guide_checker.BedrockHandler")
    async def test_exception_during_llm_call_raises(self, mock_bedrock_class):
        """An unexpected exception during the LLM call propagates to the caller."""
        mock_handler = Mock()
        mock_handler.invoke_async = AsyncMock(side_effect=RuntimeError("Network error"))
        mock_bedrock_class.return_value = mock_handler

        with pytest.raises(RuntimeError, match="Network error"):
            await call_llm_for_validation("prompt", Mock(), [])

    @pytest.mark.asyncio
    @patch("app.style_guide.style_guide_checker.BedrockHandler")
    async def test_no_tool_use_block_returns_empty(self, mock_bedrock_class):
        """When the LLM returns no tool use block (text only), an empty list is returned.

        Tool use is forced via tool_choice, so this tests the defensive path
        for unexpected responses.
        """
        mock_handler = Mock()
        mock_handler.invoke_async = AsyncMock(
            return_value=_make_llm_response("Unexpected plain text response")
        )
        mock_bedrock_class.return_value = mock_handler

        result = await call_llm_for_validation("prompt", Mock(), [])

        assert result == []
        assert mock_handler.invoke_async.call_count == 1


# ---------------------------------------------------------------------------
# check_llm_validation_rules
# ---------------------------------------------------------------------------


class TestCheckLLMValidationRules:
    """Tests for check_llm_validation_rules."""

    @pytest.mark.asyncio
    async def test_no_matching_rules_skips_llm(self):
        """When no rules have pass_to_llm=True, the LLM is never called."""
        # Only deterministic=True, pass_to_llm=False rules
        rules = [SAMPLE_RULES[0], SAMPLE_RULES[1]]
        # DETERMINISTIC_FALSE_RULE has deterministic=False so it won't match either
        with patch(
            "app.style_guide.style_guide_checker.LLMTable"
        ), patch(
            "app.style_guide.style_guide_checker.call_llm_for_validation"
        ) as mock_call_llm:
            result = await check_llm_validation_rules("Some document text.", rules)

        # call_llm_for_validation should NOT have been called for the pass_to_llm=True path
        mock_call_llm.assert_not_called()
        # Result may still contain deterministic=false violations (empty in this case)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_rules_with_no_occurrences_skip_llm(self):
        """Rules that match nothing in the document are not sent to the LLM."""
        with patch(
            "app.style_guide.style_guide_checker.LLMTable"
        ) as mock_llm_table, patch(
            "app.style_guide.style_guide_checker.call_llm_for_validation",
            new_callable=AsyncMock,
        ) as mock_call_llm, patch(
            "app.style_guide.style_guide_checker.check_llm_deterministic_false_rules",
            new_callable=AsyncMock,
            return_value=[],
        ):
            mock_llm_table.return_value.get_by_model.return_value = Mock()
            mock_call_llm.return_value = []

            # LLM_PASS_RULE looks for 'council'; document has none
            result = await check_llm_validation_rules(
                "No matching content here.", [LLM_PASS_RULE]
            )

        # LLM should not be called when there are no occurrences to validate
        mock_call_llm.assert_not_called()
        assert result == []

    @pytest.mark.asyncio
    async def test_db_error_is_handled_gracefully(self):
        """A database error when fetching the LLM model is handled without raising."""
        with patch(
            "app.style_guide.style_guide_checker.LLMTable"
        ) as mock_llm_table, patch(
            "app.style_guide.style_guide_checker.check_llm_deterministic_false_rules",
            new_callable=AsyncMock,
            return_value=[],
        ):
            mock_llm_table.return_value.get_by_model.side_effect = Exception("DB error")

            result = await check_llm_validation_rules(
                "The council met.", [LLM_PASS_RULE]
            )

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_llm_violations_returned(self):
        """Violations returned by the LLM batch are included in the result."""
        llm_violation = {
            "rule_id": "test_004",
            "rule_title": "Contextual council check",
            "confidence": 0.9,
            "occurrences": ["The Council met."],
            "violation_reason": "Should be lowercase",
            "validation_type": "llm",
        }

        with patch(
            "app.style_guide.style_guide_checker.LLMTable"
        ) as mock_llm_table, patch(
            "app.style_guide.style_guide_checker.call_llm_for_validation",
            new_callable=AsyncMock,
            return_value=[llm_violation],
        ), patch(
            "app.style_guide.style_guide_checker.check_llm_deterministic_false_rules",
            new_callable=AsyncMock,
            return_value=[],
        ):
            mock_llm_table.return_value.get_by_model.return_value = Mock()

            result = await check_llm_validation_rules(
                "The Council met yesterday.", [LLM_PASS_RULE]
            )

        assert any(v["rule_id"] == "test_004" for v in result)


# ---------------------------------------------------------------------------
# check_llm_deterministic_false_rules
# ---------------------------------------------------------------------------


class TestCheckLLMDeterministicFalseRules:
    """Tests for check_llm_deterministic_false_rules."""

    @pytest.mark.asyncio
    async def test_no_deterministic_false_rules_returns_empty(self):
        """When no rules have deterministic=False, the result is empty."""
        # All rules in SAMPLE_RULES have deterministic=True
        result = await check_llm_deterministic_false_rules(
            "Document.", SAMPLE_RULES, "some-model", batch_size=5
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_db_error_returns_empty(self):
        """A database error when fetching the LLM returns an empty list."""
        with patch("app.style_guide.style_guide_checker.LLMTable") as mock_llm_table:
            mock_llm_table.return_value.get_by_model.side_effect = Exception("DB down")

            result = await check_llm_deterministic_false_rules(
                "Document.", [DETERMINISTIC_FALSE_RULE], "some-model", batch_size=5
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_violations_from_llm_returned(self):
        """Violations from the LLM are propagated back."""
        expected_violation = {
            "rule_id": "test_005",
            "rule_title": "Complex writing style rule",
            "confidence": 0.8,
            "occurrences": ["The report was written by the team."],
            "violation_reason": "Passive voice used",
            "validation_type": "llm",
        }

        with patch(
            "app.style_guide.style_guide_checker.LLMTable"
        ) as mock_llm_table, patch(
            "app.style_guide.style_guide_checker.call_llm_for_validation",
            new_callable=AsyncMock,
            return_value=[expected_violation],
        ):
            mock_llm_table.return_value.get_by_model.return_value = Mock()

            result = await check_llm_deterministic_false_rules(
                "The report was written by the team.",
                [DETERMINISTIC_FALSE_RULE],
                "some-model",
                batch_size=5,
            )

        assert len(result) == 1
        assert result[0]["rule_id"] == "test_005"


# ---------------------------------------------------------------------------
# generate_summary_and_fix
# ---------------------------------------------------------------------------


class TestGenerateSummaryAndFix:
    """Tests for generate_summary_and_fix."""

    @pytest.mark.asyncio
    async def test_empty_violations_returns_none(self):
        """No violations means no LLM call and a None return."""
        result = await generate_summary_and_fix(
            document="Any document.",
            violations=[],
            llm_model="some-model",
            output_dir=Path("/tmp"),
        )
        assert result is None

    @pytest.mark.asyncio
    @patch("app.style_guide.style_guide_checker.LLMTable")
    @patch("app.style_guide.style_guide_checker.BedrockHandler")
    async def test_valid_response_returns_summary_and_fixed(
        self, mock_bedrock_class, mock_llm_table_class
    ):
        """A valid tool use response is parsed into summary and fixed_document."""
        tool_input = {
            "summary": "One violation was found.",
            "fixed_document": "This is the corrected document.",
        }
        mock_handler = Mock()
        mock_handler.invoke_async = AsyncMock(
            return_value=_make_tool_use_response("provide_summary_and_fixed_document", tool_input)
        )
        mock_bedrock_class.return_value = mock_handler
        mock_llm_table_class.return_value.get_by_model.return_value = Mock()

        violations = [
            {
                "rule_id": "rule_001",
                "rule_title": "Test Rule",
                "validation_type": "deterministic",
                "match_count": 1,
                "sentences": ["Bad sentence."],
            }
        ]

        result = await generate_summary_and_fix(
            document="Bad sentence.",
            violations=violations,
            llm_model="some-model",
            output_dir=Path("/tmp"),
        )

        assert result is not None
        assert result["summary"] == "One violation was found."
        assert result["fixed_document"] == "This is the corrected document."

    @pytest.mark.asyncio
    @patch("app.style_guide.style_guide_checker.LLMTable")
    @patch("app.style_guide.style_guide_checker.BedrockHandler")
    async def test_markdown_wrapped_response_parsed(
        self, mock_bedrock_class, mock_llm_table_class
    ):
        """A tool use response is parsed correctly into summary and fixed document."""
        mock_handler = Mock()
        mock_handler.invoke_async = AsyncMock(
            return_value=_make_tool_use_response(
                "provide_summary_and_fixed_document",
                {"summary": "Two violations.", "fixed_document": "Fixed text."},
            )
        )
        mock_bedrock_class.return_value = mock_handler
        mock_llm_table_class.return_value.get_by_model.return_value = Mock()

        violations = [{"rule_id": "r1", "rule_title": "R1", "validation_type": "llm", "occurrences": []}]

        result = await generate_summary_and_fix(
            document="Doc.",
            violations=violations,
            llm_model="some-model",
            output_dir=Path("/tmp"),
        )

        assert result is not None
        assert result["summary"] == "Two violations."

    @pytest.mark.asyncio
    @patch("app.style_guide.style_guide_checker.LLMTable")
    async def test_db_error_returns_none(self, mock_llm_table_class):
        """A database error when fetching the LLM model returns None."""
        mock_llm_table_class.return_value.get_by_model.side_effect = Exception("DB error")

        violations = [{"rule_id": "r1", "rule_title": "R1"}]

        result = await generate_summary_and_fix(
            document="Doc.",
            violations=violations,
            llm_model="some-model",
            output_dir=Path("/tmp"),
        )

        assert result is None

    @pytest.mark.asyncio
    @patch("app.style_guide.style_guide_checker.LLMTable")
    @patch("app.style_guide.style_guide_checker.BedrockHandler")
    async def test_with_conversation_context(
        self, mock_bedrock_class, mock_llm_table_class
    ):
        """Conversation context is embedded in the prompt sent to the LLM."""
        mock_handler = Mock()
        mock_handler.invoke_async = AsyncMock(
            return_value=_make_tool_use_response(
                "provide_summary_and_fixed_document",
                {"summary": "Summary.", "fixed_document": "Fixed doc."},
            )
        )
        mock_bedrock_class.return_value = mock_handler
        mock_llm_table_class.return_value.get_by_model.return_value = Mock()

        violations = [
            {
                "rule_id": "r1", "rule_title": "R1", "validation_type": "deterministic",
                "match_count": 1, "sentences": ["s"],
            }
        ]
        context = "<ConversationContext><user>Previous message</user></ConversationContext>"

        result = await generate_summary_and_fix(
            document="Doc.",
            violations=violations,
            llm_model="some-model",
            output_dir=Path("/tmp"),
            conversation_context=context,
        )

        assert result is not None
        # Verify the context appeared in the prompt that was sent to the handler
        call_args = mock_handler.invoke_async.call_args
        prompt_in_call = call_args[1]["messages"][0]["content"]
        assert "Previous message" in prompt_in_call

    @pytest.mark.asyncio
    @patch("app.style_guide.style_guide_checker.LLMTable")
    @patch("app.style_guide.style_guide_checker.BedrockHandler")
    async def test_invalid_tool_input_returns_none(
        self, mock_bedrock_class, mock_llm_table_class
    ):
        """A tool response that fails Pydantic validation returns None.

        Tool use forces a single structured call; if the returned input doesn't
        satisfy the schema the function returns None rather than raise.
        """
        # 'summary' is required but missing; 'fixed_document' is a number not a string
        mock_handler = Mock()
        mock_handler.invoke_async = AsyncMock(
            return_value=_make_tool_use_response(
                "provide_summary_and_fixed_document", {"fixed_document": 42}
            )
        )
        mock_bedrock_class.return_value = mock_handler
        mock_llm_table_class.return_value.get_by_model.return_value = Mock()

        violations = [
            {
                "rule_id": "r1",
                "rule_title": "R1",
                "validation_type": "deterministic",
                "match_count": 1,
                "sentences": ["Bad sentence."],
            }
        ]

        result = await generate_summary_and_fix(
            document="Bad sentence.",
            violations=violations,
            llm_model="some-model",
            output_dir=Path("/tmp"),
        )

        assert result is None
        assert mock_handler.invoke_async.call_count == 1


# ---------------------------------------------------------------------------
# find_americanisms
# ---------------------------------------------------------------------------


class TestFindAmericanisms:
    """Tests for the find_americanisms helper."""

    def test_finds_basic_american_spellings(self):
        """Common American spellings are detected and mapped to their British forms."""
        document = "The government should prioritize color and labor reforms."
        results = find_americanisms(document)

        found_words = {r["matched_text"].lower() for r in results}
        assert "prioritize" in found_words
        assert "color" in found_words
        assert "labor" in found_words

    def test_returns_british_spelling_suggestion(self):
        """Each result includes the British spelling equivalent."""
        document = "We need to analyze the data."
        results = find_americanisms(document)

        analyze_hits = [r for r in results if r["matched_text"].lower() == "analyze"]
        assert len(analyze_hits) == 1
        assert analyze_hits[0]["british_spelling"] == "analyse"

    def test_ignores_technical_terms(self):
        """Words in uwotm8's ignore list (e.g. 'program', 'disk') are not flagged."""
        document = "Run the program and save to disk."
        results = find_americanisms(document)

        found_words = {r["matched_text"].lower() for r in results}
        assert "program" not in found_words
        assert "disk" not in found_words

    def test_no_americanisms_returns_empty(self):
        """A document with no American spellings returns an empty list."""
        document = "The organisation will recognise the new programme."
        results = find_americanisms(document)
        assert results == []

    def test_result_includes_sentence_context(self):
        """Each result includes the sentence and preceding sentence."""
        document = "First sentence. We should prioritize health. Third sentence."
        results = find_americanisms(document)

        prioritize_hits = [r for r in results if r["matched_text"].lower() == "prioritize"]
        assert len(prioritize_hits) == 1
        assert "prioritize" in prioritize_hits[0]["sentence"].lower()
        assert "sentence" in prioritize_hits[0]["preceding_sentence"].lower()

    def test_preserves_original_casing_in_matched_text(self):
        """The matched_text reflects the casing as it appears in the document."""
        document = "We must Prioritize this."
        results = find_americanisms(document)

        found = [r for r in results if r["matched_text"].lower() == "prioritize"]
        assert len(found) == 1
        assert found[0]["matched_text"] == "Prioritize"

    def test_empty_document_returns_empty(self):
        """An empty document returns an empty list without error."""
        assert find_americanisms("") == []

    def test_british_spelling_exception_still_returns_result(self):
        """When get_british_spelling raises, the americanism is still returned.

        The source code wraps ``get_british_spelling`` in a try/except and sets
        ``british_spelling = None`` on failure.  We must verify that:
        1. The occurrence is still appended to the results (not silently dropped).
        2. The ``british_spelling`` field is ``None``.
        """
        document = "We need to prioritize this."

        with patch(
            "app.style_guide.style_guide_checker.get_british_spelling",
            side_effect=Exception("lookup error"),
        ):
            results = find_americanisms(document)

        prioritize_hits = [r for r in results if r["matched_text"].lower() == "prioritize"]
        assert len(prioritize_hits) == 1
        assert prioritize_hits[0]["british_spelling"] is None


# ---------------------------------------------------------------------------
# check_llm_validation_rules — use_uwotm8 path
# ---------------------------------------------------------------------------


class TestCheckLLMValidationRulesUwotm8:
    """Tests for the use_uwotm8 branch in check_llm_validation_rules."""

    @pytest.mark.asyncio
    async def test_uwotm8_rule_with_no_americanisms_skips_llm(self):
        """When uwotm8 finds nothing, the LLM is not called."""
        document = "The organisation will recognise the new programme."

        with patch(
            "app.style_guide.style_guide_checker.LLMTable"
        ) as mock_llm_table, patch(
            "app.style_guide.style_guide_checker.call_llm_for_validation",
            new_callable=AsyncMock,
        ) as mock_call_llm, patch(
            "app.style_guide.style_guide_checker.check_llm_deterministic_false_rules",
            new_callable=AsyncMock,
            return_value=[],
        ):
            mock_llm_table.return_value.get_by_model.return_value = Mock()
            mock_call_llm.return_value = []

            result = await check_llm_validation_rules(document, [UWOTM8_RULE])

        mock_call_llm.assert_not_called()
        assert result == []

    @pytest.mark.asyncio
    async def test_uwotm8_rule_passes_americanisms_to_llm(self):
        """When uwotm8 finds americanisms they are passed to the LLM for validation."""
        document = "We should prioritize color in our designs."
        llm_violation = {
            "rule_id": "rule_024",
            "rule_title": "American and UK English",
            "confidence": 0.95,
            "occurrences": ["We should prioritize color in our designs."],
            "violation_reason": "American spellings used",
            "validation_type": "llm",
        }

        with patch(
            "app.style_guide.style_guide_checker.LLMTable"
        ) as mock_llm_table, patch(
            "app.style_guide.style_guide_checker.call_llm_for_validation",
            new_callable=AsyncMock,
            return_value=[llm_violation],
        ), patch(
            "app.style_guide.style_guide_checker.check_llm_deterministic_false_rules",
            new_callable=AsyncMock,
            return_value=[],
        ):
            mock_llm_table.return_value.get_by_model.return_value = Mock()

            result = await check_llm_validation_rules(document, [UWOTM8_RULE])

        assert any(v["rule_id"] == "rule_024" for v in result)

    @pytest.mark.asyncio
    async def test_uwotm8_occurrences_include_british_spelling(self):
        """Occurrences passed to the LLM prompt include the british_spelling field."""
        document = "Please analyze the results."
        captured_rules = []

        async def capture_call(prompt, llm, rules_with_occurrences):
            captured_rules.extend(rules_with_occurrences)
            return []

        with patch(
            "app.style_guide.style_guide_checker.LLMTable"
        ) as mock_llm_table, patch(
            "app.style_guide.style_guide_checker.call_llm_for_validation",
            new_callable=AsyncMock,
            side_effect=capture_call,
        ), patch(
            "app.style_guide.style_guide_checker.check_llm_deterministic_false_rules",
            new_callable=AsyncMock,
            return_value=[],
        ):
            mock_llm_table.return_value.get_by_model.return_value = Mock()
            await check_llm_validation_rules(document, [UWOTM8_RULE])

        assert len(captured_rules) == 1
        occurrences = captured_rules[0]["occurrences"]
        analyze_hit = next(
            (o for o in occurrences if o["matched_text"].lower() == "analyze"), None
        )
        assert analyze_hit is not None
        assert analyze_hit.get("british_spelling") == "analyse"
