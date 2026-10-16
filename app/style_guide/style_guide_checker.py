#!/usr/bin/env python3
"""
Style Guide Checker - Deterministic and LLM-based rule violation detection.
Analyzes documents against GOV.UK style guide rules.
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from anthropic.types import ToolUseBlock
from pydantic import BaseModel

# Only modify sys.path when run directly as a script (not when imported as a module)
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from breame.spelling import american_spelling_exists, get_british_spelling
from uwotm8.convert import CONVERSION_IGNORE_LIST

from app.bedrock import BedrockHandler, RunMode
from app.bedrock.call_log import LlmCallContext
from app.config import (
    STYLE_GUIDE_LLM_BATCH_SIZE,
    STYLE_GUIDE_LLM_MODEL,
    STYLE_GUIDE_MAX_OUTPUT_TOKENS,
)
from app.database.models import LLM
from app.database.table import LLMTable
from app.style_guide.exceptions import StyleGuideCheckError
from app.style_guide.prompts import (
    create_llm_prompt_for_deterministic_false,
    create_llm_validation_prompt,
)

logger = logging.getLogger(__name__)


# Tool schema used to force structured output from the LLM for violation reporting.
# Using tool_choice forces the model to respond via the tool rather than free text,
# eliminating JSON parsing failures caused by trailing commentary.
TOOL_NAME_STYLE_GUIDE_VALIDATION = "report_style_guide_violations"

TOOL_STYLE_GUIDE_VALIDATION = {
    "name": TOOL_NAME_STYLE_GUIDE_VALIDATION,
    "description": (
        "Report all style guide violations found in the document. "
        "Use an empty violations array if no violations are found."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "violations": {
                "type": "array",
                "description": "List of style guide violations found. Empty array if none.",
                "items": {
                    "type": "object",
                    "properties": {
                        "rule_id": {"type": "string", "description": "The rule ID, e.g. rule_001"},
                        "rule_name": {"type": "string", "description": "The name of the rule that was violated"},
                        "confidence": {"type": "number", "description": "Confidence score between 0.0 and 1.0"},
                        "violation_reason": {
                            "type": "string",
                            "description": "Specific explanation citing the rule detail that is violated",
                        },
                        "occurrences": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exact text from the document that violates the rule",
                        },
                    },
                    "required": ["rule_id", "rule_name", "confidence", "occurrences"],
                },
            }
        },
        "required": ["violations"],
    },
}


def _extract_tool_use_input(response) -> Optional[dict]:
    """Extract the input dict from the first ToolUseBlock in an LLM response."""
    for block in response.content:
        if isinstance(block, ToolUseBlock):
            return block.input
    return None


def load_rule_mapping(rules_file: Path) -> List[Dict]:
    """Load the rule mapping JSON file.

    Args:
        rules_file: Path to the rule_mapping.json file

    Returns:
        List of rule dictionaries (excluding redundant rules)
    """
    logger.debug(f"Loading rules from {rules_file}")
    with open(rules_file, "r", encoding="utf-8") as f:
        all_rules = json.load(f)

    # Filter out redundant rules
    rules = [rule for rule in all_rules if not rule.get("redundant", False)]

    logger.debug(f"Loaded {len(rules)} rules ({len(all_rules) - len(rules)} redundant rules filtered out)")
    return rules


def load_document(document_path: Path) -> str:
    """Load a document as a text string.

    Args:
        document_path: Path to the document file

    Returns:
        Document content as string
    """
    logger.debug(f"Loading document from {document_path}")
    with open(document_path, "r", encoding="utf-8") as f:
        content = f.read()
    logger.debug(f"Loaded document with {len(content)} characters")
    return content


def get_sentence_containing_text(document: str, text: str, position: int) -> str:
    """Extract the full sentence containing the found text.

    Args:
        document: Full document text
        text: The text that was found
        position: Position in document where text was found

    Returns:
        The full sentence containing the text
    """
    # Find sentence boundaries (., !, ?, or start/end of document)
    # Look backwards for start of sentence
    start = position
    while start > 0 and document[start - 1] not in ".!?\n":
        start -= 1

    # Skip any whitespace or newlines at the start
    while start < len(document) and document[start] in " \n\t":
        start += 1

    # Look forwards for end of sentence
    end = position + len(text)
    while end < len(document) and document[end] not in ".!?\n":
        end += 1

    # Include the punctuation if found
    if end < len(document):
        end += 1

    sentence = document[start:end].strip()
    return sentence


def get_sentence_with_context(document: str, text: str, position: int) -> Dict[str, str]:
    """Extract the sentence containing the found text plus the preceding sentence.

    Args:
        document: Full document text
        text: The text that was found
        position: Position in document where text was found

    Returns:
        Dict with 'current_sentence' and 'preceding_sentence'
    """
    # Get the current sentence
    current_sentence = get_sentence_containing_text(document, text, position)

    # Find where the current sentence starts in the document
    current_start = position
    while current_start > 0 and document[current_start - 1] not in ".!?\n":
        current_start -= 1

    # Skip whitespace at the start
    while current_start < len(document) and document[current_start] in " \n\t":
        current_start += 1

    # Now find the preceding sentence
    preceding_sentence = ""
    if current_start > 0:
        # Look backwards from the start of current sentence
        # Skip the sentence-ending punctuation and whitespace
        preceding_end = current_start - 1
        while preceding_end > 0 and document[preceding_end] in " \n\t":
            preceding_end -= 1

        if preceding_end > 0:
            # Now find the start of the preceding sentence
            preceding_start = preceding_end
            while preceding_start > 0 and document[preceding_start - 1] not in ".!?\n":
                preceding_start -= 1

            # Skip whitespace at the start
            while preceding_start < len(document) and document[preceding_start] in " \n\t":
                preceding_start += 1

            # Extract preceding sentence (include punctuation)
            preceding_end_with_punct = preceding_end
            while preceding_end_with_punct < len(document) and document[preceding_end_with_punct] in ".!?":
                preceding_end_with_punct += 1

            preceding_sentence = document[preceding_start : preceding_end_with_punct + 1].strip()

    return {"preceding_sentence": preceding_sentence, "current_sentence": current_sentence}


def split_text_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int = 0,
) -> List[str]:
    """Split text into chunks of at most *chunk_size* characters.

    Splitting is attempted at sentence boundaries (``'.'``, ``'!'``, ``'?'``,
    or newline) so that sentences are not cut mid-way.  If no boundary is found
    in the second half of the current chunk window, a hard cut is made at
    exactly *chunk_size* characters.

    An optional character *overlap* can be specified so that each chunk after
    the first begins *overlap* characters before the previous chunk ended,
    preserving cross-boundary sentence context for LLM calls.  When
    *overlap* is 0 (the default) chunks are strictly non-overlapping and safe
    to concatenate without duplicating text.

    Args:
        text: The text to split.
        chunk_size: Maximum number of characters per chunk.
        overlap: Characters from the end of the previous chunk to include at
            the start of the next chunk (default 0).

    Returns:
        List of text chunks.  Returns ``[text]`` unchanged when
        ``len(text) <= chunk_size``.
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # When we are not yet at the very end of the text, search backwards
        # for a sentence boundary to avoid splitting mid-sentence.
        if end < text_len:
            # Search the full window so we don't miss boundaries near the
            # start of the chunk.  An infinite-loop guard below ensures we
            # always make forward progress even if the boundary is at `start`.
            boundary = end  # fallback: hard cut
            for i in range(end - 1, start - 1, -1):
                if text[i] in ".!?\n":
                    boundary = i + 1
                    break
            end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance, stepping back by *overlap* to give the next chunk context.
        next_start = end - overlap if overlap > 0 else end
        # Guard against infinite loop when no forward progress is possible.
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def check_case_insensitive_rules(document: str, rules: List[Dict]) -> List[Dict]:
    """Check for violations of case-insensitive deterministic rules.

    Processes rules where:
    - deterministic = true
    - case_sensitive = false
    - pass_to_llm = false (or not present)
    - check_sentence_length = false or not present

    Note: Rules with case_sensitive=false AND pass_to_llm=true are intentionally
    excluded here and handled instead by check_llm_validation_rules(), which
    performs regex pre-filtering before passing candidates to the LLM for
    context-aware judgement.

    Args:
        document: The document text to analyze
        rules: List of all rules from rule_mapping.json

    Returns:
        List of violations found
    """
    violations = []

    # Filter rules to process (excluding sentence length rule)
    filtered_rules = [
        rule
        for rule in rules
        if (
            rule.get("detection_strategy", {}).get("deterministic") is True
            and rule.get("detection_strategy", {}).get("case_sensitive") is False
            and rule.get("detection_strategy", {}).get("pass_to_llm", False) is False
            and not rule.get("detection_strategy", {}).get("check_sentence_length", False)
        )
    ]

    logger.debug(f"Checking {len(filtered_rules)} case-insensitive deterministic rules")

    for rule in filtered_rules:
        rule_id = rule.get("rule_id")
        rule_name = rule.get("rule")
        rule_details = rule.get("details")
        find_strings = rule.get("detection_strategy", {}).get("find", [])
        standalone_string = rule.get("detection_strategy", {}).get("standalone_string", True)

        if not find_strings:
            continue

        # Search for each string in the find list (case insensitive)
        for search_string in find_strings:
            # Use case-insensitive regex search with optional word boundaries
            if standalone_string:
                # Allow optional 's' at the end for plurals
                pattern = re.compile(r"\b" + re.escape(search_string) + r"s?\b", re.IGNORECASE)
            else:
                pattern = re.compile(re.escape(search_string), re.IGNORECASE)
            matches = list(pattern.finditer(document))

            if matches:
                # Get all sentences where violations occur
                sentences = []
                for match in matches:
                    sentence = get_sentence_containing_text(document, match.group(), match.start())
                    if sentence not in sentences:  # Avoid duplicates
                        sentences.append(sentence)

                violation = {
                    "rule_id": rule_id,
                    "rule_title": rule_name,
                    "rule_details": rule_details,
                    "rule_broken": search_string,
                    "match_count": len(matches),
                    "sentences": sentences,
                }
                violations.append(violation)
                logger.debug(f"Found violation: {rule_id} - '{search_string}' ({len(matches)} occurrences)")

    # Add deterministic sentence length check
    for rule in rules:
        detection = rule.get("detection_strategy", {})
        if detection.get("deterministic") and detection.get("check_sentence_length", False):
            max_words = detection.get("max_sentence_word_length", 25)
            rule_id = rule.get("rule_id")
            rule_name = rule.get("rule")
            rule_details = rule.get("details")

            # Replace line breaks with spaces for sentence splitting
            doc_for_split = re.sub(r"[\r\n]+", " ", document)
            # Split on .!? followed by space or end of string
            sentence_pattern = re.compile(r"[^.!?]+[.!?]")
            sentences = sentence_pattern.findall(doc_for_split)
            long_sentences = []
            for sentence in sentences:
                word_count = len(re.findall(r"\b\w+\b", sentence))
                if word_count > max_words:
                    long_sentences.append(sentence.strip())

            if long_sentences:
                violation = {
                    "rule_id": rule_id,
                    "rule_title": rule_name,
                    "rule_details": rule_details,
                    "rule_broken": f"Sentence exceeds {max_words} words",
                    "match_count": len(long_sentences),
                    "sentences": long_sentences,
                }
                violations.append(violation)
                logger.debug(f"Found sentence length violation: {rule_id} - {len(long_sentences)} occurrences")

    return violations


def check_case_sensitive_rules(document: str, rules: List[Dict]) -> List[Dict]:
    """Check for violations of case-sensitive deterministic rules.

    Processes rules where:
    - deterministic = true
    - case_sensitive = true
    - pass_to_llm = false (or not present)

    For these rules, the 'find' array contains strings in their CORRECT form.
    If the string appears with different casing (and not at sentence start), it's a violation.

    Args:
        document: The document text to analyze
        rules: List of all rules from rule_mapping.json

    Returns:
        List of violations found
    """
    violations = []

    # Filter rules to process
    filtered_rules = [
        rule
        for rule in rules
        if (
            rule.get("detection_strategy", {}).get("deterministic") is True
            and rule.get("detection_strategy", {}).get("case_sensitive") is True
            and rule.get("detection_strategy", {}).get("pass_to_llm", False) is False
        )
    ]

    logger.debug(f"Checking {len(filtered_rules)} case-sensitive deterministic rules")

    for rule in filtered_rules:
        rule_id = rule.get("rule_id")
        rule_name = rule.get("rule")
        rule_details = rule.get("details")
        find_strings = rule.get("detection_strategy", {}).get("find", [])
        standalone_string = rule.get("detection_strategy", {}).get("standalone_string", True)

        if not find_strings:
            continue

        # Search for each correct string (case insensitive) to find all occurrences
        for correct_string in find_strings:
            # Use case-insensitive search to find all instances with optional word boundaries
            if standalone_string:
                # Allow optional 's' at the end for plurals
                pattern = re.compile(r"\b" + re.escape(correct_string) + r"s?\b", re.IGNORECASE)
            else:
                pattern = re.compile(re.escape(correct_string), re.IGNORECASE)
            matches = list(pattern.finditer(document))

            if not matches:
                continue

            # Check each match to see if it's a violation
            violation_matches = []
            sentences = []

            for match in matches:
                found_text = match.group()
                position = match.start()

                # Check if the casing matches exactly - if so, it's correct
                if found_text == correct_string:
                    continue

                # Check if this is at the start of a sentence
                # If so, check if only the first letter is capitalized
                is_sentence_start = False
                if position == 0:
                    is_sentence_start = True
                else:
                    # Look backwards to find if we're at the start of a sentence
                    # Check for preceding sentence-ending punctuation (., !, ?) followed by whitespace
                    preceding_text = document[:position].rstrip()
                    if preceding_text and preceding_text[-1] in ".!?":
                        is_sentence_start = True

                if is_sentence_start:
                    # At sentence start, first letter should be capitalized
                    # Check if it's just the correct form with first letter capitalized
                    expected_at_start = correct_string[0].upper() + correct_string[1:]
                    if found_text == expected_at_start:
                        continue  # This is acceptable

                # If we get here, it's a violation
                violation_matches.append(match)
                sentence = get_sentence_containing_text(document, found_text, position)
                if sentence not in sentences:
                    sentences.append(sentence)

            if violation_matches:
                violation = {
                    "rule_id": rule_id,
                    "rule_title": rule_name,
                    "rule_details": rule_details,
                    "rule_broken": f"Incorrect casing (expected: '{correct_string}')",
                    "correct_string": correct_string,
                    "match_count": len(violation_matches),
                    "sentences": sentences,
                }
                violations.append(violation)
                logger.debug(
                    f"Found violation: {rule_id} - incorrect casing of '{correct_string}' "
                    f"({len(violation_matches)} occurrences)"
                )

    return violations


async def _process_llm_validation_batch(
    document: str,
    batch: List[Dict],
    batch_num: int,
    total_batches: int,
    llm: LLM,
    checker_context: str,
    context: LlmCallContext,
) -> List[Dict]:
    """Process a single batch of deterministic/pass_to_llm rules with LLM validation.

    Args:
        document: Full document text
        batch: Rules-with-occurrences for this batch
        batch_num: 0-based batch index (for logging)
        total_batches: Total number of batches (for logging)
        llm: LLM model instance
        checker_context: Context from the assistant (see format_checker_context), or "".
        context: What the call is for, recorded on its llm_internal_response row

    Returns:
        List of violations found in this batch
    """
    logger.debug(f"Processing batch {batch_num + 1}/{total_batches} with {len(batch)} rules")
    prompt = create_llm_validation_prompt(document, batch, checker_context)
    return await call_llm_for_validation(prompt, llm, batch, context)


async def _process_deterministic_false_batch(
    document: str,
    batch: List[Dict],
    batch_num: int,
    total_batches: int,
    llm: LLM,
    checker_context: str,
    context: LlmCallContext,
) -> List[Dict]:
    """Process a single batch of deterministic=false rules with LLM validation.

    Args:
        document: Full document text
        batch: Rules for this batch
        batch_num: 0-based batch index (for logging)
        total_batches: Total number of batches (for logging)
        llm: LLM model instance
        checker_context: Context from the assistant (see format_checker_context), or "".
        context: What the call is for, recorded on its llm_internal_response row

    Returns:
        List of violations found in this batch
    """
    logger.debug(f"Processing deterministic=false batch {batch_num + 1}/{total_batches} with {len(batch)} rules")
    prompt = create_llm_prompt_for_deterministic_false(document, batch, checker_context)
    return await call_llm_for_validation(prompt, llm, batch, context)


async def _gather_batches(tasks: List, label: str) -> List[Dict]:
    """Run validation batches in parallel and combine their violations.

    Every batch is awaited before anything is raised, so no call is left running unobserved. If any
    batch failed, the first error is raised: a check that couldn't apply all its rules must not be
    reported as complete.

    Args:
        tasks: Coroutines, one per batch, each returning a list of violations.
        label: Which kind of batch, for logging.

    Returns:
        The violations from every batch.

    Raises:
        Exception: The first batch's error, if any batch failed.
    """
    results = await asyncio.gather(*tasks, return_exceptions=True)
    violations: List[Dict] = []
    errors: List[BaseException] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.error(f"Style guide {label} batch {i + 1}/{len(results)} failed: {result!r}")
            errors.append(result)
        else:
            violations.extend(result)
    if errors:
        raise errors[0]
    return violations


def find_americanisms(document: str) -> List[Dict]:
    """Find American English spellings in the document using breame/uwotm8.

    Uses breame's spelling database (1700+ word pairs) together with uwotm8's
    technical ignore list (e.g. 'program', 'disk', 'analog') to detect American
    spellings while skipping common false positives.

    Args:
        document: The document text to analyse

    Returns:
        List of occurrence dicts, each containing:
        - matched_text: the American word as it appears in the document
        - british_spelling: the British English equivalent
        - sentence: the sentence containing the word
        - preceding_sentence: the sentence before it
        - position: character offset in the document
    """
    word_pattern = re.compile(r"\b[a-zA-Z]+\b")
    occurrences = []

    for match in word_pattern.finditer(document):
        word = match.group()
        word_lower = word.lower()

        # Skip technical terms that uwotm8 deliberately leaves unconverted
        if word_lower in CONVERSION_IGNORE_LIST:
            continue

        if american_spelling_exists(word_lower):
            try:
                british = get_british_spelling(word_lower)
            except Exception:
                british = None

            sentence_context = get_sentence_with_context(document, word, match.start())
            occurrences.append(
                {
                    "matched_text": word,
                    "british_spelling": british,
                    "sentence": sentence_context["current_sentence"],
                    "preceding_sentence": sentence_context["preceding_sentence"],
                    "position": match.start(),
                }
            )

    return occurrences


async def check_llm_validation_rules(
    document: str,
    rules: List[Dict],
    llm_model: str = STYLE_GUIDE_LLM_MODEL,
    batch_size: int = STYLE_GUIDE_LLM_BATCH_SIZE,
    checker_context: str = "",
    context: LlmCallContext | None = None,
) -> List[Dict]:
    """Check rules that require LLM validation for context.

    Processes rules where:
    - deterministic = true
    - pass_to_llm = true
    - case_sensitive = true or false

    Args:
        document: The document text to analyze
        rules: List of all rules from rule_mapping.json
        llm_model: LLM model to use for validation
        batch_size: Number of rules to process in each LLM call
        checker_context: Context from the assistant (see format_checker_context), or "".
        context: What the calls are for, recorded on their llm_internal_response rows. Defaults to
            purpose "style_guide_validation" with no chat.

    Returns:
        List of violations found

    Raises:
        Exception: If the model can't be loaded or any validation call fails.
    """
    context = context or LlmCallContext(purpose="style_guide_validation")
    violations = []

    # (1) Filter rules that need LLM validation (deterministic: true, pass_to_llm: true)
    filtered_rules = [
        rule
        for rule in rules
        if (
            rule.get("detection_strategy", {}).get("deterministic") is True
            and rule.get("detection_strategy", {}).get("pass_to_llm", False) is True
        )
    ]

    if filtered_rules:
        logger.debug(
            f"Checking {len(filtered_rules)} rules requiring LLM validation (deterministic: true, pass_to_llm: true)"
        )

    # First, find all rules that have occurrences in the document
    logger.debug("Finding occurrences for all LLM validation rules...")
    rules_with_occurrences = []

    for rule in filtered_rules:
        rule_id = rule.get("rule_id")
        rule_name = rule.get("rule")
        rule_details = rule.get("details")
        detection = rule.get("detection_strategy", {})
        case_sensitive = detection.get("case_sensitive", False)

        if detection.get("use_uwotm8"):
            # Use uwotm8/breame for comprehensive American English detection
            logger.debug(f"Using uwotm8 to find americanisms for {rule_id}")
            occurrences = find_americanisms(document)
        else:
            find_strings = detection.get("find", [])
            standalone_string = detection.get("standalone_string", True)

            if not find_strings:
                continue

            # Find all occurrences of the search strings
            # Always search case-insensitively to find all potential violations
            # The LLM will validate if the casing is correct based on context
            occurrences = []
            for search_string in find_strings:
                # If standalone_string is true, use word boundaries
                if standalone_string:
                    # \b for word boundary - matches position between word and non-word character
                    # Allow optional 's' at the end for plurals
                    pattern = re.compile(r"\b" + re.escape(search_string) + r"s?\b", re.IGNORECASE)
                else:
                    # Always case-insensitive for LLM validation rules
                    pattern = re.compile(re.escape(search_string), re.IGNORECASE)

                matches = list(pattern.finditer(document))

                for match in matches:
                    sentence_context = get_sentence_with_context(document, match.group(), match.start())
                    occurrences.append(
                        {
                            "matched_text": match.group(),
                            "sentence": sentence_context["current_sentence"],
                            "preceding_sentence": sentence_context["preceding_sentence"],
                            "position": match.start(),
                        }
                    )

        if occurrences:
            rules_with_occurrences.append(
                {
                    "rule_id": rule_id,
                    "rule_title": rule_name,
                    "rule_details": rule_details,
                    "case_sensitive": case_sensitive,
                    "occurrences": occurrences,
                }
            )

    if rules_with_occurrences:
        logger.debug(f"Found {len(rules_with_occurrences)} rules with occurrences in the document")
        llm = LLMTable().get_by_model(llm_model)
        batches = [
            rules_with_occurrences[batch_start : batch_start + batch_size]
            for batch_start in range(0, len(rules_with_occurrences), batch_size)
        ]
        logger.debug(f"Processing {len(batches)} batches of rules in parallel")
        tasks = [
            _process_llm_validation_batch(document, batch, i, len(batches), llm, checker_context, context)
            for i, batch in enumerate(batches)
        ]
        violations.extend(await _gather_batches(tasks, "LLM validation"))

    # (2) Process deterministic: false rules (also in parallel)
    deterministic_false_violations = await check_llm_deterministic_false_rules(
        document, rules, llm_model, batch_size, checker_context=checker_context, context=context
    )
    violations.extend(deterministic_false_violations)

    return violations


async def check_llm_deterministic_false_rules(
    document: str,
    rules: List[Dict],
    llm_model: str,
    batch_size: int,
    checker_context: str = "",
    context: LlmCallContext | None = None,
) -> List[Dict]:
    """Check rules where deterministic: false by sending them to LLM.

    Raises:
        Exception: If the model can't be loaded or any validation call fails.
    """
    context = context or LlmCallContext(purpose="style_guide_validation")
    filtered_rules_false = [rule for rule in rules if rule.get("detection_strategy", {}).get("deterministic") is False]
    if not filtered_rules_false:
        return []

    logger.debug(f"Checking {len(filtered_rules_false)} rules requiring LLM validation (deterministic: false)")
    llm = LLMTable().get_by_model(llm_model)
    batches = [
        filtered_rules_false[batch_start : batch_start + batch_size]
        for batch_start in range(0, len(filtered_rules_false), batch_size)
    ]
    logger.debug(f"Processing {len(batches)} batches of deterministic=false rules in parallel")
    tasks = [
        _process_deterministic_false_batch(document, batch, i, len(batches), llm, checker_context, context)
        for i, batch in enumerate(batches)
    ]
    return await _gather_batches(tasks, "deterministic=false validation")


async def call_llm_for_validation(
    prompt: str, llm: LLM, rules_with_occurrences: List[Dict], context: LlmCallContext | None = None
) -> List[Dict]:
    """Call LLM to validate rule violations.

    Args:
        prompt: The formatted prompt
        llm: LLM model instance
        rules_with_occurrences: Rules being checked
        context: What the call is for, recorded on its llm_internal_response row. Defaults to
            purpose "style_guide_validation" with no chat.

    Returns:
        List of validated violations

    Raises:
        StyleGuideCheckError: The model didn't answer with the report tool.
        ValidationError: The tool input didn't match the expected shape.
    """

    class LLMViolation(BaseModel):
        rule_id: Optional[str]
        rule_name: Optional[str]
        confidence: Optional[float]
        violation_reason: Optional[str]
        occurrences: List[str] = []

    class LLMValidationResponse(BaseModel):
        violations: List[LLMViolation] = []

    handler = BedrockHandler(llm=llm, mode=RunMode.ASYNC, max_tokens=STYLE_GUIDE_MAX_OUTPUT_TOKENS)
    response = await handler.invoke_async(
        messages=[{"role": "user", "content": prompt}],
        context=context or LlmCallContext(purpose="style_guide_validation"),
        tools=[TOOL_STYLE_GUIDE_VALIDATION],
        tool_choice={"type": "tool", "name": TOOL_NAME_STYLE_GUIDE_VALIDATION},
    )

    tool_input = _extract_tool_use_input(response)
    if tool_input is None:
        raise StyleGuideCheckError("The model did not return a tool use block for style guide validation")

    validated = LLMValidationResponse.model_validate(tool_input)
    violations = [
        {
            "rule_id": violation.rule_id,
            "rule_title": violation.rule_name,
            "confidence": violation.confidence,
            "occurrences": violation.occurrences,
            "violation_reason": violation.violation_reason,
            "validation_type": "llm",
        }
        for violation in validated.violations
    ]
    logger.debug(f"LLM validated {len(violations)} violations from this batch")
    return violations


async def async_main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Check a document for style guide violations")
    parser.add_argument(
        "--document",
        type=str,
        help="Path to the document to check (relative to golden_dataset directory)",
        default=None,
    )
    parser.add_argument("--rules", type=str, help="Path to rule_mapping.json file", default=None)
    parser.add_argument("--output", type=str, help="Output file for violations (JSON)", default=None)
    parser.add_argument("--llm-model", type=str, help="LLM model to use for validation", default=STYLE_GUIDE_LLM_MODEL)
    parser.add_argument(
        "--skip-llm", action="store_true", help="Skip LLM validation and only run deterministic checks", default=False
    )

    args = parser.parse_args()

    # Set up paths
    script_dir = Path(__file__).parent
    rules_file = Path(args.rules) if args.rules else script_dir / "rule_mapping.json"

    if args.document:
        # If a specific document is provided, use it
        doc_path = Path(args.document)
        if not doc_path.is_absolute():
            # Assume it's relative to golden_dataset (now lives in scripts/style_guide/)
            doc_path = script_dir.parent.parent / "scripts" / "style_guide" / "golden_dataset" / args.document
    else:  # Default to first golden dataset document
        doc_path = (
            script_dir.parent.parent / "scripts" / "style_guide" / "golden_dataset" / "digital_services_intro.txt"
        )

    # Load rules and document
    try:
        rules = load_rule_mapping(rules_file)
        document = load_document(doc_path)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON: {e}")
        return

    # Check for violations
    logger.info("Starting violation detection...")
    case_insensitive_violations = check_case_insensitive_rules(document, rules)
    case_sensitive_violations = check_case_sensitive_rules(document, rules)

    # Check LLM validation rules if not skipped
    llm_violations = []
    if not args.skip_llm:
        logger.info("Starting LLM validation...")
        llm_violations = await check_llm_validation_rules(document, rules, llm_model=args.llm_model)
    else:
        logger.info("Skipping LLM validation (--skip-llm flag set)")

    # Combine all violations
    violations = case_insensitive_violations + case_sensitive_violations + llm_violations

    # Report results
    logger.info(f"\n{'=' * 60}")
    logger.info(f"VIOLATION REPORT for {doc_path.name}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total violations found: {len(violations)}\n")

    for i, violation in enumerate(violations, 1):
        logger.info(f"\nViolation {i}:")
        logger.info(f"  Rule ID: {violation['rule_id']}")
        logger.info(f"  Rule Name: {violation['rule_title']}")
        # logger.info(f"  Rule Details: {violation['rule_details']}")

        # Handle different violation types
        if violation.get("validation_type") == "llm":
            logger.info("  Validation Type: LLM")
            logger.info(f"  Confidence: {violation.get('confidence', 'N/A')}")
            if violation.get("violation_reason"):
                logger.info(f"  Reasoning: {violation.get('violation_reason')}")
            logger.info(f"  Occurrences: {len(violation.get('occurrences', []))}")
            logger.info("  Sentences:")
            for occurrence in violation.get("occurrences", []):
                logger.info(f"    - {occurrence}")
        else:
            logger.info(f"  String Found: '{violation.get('rule_broken', 'N/A')}'")
            logger.info(f"  Occurrences: {violation.get('match_count', 0)}")
            if "correct_string" in violation:
                logger.info(f"  Correct Form: '{violation['correct_string']}'")
            logger.info("  Sentences:")
            for sentence in violation.get("sentences", []):
                logger.info(f"    - {sentence}")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:  # noqa: ASYNC230
            json.dump(violations, f, indent=2, ensure_ascii=False)
        logger.info(f"\nViolations saved to {output_path}")

    logger.info(f"\n{'=' * 60}")


def main():
    """Wrapper to run async main function."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
