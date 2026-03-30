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
from pydantic import BaseModel, ValidationError

# Only modify sys.path when run directly as a script (not when imported as a module)
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from breame.spelling import american_spelling_exists, get_british_spelling
from uwotm8.convert import CONVERSION_IGNORE_LIST

from app.bedrock import BedrockHandler, RunMode
from app.config import STYLE_GUIDE_LLM_BATCH_SIZE, STYLE_GUIDE_LLM_MODEL
from app.database.models import LLM
from app.database.table import LLMTable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
                        "rule_id": {
                            "type": "string",
                            "description": "The rule ID, e.g. rule_001"
                        },
                        "rule_name": {
                            "type": "string",
                            "description": "The name of the rule that was violated"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence score between 0.0 and 1.0"
                        },
                        "violation_reason": {
                            "type": "string",
                            "description": "Specific explanation citing the rule detail that is violated"
                        },
                        "occurrences": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exact text from the document that violates the rule"
                        }
                    },
                    "required": ["rule_id", "rule_name", "confidence", "occurrences"]
                }
            }
        },
        "required": ["violations"]
    }
}


# Tool schema for the summary-and-fix step, enforcing structured output for
# the document summary and corrected document fields.
TOOL_NAME_STYLE_GUIDE_SUMMARY_FIX = "provide_summary_and_fixed_document"

TOOL_STYLE_GUIDE_SUMMARY_FIX = {
    "name": TOOL_NAME_STYLE_GUIDE_SUMMARY_FIX,
    "description": (
        "Provide a concise summary of the style guide violations found and "
        "a corrected version of the document with all violations fixed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "A succinct 2-3 sentence summary of the main style guide "
                    "rules that were violated."
                )
            },
            "fixed_document": {
                "type": "string",
                "description": (
                    "The complete original document text with ALL violations corrected. "
                    "Preserve the original content, meaning, and structure exactly. "
                    "Make minimal changes - only fix the specific violations listed."
                )
            }
        },
        "required": ["summary", "fixed_document"]
    }
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
    with open(rules_file, 'r', encoding='utf-8') as f:
        all_rules = json.load(f)

    # Filter out redundant rules
    rules = [rule for rule in all_rules if not rule.get('redundant', False)]

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
    with open(document_path, 'r', encoding='utf-8') as f:
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
    while start > 0 and document[start - 1] not in '.!?\n':
        start -= 1

    # Skip any whitespace or newlines at the start
    while start < len(document) and document[start] in ' \n\t':
        start += 1

    # Look forwards for end of sentence
    end = position + len(text)
    while end < len(document) and document[end] not in '.!?\n':
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
    while current_start > 0 and document[current_start - 1] not in '.!?\n':
        current_start -= 1

    # Skip whitespace at the start
    while current_start < len(document) and document[current_start] in ' \n\t':
        current_start += 1

    # Now find the preceding sentence
    preceding_sentence = ""
    if current_start > 0:
        # Look backwards from the start of current sentence
        # Skip the sentence-ending punctuation and whitespace
        preceding_end = current_start - 1
        while preceding_end > 0 and document[preceding_end] in ' \n\t':
            preceding_end -= 1

        if preceding_end > 0:
            # Now find the start of the preceding sentence
            preceding_start = preceding_end
            while preceding_start > 0 and document[preceding_start - 1] not in '.!?\n':
                preceding_start -= 1

            # Skip whitespace at the start
            while preceding_start < len(document) and document[preceding_start] in ' \n\t':
                preceding_start += 1

            # Extract preceding sentence (include punctuation)
            preceding_end_with_punct = preceding_end
            while preceding_end_with_punct < len(document) and document[preceding_end_with_punct] in '.!?':
                preceding_end_with_punct += 1

            preceding_sentence = document[preceding_start:preceding_end_with_punct + 1].strip()

    return {
        'preceding_sentence': preceding_sentence,
        'current_sentence': current_sentence
    }


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


def check_case_insensitive_rules(
    document: str,
    rules: List[Dict]
) -> List[Dict]:
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
        rule for rule in rules
        if (
            rule.get('detection_strategy', {}).get('deterministic') is True
            and rule.get('detection_strategy', {}).get('case_sensitive') is False
            and rule.get('detection_strategy', {}).get('pass_to_llm', False) is False
            and not rule.get('detection_strategy', {}).get('check_sentence_length', False)
        )
    ]

    logger.debug(f"Checking {len(filtered_rules)} case-insensitive deterministic rules")

    for rule in filtered_rules:
        rule_id = rule.get('rule_id')
        rule_name = rule.get('rule')
        rule_details = rule.get('details')
        find_strings = rule.get('detection_strategy', {}).get('find', [])
        standalone_string = rule.get('detection_strategy', {}).get('standalone_string', True)

        if not find_strings:
            continue

        # Search for each string in the find list (case insensitive)
        for search_string in find_strings:
            # Use case-insensitive regex search with optional word boundaries
            if standalone_string:
                # Allow optional 's' at the end for plurals
                pattern = re.compile(r'\b' + re.escape(search_string) + r's?\b', re.IGNORECASE)
            else:
                pattern = re.compile(re.escape(search_string), re.IGNORECASE)
            matches = list(pattern.finditer(document))

            if matches:
                # Get all sentences where violations occur
                sentences = []
                for match in matches:
                    sentence = get_sentence_containing_text(
                        document,
                        match.group(),
                        match.start()
                    )
                    if sentence not in sentences:  # Avoid duplicates
                        sentences.append(sentence)

                violation = {
                    'rule_id': rule_id,
                    'rule_title': rule_name,
                    'rule_details': rule_details,
                    'rule_broken': search_string,
                    'match_count': len(matches),
                    'sentences': sentences
                }
                violations.append(violation)
                logger.debug(
                    f"Found violation: {rule_id} - '{search_string}' "
                    f"({len(matches)} occurrences)"
                )

    # Add deterministic sentence length check
    for rule in rules:
        detection = rule.get('detection_strategy', {})
        if detection.get('deterministic') and detection.get('check_sentence_length', False):
            max_words = detection.get('max_sentence_word_length', 25)
            rule_id = rule.get('rule_id')
            rule_name = rule.get('rule')
            rule_details = rule.get('details')

            # Replace line breaks with spaces for sentence splitting
            doc_for_split = re.sub(r'[\r\n]+', ' ', document)
            # Split on .!? followed by space or end of string
            sentence_pattern = re.compile(r'[^.!?]+[.!?]')
            sentences = sentence_pattern.findall(doc_for_split)
            long_sentences = []
            for sentence in sentences:
                word_count = len(re.findall(r'\b\w+\b', sentence))
                if word_count > max_words:
                    long_sentences.append(sentence.strip())

            if long_sentences:
                violation = {
                    'rule_id': rule_id,
                    'rule_title': rule_name,
                    'rule_details': rule_details,
                    'rule_broken': f"Sentence exceeds {max_words} words",
                    'match_count': len(long_sentences),
                    'sentences': long_sentences
                }
                violations.append(violation)
                logger.debug(
                    f"Found sentence length violation: {rule_id} - {len(long_sentences)} occurrences"
                )

    return violations


def check_case_sensitive_rules(
    document: str,
    rules: List[Dict]
) -> List[Dict]:
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
        rule for rule in rules
        if (
            rule.get('detection_strategy', {}).get('deterministic') is True
            and rule.get('detection_strategy', {}).get('case_sensitive') is True
            and rule.get('detection_strategy', {}).get('pass_to_llm', False) is False
        )
    ]

    logger.debug(f"Checking {len(filtered_rules)} case-sensitive deterministic rules")

    for rule in filtered_rules:
        rule_id = rule.get('rule_id')
        rule_name = rule.get('rule')
        rule_details = rule.get('details')
        find_strings = rule.get('detection_strategy', {}).get('find', [])
        standalone_string = rule.get('detection_strategy', {}).get('standalone_string', True)

        if not find_strings:
            continue

        # Search for each correct string (case insensitive) to find all occurrences
        for correct_string in find_strings:
            # Use case-insensitive search to find all instances with optional word boundaries
            if standalone_string:
                # Allow optional 's' at the end for plurals
                pattern = re.compile(r'\b' + re.escape(correct_string) + r's?\b', re.IGNORECASE)
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
                    if preceding_text and preceding_text[-1] in '.!?':
                        is_sentence_start = True

                if is_sentence_start:
                    # At sentence start, first letter should be capitalized
                    # Check if it's just the correct form with first letter capitalized
                    expected_at_start = correct_string[0].upper() + correct_string[1:]
                    if found_text == expected_at_start:
                        continue  # This is acceptable

                # If we get here, it's a violation
                violation_matches.append(match)
                sentence = get_sentence_containing_text(
                    document,
                    found_text,
                    position
                )
                if sentence not in sentences:
                    sentences.append(sentence)

            if violation_matches:
                violation = {
                    'rule_id': rule_id,
                    'rule_title': rule_name,
                    'rule_details': rule_details,
                    'rule_broken': f"Incorrect casing (expected: '{correct_string}')",
                    'correct_string': correct_string,
                    'match_count': len(violation_matches),
                    'sentences': sentences
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
    llm,
) -> List[Dict]:
    """Process a single batch of deterministic/pass_to_llm rules with LLM validation.

    Args:
        document: Full document text
        batch: Rules-with-occurrences for this batch
        batch_num: 0-based batch index (for logging)
        total_batches: Total number of batches (for logging)
        llm: LLM model instance

    Returns:
        List of violations found in this batch
    """
    logger.debug(f"Processing batch {batch_num + 1}/{total_batches} with {len(batch)} rules")
    prompt = create_llm_validation_prompt(document, batch)
    return await call_llm_for_validation(prompt, llm, batch)


async def _process_deterministic_false_batch(
    document: str,
    batch: List[Dict],
    batch_num: int,
    total_batches: int,
    llm,
) -> List[Dict]:
    """Process a single batch of deterministic=false rules with LLM validation.

    Args:
        document: Full document text
        batch: Rules for this batch
        batch_num: 0-based batch index (for logging)
        total_batches: Total number of batches (for logging)
        llm: LLM model instance

    Returns:
        List of violations found in this batch
    """
    logger.debug(
        f"Processing deterministic=false batch {batch_num + 1}/{total_batches} with {len(batch)} rules"
    )
    prompt = create_llm_prompt_for_deterministic_false(document, batch)
    return await call_llm_for_validation(prompt, llm, batch)


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
    word_pattern = re.compile(r'\b[a-zA-Z]+\b')
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
            occurrences.append({
                'matched_text': word,
                'british_spelling': british,
                'sentence': sentence_context['current_sentence'],
                'preceding_sentence': sentence_context['preceding_sentence'],
                'position': match.start()
            })

    return occurrences


async def check_llm_validation_rules(
    document: str,
    rules: List[Dict],
    llm_model: str = STYLE_GUIDE_LLM_MODEL,
    batch_size: int = STYLE_GUIDE_LLM_BATCH_SIZE
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

    Returns:
        List of violations found
    """
    violations = []

    # (1) Filter rules that need LLM validation (deterministic: true, pass_to_llm: true)
    filtered_rules = [
        rule for rule in rules
        if (
            rule.get('detection_strategy', {}).get('deterministic') is True
            and rule.get('detection_strategy', {}).get('pass_to_llm', False) is True
        )
    ]

    if filtered_rules:
        logger.debug(
            f"Checking {len(filtered_rules)} rules requiring LLM validation "
            "(deterministic: true, pass_to_llm: true)"
        )

    # First, find all rules that have occurrences in the document
    logger.debug("Finding occurrences for all LLM validation rules...")
    rules_with_occurrences = []

    for rule in filtered_rules:
        rule_id = rule.get('rule_id')
        rule_name = rule.get('rule')
        rule_details = rule.get('details')
        detection = rule.get('detection_strategy', {})
        case_sensitive = detection.get('case_sensitive', False)

        if detection.get('use_uwotm8'):
            # Use uwotm8/breame for comprehensive American English detection
            logger.debug(f"Using uwotm8 to find americanisms for {rule_id}")
            occurrences = find_americanisms(document)
        else:
            find_strings = detection.get('find', [])
            standalone_string = detection.get('standalone_string', True)

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
                    pattern = re.compile(r'\b' + re.escape(search_string) + r's?\b', re.IGNORECASE)
                else:
                    # Always case-insensitive for LLM validation rules
                    pattern = re.compile(re.escape(search_string), re.IGNORECASE)

                matches = list(pattern.finditer(document))

                for match in matches:
                    sentence_context = get_sentence_with_context(
                        document,
                        match.group(),
                        match.start()
                    )
                    occurrences.append({
                        'matched_text': match.group(),
                        'sentence': sentence_context['current_sentence'],
                        'preceding_sentence': sentence_context['preceding_sentence'],
                        'position': match.start()
                    })

        if occurrences:
            rules_with_occurrences.append({
                'rule_id': rule_id,
                'rule_title': rule_name,
                'rule_details': rule_details,
                'case_sensitive': case_sensitive,
                'occurrences': occurrences
            })

    if rules_with_occurrences:
        logger.debug(f"Found {len(rules_with_occurrences)} rules with occurrences in the document")

        # Get LLM from database
        try:
            llm = LLMTable().get_by_model(llm_model)
        except Exception as e:
            logger.error(f"Failed to get LLM model {llm_model}: {e}")
            # Don't return, continue to deterministic: false rules
        else:
            # Create batches and process all LLM calls in parallel
            batches = []
            for batch_start in range(0, len(rules_with_occurrences), batch_size):
                batch = rules_with_occurrences[batch_start:batch_start + batch_size]
                batches.append(batch)

            total_batches = len(batches)
            logger.debug(f"Processing {total_batches} batches of rules in parallel")

            # Run all batch tasks in parallel using asyncio.gather
            tasks = [
                _process_llm_validation_batch(document, batch, i, total_batches, llm)
                for i, batch in enumerate(batches)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results, filtering out exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error during LLM validation for batch {i + 1}: {result}")
                else:
                    violations.extend(result)

    # (2) Process deterministic: false rules (also in parallel)
    deterministic_false_violations = await check_llm_deterministic_false_rules(document, rules, llm_model, batch_size)
    violations.extend(deterministic_false_violations)

    return violations


async def check_llm_deterministic_false_rules(
    document: str,
    rules: List[Dict],
    llm_model: str,
    batch_size: int
) -> List[Dict]:
    """Check rules where deterministic: false by sending them to LLM."""
    violations = []

    filtered_rules_false = [
        rule for rule in rules
        if rule.get('detection_strategy', {}).get('deterministic') is False
    ]

    if filtered_rules_false:
        logger.debug(f"Checking {len(filtered_rules_false)} rules requiring LLM validation (deterministic: false)")
        logger.debug(
            "deterministic=false rules being sent to LLM: "
            + ", ".join([f"{r.get('rule_id')} ({r.get('rule')})" for r in filtered_rules_false])
        )
        try:
            llm = LLMTable().get_by_model(llm_model)
        except Exception as e:
            logger.error(f"Failed to get LLM model {llm_model}: {e}")
            return violations

        # Create batches and process all LLM calls in parallel
        batches = []
        for batch_start in range(0, len(filtered_rules_false), batch_size):
            batch = filtered_rules_false[batch_start:batch_start + batch_size]
            batches.append(batch)

        total_batches = len(batches)
        logger.debug(f"Processing {total_batches} batches of deterministic=false rules in parallel")

        # Run all batch tasks in parallel using asyncio.gather
        tasks = [
            _process_deterministic_false_batch(document, batch, i, total_batches, llm)
            for i, batch in enumerate(batches)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, filtering out exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error during LLM validation for deterministic=false batch {i + 1}: {result}")
            else:
                violations.extend(result)

    return violations


def create_llm_prompt_for_deterministic_false(document: str, rules: List[Dict]) -> str:
    """Create a focused prompt for LLM validation of deterministic=false rules.

    This prompt is designed to reduce false positives by:
    1. Understanding the specific scope and purpose of each rule
    2. Requiring violations to match the exact context described in Rule Details
    3. Explicitly warning against over-generalization
    4. Prioritizing precision over recall
    """
    prompt = """You are an expert GOV.UK style guide auditor. Your job is NOT to find violations but to find CLEAR
violations that unambiguously break the specific guidance in Rule Details.

FUNDAMENTAL APPROACH:
- Read each Rule Details carefully to understand its SPECIFIC PURPOSE and SCOPE
- Only flag violations where the document's usage clearly contradicts what Rule Details prescribe
- Rules often have narrow application domains - do NOT over-generalize the rule's scope
- If there's any reasonable interpretation where the document is correct, do NOT flag it

CRITICAL WARNINGS AGAINST FALSE POSITIVES:
1. SCOPE MISMATCH - Understand WHAT the rule actually applies to:
   - "Titles" rule applies to PAGE TITLES/HEADINGS, NOT every sentence with capital letters or dashes
   - "Summaries" rule applies to META DESCRIPTIONS and page summaries, NOT regular body text sentences
   - "Maths content" rule applies to mathematical notation, NOT to date ranges, account numbers, or ID numbers
   - "Numbers" rule applies to how numbers are written in text, NOT formatting of phone/account numbers
   - "Telephone numbers" rule applies to phone numbers ONLY, not account numbers or reference codes
   - Do NOT assume a rule name applies to anything containing a related word
2. DOCUMENT STRUCTURE - Recognize what parts of the document are what:
   - PAGE TITLES are standalone headers at the start of sections (typically short, descriptive)
   - SUMMARIES are meta descriptions (HTML meta tags) or explicit introductory summaries
     * They are typically placed at the very beginning of a document
     * They summarize the entire page content
     * Regular sentences in body text are NOT summaries, even if they're descriptive
     * A long sentence in the middle or end of a document is NOT a summary
   - BODY TEXT is the main content - most rules about titles/summaries don't apply here
   - Account numbers, sort codes, reference numbers are NOT telephone numbers
   - Date ranges like "2024-2025" are body content, not titles or math expressions
   - If text appears in the middle or end of a document, it's almost certainly NOT a title or summary
   - If something is clearly NOT in the scope defined by Rule Details, don't flag it
3. HALLUCINATING REQUIREMENTS: Do NOT add requirements not explicitly stated in Rule Details
   - Rule Details are the COMPLETE specification - nothing more is required
   - If Rule Details don't mention a constraint, that constraint does NOT exist
   - Examples in Rule Details show WHAT APPLIES - use them to understand scope

EVALUATION CHECKLIST FOR EACH RULE:
□ What is this rule specifically about? (What's its purpose/domain?)
□ What context must apply for the rule to be violated? (Where does it apply?)
□ Does the document text actually appear in that specific context?
□ Is this text clearly a case where the rule should apply?
□ Does the text explicitly violate what Rule Details prescribe?
If NO to questions 2-4, do NOT flag it.

RULES TO CHECK:
"""
    for i, rule in enumerate(rules, 1):
        prompt += f"\n{'-'*70}\nRULE {i}: {rule.get('rule_id')}\n"
        prompt += f"Name: {rule.get('rule')}\n"
        prompt += f"Details: {rule.get('details')}\n"

    prompt += f"\n{'-'*70}\nDOCUMENT TO CHECK:\n{'-'*70}\n\n"
    prompt += document
    return prompt


def create_llm_validation_prompt(document: str, rules_with_occurrences: List[Dict]) -> str:
    """Create a dynamic prompt for LLM validation.

    Args:
        document: The full document text
        rules_with_occurrences: List of rules with their found occurrences

    Returns:
        Formatted prompt string
    """
    prompt = """You are an expert GOV.UK style guide checker. Your task is to validate whether specific occurrences of
text in a document violate style guide rules.

CRITICAL INSTRUCTIONS:
1. The "Rule Details" field is your PRIMARY SOURCE OF TRUTH - read it carefully for each rule
2. The "Matched text" shows what was actually found in the document (which may have different casing)
3. For case-insensitive rules, the matched text can appear in ANY casing - this is NOT automatically a violation
4. A match is ONLY a violation if it breaks the guidance in the Rule Details given its context
5. Rule Details often contain contextual guidance (e.g., "uppercase when referring to X, lowercase otherwise")
6. Use the provided context (current sentence and preceding sentence) to understand the meaning and intent
7. If the Rule Details say "use X in context A, use Y in context B", determine which context applies
8. Just because text was matched does NOT mean it's a violation - evaluate against Rule Details

EXAMPLES OF CORRECT USAGE (NOT VIOLATIONS):
- Rule: "Access to Work - Upper case when referring to the programme, otherwise use lower case"
  - "access to work" in "people need access to work" = CORRECT (general meaning, lowercase is right)
  - "Access to Work" in "Apply for Access to Work funding" = CORRECT (the programme, uppercase is right)
  - "access to work" in "Apply for access to work funding" = VIOLATION (should be uppercase for programme)

Only include violations where the text usage clearly breaks the Rule Details guidance. If the matched text is being
used correctly according to the Rule Details, do NOT include it as a violation.

RULES TO CHECK:\n\n"""

    # Add each rule with its details
    for i, rule_info in enumerate(rules_with_occurrences, 1):
        prompt += f"\nRULE {i}:\n"
        prompt += f"Rule ID: {rule_info['rule_id']}\n"
        prompt += f"Rule Name: {rule_info['rule_title']}\n"
        prompt += f"Rule Details: {rule_info['rule_details']}\n"
        prompt += f"Case Sensitive: {rule_info['case_sensitive']}\n"
        prompt += "\nOccurrences found in document:\n"

        for j, occurrence in enumerate(rule_info['occurrences'], 1):
            matched = occurrence['matched_text']
            british = occurrence.get('british_spelling')
            suggestion = f" (British spelling: '{british}')".format() if british else ""
            prompt += f"  {j}. Matched text: '{matched}'{suggestion}\n"
            if occurrence.get('preceding_sentence'):
                prompt += f"     Preceding sentence: {occurrence['preceding_sentence']}\n"
            prompt += f"     Current sentence: {occurrence['sentence']}\n"

        prompt += "\n"

    prompt += "\n\nFULL DOCUMENT FOR CONTEXT:\n\n"
    prompt += document
    return prompt


async def call_llm_for_validation(
    prompt: str,
    llm: LLM,
    rules_with_occurrences: List[Dict]
) -> List[Dict]:
    """Call LLM to validate rule violations.

    Args:
        prompt: The formatted prompt
        llm: LLM model instance
        rules_with_occurrences: Rules being checked

    Returns:
        List of validated violations
    """
    violations = []

    handler = BedrockHandler(
        llm=llm,
        mode=RunMode.ASYNC,
        max_tokens=4096
    )

    messages = [{"role": "user", "content": prompt}]

    class LLMViolation(BaseModel):
        rule_id: Optional[str]
        rule_name: Optional[str]
        confidence: Optional[float]
        violation_reason: Optional[str]
        occurrences: List[str] = []

    class LLMValidationResponse(BaseModel):
        violations: List[LLMViolation] = []

    try:
        response = await handler.invoke_async(
            messages=messages,
            tools=[TOOL_STYLE_GUIDE_VALIDATION],
            tool_choice={"type": "tool", "name": TOOL_NAME_STYLE_GUIDE_VALIDATION},
        )

        tool_input = _extract_tool_use_input(response)

        if tool_input is None:
            logger.error("LLM did not return a tool use block for style guide validation")
            return violations

        validated = LLMValidationResponse.model_validate(tool_input)

        for violation in validated.violations:
            violations.append({
                'rule_id': violation.rule_id,
                'rule_title': violation.rule_name,
                'confidence': violation.confidence,
                'occurrences': violation.occurrences,
                'violation_reason': violation.violation_reason,
                'validation_type': 'llm'
            })

        logger.debug(f"LLM validated {len(violations)} violations from this batch")

    except ValidationError as e:
        logger.exception(f"Failed to validate LLM tool response: {e}")
        raise
    except Exception as e:
        logger.exception(f"Error calling LLM for style guide validation: {e}")
        raise

    return violations


def create_summary_and_fix_prompt(document: str, violations: List[Dict], conversation_context: str = "") -> str:
    """Create a prompt for LLM to summarize violations and fix the document.

    Args:
        document: The original document text
        violations: List of all violations found
        conversation_context: Optional conversation history for follow-up context

    Returns:
        Formatted prompt string
    """
    # For follow-ups, include conversation context in the system message
    conversation_note = ""
    if conversation_context:
        conversation_note = """

IMPORTANT CONTEXT: This is a follow-up message in an ongoing conversation.
The user is requesting modifications or clarifications based on our previous analysis.
Please keep the conversation context in mind when providing your response.

Previous conversation:
""" + conversation_context

    prompt = (
        "You are an expert Government Digital Service (GDS) style guide editor. "
        "You have been given a document and a list of style guide violations found in it."
    ) + conversation_note + """

Your task is to provide THREE outputs:

1. SUMMARY: A very succinct summary (2-3 sentences) of the main style guide rules that were violated.

2. FIXED_DOCUMENT: The complete original document text, but with ALL violations corrected
while maintaining the original content, meaning, and structure. Make minimal changes - only fix the specific violations.

VIOLATIONS FOUND:
"""
    # Add each violation with details
    for i, violation in enumerate(violations, 1):
        prompt += f"\n{'-'*60}\nVIOLATION {i}:\n"
        prompt += f"Rule ID: {violation.get('rule_id')}\n"
        prompt += f"Rule Name: {violation.get('rule_title')}\n"

        if violation.get('validation_type') == 'llm':
            prompt += "Type: LLM Validated\n"
            if violation.get('violation_reason'):
                prompt += f"Reason: {violation.get('violation_reason')}\n"
            prompt += f"Occurrences ({len(violation.get('occurrences', []))}):\n"
            for occ in violation.get('occurrences', [])[:3]:  # Show first 3
                prompt += f"  - {occ}\n"
        else:
            prompt += "Type: Deterministic\n"
            if 'rule_broken' in violation:
                prompt += f"Issue: {violation['rule_broken']}\n"
            if 'correct_string' in violation:
                prompt += f"Correct Form: {violation['correct_string']}\n"
            prompt += f"Occurrences ({violation.get('match_count', 0)}):\n"
            for sentence in violation.get('sentences', [])[:3]:  # Show first 3
                prompt += f"  - {sentence}\n"

    prompt += f"\n{'-'*60}\nORIGINAL DOCUMENT:\n{'-'*60}\n\n"
    prompt += document
    return prompt


async def generate_summary_and_fix(
    document: str,
    violations: List[Dict],
    llm_model: str,
    output_dir: Path,
    conversation_context: str = ""
) -> Optional[Dict]:
    """Call LLM to generate summary and fixed document.

    Args:
        document: The original document text
        violations: List of all violations found
        llm_model: LLM model to use
        output_dir: Directory to save raw LLM response
        conversation_context: Optional conversation history for follow-up context

    Returns:
        Dict with summary, fixed_document, and violation_list, or None if error
    """
    if not violations:
        logger.debug("No violations to summarize")
        return None

    logger.debug("Generating summary and fixed document...")

    try:
        llm = LLMTable().get_by_model(llm_model)
    except Exception as e:
        logger.error(f"Failed to get LLM model {llm_model}: {e}")
        return None

    # Create prompt with conversation context
    prompt = create_summary_and_fix_prompt(document, violations, conversation_context)

    handler = BedrockHandler(
        llm=llm,
        mode=RunMode.ASYNC,
        max_tokens=8192  # Larger for fixed document
    )

    messages = [{"role": "user", "content": prompt}]

    class SummaryFixResponse(BaseModel):
        summary: str
        fixed_document: str

    try:
        response = await handler.invoke_async(
            messages=messages,
            tools=[TOOL_STYLE_GUIDE_SUMMARY_FIX],
            tool_choice={"type": "tool", "name": TOOL_NAME_STYLE_GUIDE_SUMMARY_FIX},
        )

        tool_input = _extract_tool_use_input(response)

        if tool_input is None:
            logger.error("LLM did not return a tool use block for style guide summary/fix")
            return None

        validated = SummaryFixResponse.model_validate(tool_input)
        logger.info("Successfully generated summary and fixed document")
        return {"summary": validated.summary, "fixed_document": validated.fixed_document}

    except ValidationError as e:
        logger.error(f"Failed to validate LLM tool response for summary/fix: {e}")
    except Exception as e:
        logger.error(f"Error calling LLM for summary/fix: {e}")

    return None


async def async_main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Check a document for style guide violations'
    )
    parser.add_argument(
        '--document',
        type=str,
        help='Path to the document to check (relative to golden_dataset directory)',
        default=None
    )
    parser.add_argument(
        '--rules',
        type=str,
        help='Path to rule_mapping.json file',
        default=None
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for violations (JSON)',
        default=None
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        help='LLM model to use for validation',
        default=STYLE_GUIDE_LLM_MODEL
    )
    parser.add_argument(
        '--skip-llm',
        action='store_true',
        help='Skip LLM validation and only run deterministic checks',
        default=False
    )

    args = parser.parse_args()

    # Set up paths
    script_dir = Path(__file__).parent
    rules_file = Path(args.rules) if args.rules else script_dir / 'rule_mapping.json'

    if args.document:
        # If a specific document is provided, use it
        doc_path = Path(args.document)
        if not doc_path.is_absolute():
            # Assume it's relative to golden_dataset (now lives in scripts/style_guide/)
            doc_path = script_dir.parent.parent /'scripts'/'style_guide'/'golden_dataset'/args.document
    else:        # Default to first golden dataset document
        doc_path = script_dir.parent.parent /'scripts'/'style_guide'/'golden_dataset'/'digital_services_intro.txt'

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
        llm_violations = await check_llm_validation_rules(
            document,
            rules,
            llm_model=args.llm_model
        )
    else:
        logger.info("Skipping LLM validation (--skip-llm flag set)")

    # Combine all violations
    violations = case_insensitive_violations + case_sensitive_violations + llm_violations

    # Generate summary and fixed document
    summary_result = None
    if violations:
        summary_result = await generate_summary_and_fix(
            document,
            violations,
            llm_model=args.llm_model,
            output_dir=script_dir
        )

    # Report results
    logger.info(f"\n{'='*60}")
    logger.info(f"VIOLATION REPORT for {doc_path.name}")
    logger.info(f"{'='*60}")
    logger.info(f"Total violations found: {len(violations)}\n")

    for i, violation in enumerate(violations, 1):
        logger.info(f"\nViolation {i}:")
        logger.info(f"  Rule ID: {violation['rule_id']}")
        logger.info(f"  Rule Name: {violation['rule_title']}")
        # logger.info(f"  Rule Details: {violation['rule_details']}")

        # Handle different violation types
        if violation.get('validation_type') == 'llm':
            logger.info("  Validation Type: LLM")
            logger.info(f"  Confidence: {violation.get('confidence', 'N/A')}")
            if violation.get('violation_reason'):
                logger.info(f"  Reasoning: {violation.get('violation_reason')}")
            logger.info(f"  Occurrences: {len(violation.get('occurrences', []))}")
            logger.info("  Sentences:")
            for occurrence in violation.get('occurrences', []):
                logger.info(f"    - {occurrence}")
        else:
            logger.info(f"  String Found: '{violation.get('rule_broken', 'N/A')}'")
            logger.info(f"  Occurrences: {violation.get('match_count', 0)}")
            if 'correct_string' in violation:
                logger.info(f"  Correct Form: '{violation['correct_string']}'")
            logger.info("  Sentences:")
            for sentence in violation.get('sentences', []):
                logger.info(f"    - {sentence}")

    # Display and save summary and fixed document
    if summary_result:
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY OF VIOLATIONS")
        logger.info(f"{'='*60}")
        logger.info(f"\n{summary_result.get('summary', 'N/A')}\n")

        logger.info(f"\n{'='*60}")
        logger.info("FIXED DOCUMENT")
        logger.info(f"{'='*60}")
        logger.info(f"\n{summary_result.get('fixed_document', 'N/A')}\n")

        # Save fixed document and summary to a dedicated output directory
        # (avoids polluting golden_dataset/ with generated files that would be
        # picked up as input documents on subsequent runs)
        output_dir_cli = doc_path.parent / "output"
        output_dir_cli.mkdir(exist_ok=True)

        fixed_doc_path = output_dir_cli / f"{doc_path.stem}_FIXED{doc_path.suffix}"
        with open(fixed_doc_path, 'w', encoding='utf-8') as f:  # noqa: ASYNC230
            f.write(summary_result.get('fixed_document', ''))
        logger.info(f"\n{'='*60}")
        logger.info(f"FIXED DOCUMENT saved to: {fixed_doc_path}")

        # Save complete summary to JSON
        summary_path = output_dir_cli / f"{doc_path.stem}_SUMMARY.json"
        with open(summary_path, 'w', encoding='utf-8') as f:  # noqa: ASYNC230
            json.dump(summary_result, f, indent=2, ensure_ascii=False)
        logger.info(f"SUMMARY JSON saved to: {summary_path}")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:  # noqa: ASYNC230
            json.dump(violations, f, indent=2, ensure_ascii=False)
        logger.info(f"\nViolations saved to {output_path}")

    logger.info(f"\n{'='*60}")


def main():
    """Wrapper to run async main function."""
    asyncio.run(async_main())


if __name__ == '__main__':
    main()



