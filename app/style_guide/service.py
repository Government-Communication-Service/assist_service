"""Style guide checks for the chat agent loop's check_style_guide tool."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.bedrock.call_log import LlmCallContext
from app.config import (
    STYLE_GUIDE_LLM_MODEL,
    STYLE_GUIDE_MAX_CHUNK_CHARS,
    STYLE_GUIDE_MAX_DOCUMENT_CHARS,
)
from app.database.models import Document, DocumentChunk, DocumentUserMapping
from app.database.table import async_db_session
from app.style_guide.prompts import format_checker_context
from app.style_guide.style_guide_checker import (
    check_case_insensitive_rules,
    check_case_sensitive_rules,
    check_llm_validation_rules,
    load_rule_mapping,
    split_text_into_chunks,
)

logger = logging.getLogger(__name__)

# Load rules once at module import
SCRIPT_DIR = Path(__file__).parent
RULES_FILE = SCRIPT_DIR / "rule_mapping.json"
RULES = load_rule_mapping(RULES_FILE)
RULE_GUIDANCE = {rule["rule_id"]: rule.get("details") for rule in RULES}

STYLE_GUIDE_TOOL_TAG = "style-guide-check"

# How many example sentences to list per issue, so a long document doesn't flood the chat model's context.
MAX_EXAMPLES_PER_ISSUE = 5
# Hard cap on the issues returned to the chat model, so a long document can't flood its context.
MAX_ISSUES = 25


def _tagged(body: str) -> str:
    return f"<{STYLE_GUIDE_TOOL_TAG}>\n{body}\n</{STYLE_GUIDE_TOOL_TAG}>"


async def _get_document_text(
    document_uuid: str,
    user_id: int,
    db_session: AsyncSession,
) -> Tuple[str, str] | None:
    """Retrieve the full text of one of the user's uploaded documents by getting all its chunks.

    Args:
        document_uuid: The document to read.
        user_id: The user's ID (integer) for access verification
        db_session: Session to query in

    Returns:
        The document's name and text, or None if the user can't access it.
    """
    document = await db_session.scalar(
        select(Document)
        .join(DocumentUserMapping, DocumentUserMapping.document_id == Document.id)
        .where(Document.uuid == document_uuid)
        .where(DocumentUserMapping.user_id == user_id)
        .where(DocumentUserMapping.deleted_at.is_(None))
        .where(Document.deleted_at.is_(None))
    )
    if document is None:
        logger.warning(f"User {user_id} does not have access to document {document_uuid}")
        return None

    chunks = (
        await db_session.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document.id)
            .where(DocumentChunk.deleted_at.is_(None))
            .order_by(DocumentChunk.id)
        )
    ).all()
    logger.info(f"Retrieved {len(chunks)} chunks from document '{document.name}'")
    return document.name, "".join(f"{chunk.content}\n" for chunk in chunks)


async def _run_llm_validation_on_chunks(
    chunks: List[str],
    llm_model: str,
    checker_context: str = "",
    context: LlmCallContext | None = None,
) -> List[Dict]:
    """Run LLM validation rules across multiple document chunks.

    A rule triggered in more than one chunk is reported once, with the occurrences from every
    chunk merged into it.

    Args:
        chunks: Text chunks to validate.
        llm_model: LLM model identifier.
        checker_context: Context from the assistant (see format_checker_context), or "".
        context: What the calls are for, recorded on their llm_internal_response rows.

    Returns:
        List of LLM violations found across all chunks, one per rule.
    """
    by_rule_id: Dict[str, Dict] = {}
    llm_violations: List[Dict] = []

    for idx, chunk in enumerate(chunks):
        logger.info(f"Running LLM validation on chunk {idx + 1}/{len(chunks)}")
        chunk_violations = await check_llm_validation_rules(
            chunk, RULES, llm_model=llm_model, checker_context=checker_context, context=context
        )
        for violation in chunk_violations:
            rule_id = violation.get("rule_id")
            existing = by_rule_id.get(rule_id) if rule_id else None
            if existing is None:
                violation = {**violation, "occurrences": list(violation.get("occurrences") or [])}
                llm_violations.append(violation)
                if rule_id:
                    by_rule_id[rule_id] = violation
            else:
                existing["occurrences"].extend(
                    occurrence
                    for occurrence in violation.get("occurrences") or []
                    if occurrence not in existing["occurrences"]
                )

    return llm_violations


def describe_violation(violation: Dict) -> Dict:
    """Put a violation from any of the checks into one shape.

    Pattern and LLM checks report violations with different keys; this gives both the same fields,
    with the rule's guidance looked up when the violation doesn't carry it.

    Args:
        violation: A violation from check_case_*_rules or check_llm_validation_rules.

    Returns:
        Dict with rule_id, rule_title, guidance, detected_by ("pattern match" or "model review"),
        problem, correct_form, reason, count and examples (the sentences or text where it occurs).
    """
    rule_id = violation.get("rule_id")
    is_llm = violation.get("validation_type") == "llm"
    examples = [e for e in violation.get("occurrences" if is_llm else "sentences") or [] if e]
    return {
        "rule_id": rule_id,
        "rule_title": violation.get("rule_title") or rule_id,
        "guidance": violation.get("rule_details") or RULE_GUIDANCE.get(rule_id),
        "detected_by": "model review" if is_llm else "pattern match",
        "problem": None if is_llm else violation.get("rule_broken"),
        "correct_form": None if is_llm else violation.get("correct_string"),
        "reason": violation.get("violation_reason") if is_llm else None,
        "count": len(examples) if is_llm else (violation.get("match_count") or len(examples)),
        "examples": examples,
    }


def _format_issue(violation: Dict, include_guidance: bool = True) -> str:
    """Format one violation as an <issue> element for the chat model.

    Args:
        violation: The violation from the checker.
        include_guidance: False when an earlier issue already gave this rule's guidance.
    """
    issue = describe_violation(violation)
    parts = ["<issue>", f"<rule>{issue['rule_title']}</rule>"]
    if include_guidance and issue["guidance"]:
        parts.append(f"<guidance>{issue['guidance']}</guidance>")
    parts.append(f"<detected-by>{issue['detected_by']}</detected-by>")
    if issue["reason"]:
        parts.append(f"<reason>{issue['reason']}</reason>")
    if issue["problem"]:
        parts.append(f"<problem>{issue['problem']}</problem>")
    if issue["correct_form"]:
        parts.append(f"<correct-form>{issue['correct_form']}</correct-form>")
    if issue["count"]:
        parts.append(f"<count>{issue['count']}</count>")
    examples = issue["examples"]
    parts.extend(f"<found-in>{example}</found-in>" for example in examples[:MAX_EXAMPLES_PER_ISSUE])
    if len(examples) > MAX_EXAMPLES_PER_ISSUE:
        parts.append(f"<more>{len(examples) - MAX_EXAMPLES_PER_ISSUE} more not listed</more>")
    parts.append("</issue>")
    return "\n".join(parts)


async def analyse_content_for_style_guide(
    content_to_check: str,
    checked: str,
    checker_context: str = "",
    context: LlmCallContext | None = None,
    llm_model: str = STYLE_GUIDE_LLM_MODEL,
) -> Tuple[str, bool]:
    """Run the deterministic and LLM style guide checks on some content and list what they found.

    The result is data for the chat model: what was checked and each issue found, with no corrected
    version and no instructions. How to present it is up to the chat model, guided by the tool
    description.

    Args:
        content_to_check: Text to check.
        checked: What was checked, in words, e.g. "the user's latest message".
        checker_context: Context from the assistant (see format_checker_context), or "".
        context: What the validation calls are for, recorded on their llm_internal_response rows.
        llm_model: LLM model for the validation rules.

    Returns:
        The <style-guide-check> segment, and whether the check couldn't be run.

    Raises:
        Exception: If a validation call fails, so the loop reports the tool as failed rather than
            the check as complete.
    """
    if len(content_to_check) > STYLE_GUIDE_MAX_DOCUMENT_CHARS:
        char_count = len(content_to_check)
        logger.warning(
            f"Document too large for style guide check: {char_count:,} chars "
            f"(limit: {STYLE_GUIDE_MAX_DOCUMENT_CHARS:,})"
        )
        return _tagged(
            f"<checked>{checked}</checked>\n"
            f"<error>The text is too long to check in one go ({char_count:,} characters). It can be checked "
            f"in sections of under {STYLE_GUIDE_MAX_DOCUMENT_CHARS:,} characters.</error>"
        ), True

    if not content_to_check.strip():
        logger.warning("No content to check against style guide")
        return _tagged(f"<checked>{checked}</checked>\n<error>There was no text to check.</error>"), True

    logger.debug(f"Running style guide analysis on {len(content_to_check)} chars of content")

    # Deterministic checks are regex, so they run on the full text whatever its size.
    case_insensitive_violations = check_case_insensitive_rules(content_to_check, RULES)
    case_sensitive_violations = check_case_sensitive_rules(content_to_check, RULES)

    # LLM validation runs per chunk on long text, to keep each prompt a manageable size.
    if len(content_to_check) > STYLE_GUIDE_MAX_CHUNK_CHARS:
        chunks = split_text_into_chunks(content_to_check, STYLE_GUIDE_MAX_CHUNK_CHARS)
        logger.debug(f"Document ({len(content_to_check):,} chars) split into {len(chunks)} chunk(s) for LLM processing")
        llm_violations = await _run_llm_validation_on_chunks(chunks, llm_model, checker_context, context)
    else:
        llm_violations = await check_llm_validation_rules(
            content_to_check, RULES, llm_model=llm_model, checker_context=checker_context, context=context
        )

    all_violations = case_insensitive_violations + case_sensitive_violations + llm_violations
    logger.info(f"Style guide check complete: {len(all_violations)} total violations found")

    parts = [f"<checked>{checked}</checked>"]
    if not all_violations:
        parts.append("<result>No issues found against the GOV.UK style guide rules this check covers.</result>")
    else:
        issue_word = "issue" if len(all_violations) == 1 else "issues"
        parts.append(f"<result>{len(all_violations)} {issue_word} found.</result>")
        listed = all_violations
        if len(all_violations) > MAX_ISSUES:
            # Keep the most frequent issues; sorted is stable, so ties keep their order.
            listed = sorted(all_violations, key=lambda v: describe_violation(v)["count"], reverse=True)[:MAX_ISSUES]
            parts.append(
                f"<truncated>Only the {MAX_ISSUES} most frequent issues are listed. The other "
                f"{len(all_violations) - MAX_ISSUES} were left out; checking a shorter section will show them."
                "</truncated>"
            )
        # A rule with several search terms, such as "Words to avoid", gives an issue per term matched,
        # so its guidance goes on the first of them only.
        seen_rules = set()
        for violation in listed:
            rule_id = violation.get("rule_id")
            parts.append(_format_issue(violation, include_guidance=rule_id not in seen_rules))
            seen_rules.add(rule_id)
    return _tagged("\n".join(parts)), False


async def check_style_guide_for_tool(
    text: str | None,
    document_id: str | None,
    checker_context: str | None,
    document_uuids: List[str],
    user_id: int,
    context: LlmCallContext,
) -> Tuple[str, bool]:
    """Run a style guide check chosen by the chat model, on either some text or one attached document.

    Writes nothing, so it takes no session: a document is read in a short session of its own, which is
    closed before the validation calls, so no DB connection is held while they run.

    Args:
        text: The text to check, or None to check a document.
        document_id: The document to check, or None to check text; must be attached to the chat.
        checker_context: Optional guidance from the chat model for the validation prompts.
        document_uuids: Documents attached to the chat, from the request context.
        user_id: The chat owner's id, for document access checks.
        context: What the validation calls are for, recorded on their llm_internal_response rows.

    Returns:
        The tool result text, and whether the check couldn't be run.
    """
    if document_id is not None:
        attached = {uuid.lower(): uuid for uuid in document_uuids}
        document_uuid = attached.get(document_id.strip().lower())
        if document_uuid is None:
            return _tagged(
                f"<error>Unknown document id: {document_id}. Use an id from the list of attached documents: "
                f"{', '.join(document_uuids) or 'none are attached'}.</error>"
            ), True
        async with async_db_session() as read_session:
            document = await _get_document_text(document_uuid, user_id, read_session)
        if document is None:
            return _tagged("<error>This document could not be read.</error>"), True
        document_name, content_to_check = document
        if not content_to_check.strip():
            return _tagged(
                "<error>This document has no readable text yet: it may still be processing, or have none.</error>"
            ), True
        checked = f"the attached document {document_name}"
    else:
        content_to_check = text or ""
        checked = "the text passed in"

    logger.info(f"Style guide check of {checked} ({len(content_to_check):,} chars)")
    return await analyse_content_for_style_guide(
        content_to_check, checked, format_checker_context(checker_context), context
    )
