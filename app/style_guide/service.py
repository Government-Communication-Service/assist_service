"""
Style Guide Service
Functions for integrating style guide checking into chat flow.
"""
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select

from app.bedrock import BedrockHandler, RunMode
from app.chat.utils import prepare_message_objects_for_llm
from app.config import (
    STYLE_GUIDE_LLM_MODEL,
    STYLE_GUIDE_MAX_CHUNK_CHARS,
    STYLE_GUIDE_MAX_DOCUMENT_CHARS,
)
from app.database.models import Document, DocumentChunk, DocumentUserMapping, Message
from app.database.table import LLMTable, async_db_session
from app.style_guide.style_guide_checker import (
    check_case_insensitive_rules,
    check_case_sensitive_rules,
    check_llm_validation_rules,
    generate_summary_and_fix,
    load_rule_mapping,
    split_text_into_chunks,
)

logger = logging.getLogger(__name__)

# Load rules once at module import
SCRIPT_DIR = Path(__file__).parent
RULES_FILE = SCRIPT_DIR / "rule_mapping.json"
RULES = load_rule_mapping(RULES_FILE)


class StyleGuideContentType(Enum):
    """Enum to indicate what content the user wants to check against the style guide."""
    QUERY_TEXT = "query_text"  # User wants to check text they've provided in the query
    DOCUMENTS = "documents"  # User wants to check their uploaded/selected documents
    SIMPLE_RESPONSE = "simple_response"  # User is asking a question, not requesting a check


async def _get_document_content_for_style_guide(
    document_uuids: List[str],
    user_id: int,
) -> Tuple[str, List[str]]:
    """
    Retrieve full content from user's uploaded documents by getting all chunks.

    Args:
        document_uuids: List of document UUIDs to retrieve content from
        user_id: The user's ID (integer) for access verification

    Returns:
        Tuple of (concatenated document content, list of document names that were retrieved)
    """
    if not document_uuids:
        return "", []

    content_parts = []
    document_names = []

    try:
        async with async_db_session() as db_session:
            for doc_uuid in document_uuids:
                # Verify user has access to this document
                doc_mapping_query = await db_session.execute(
                    select(DocumentUserMapping)
                    .join(Document, Document.id == DocumentUserMapping.document_id)
                    .where(Document.uuid == doc_uuid)
                    .where(DocumentUserMapping.user_id == user_id)
                    .where(DocumentUserMapping.deleted_at.is_(None))
                    .where(Document.deleted_at.is_(None))
                )
                if not doc_mapping_query.scalar_one_or_none():
                    logger.warning(f"User {user_id} does not have access to document {doc_uuid}")
                    continue

                # Get document and its chunks
                doc_query = await db_session.execute(
                    select(Document).where(Document.uuid == doc_uuid)
                )
                document = doc_query.scalar_one_or_none()
                if not document:
                    logger.warning(f"Document {doc_uuid} not found")
                    continue

                chunks_query = await db_session.execute(
                    select(DocumentChunk)
                    .where(DocumentChunk.document_id == document.id)
                    .order_by(DocumentChunk.id)
                )
                chunks = chunks_query.scalars().all()

                if chunks:
                    doc_content = f"--- Document: {document.name} ---\n"
                    for chunk in chunks:
                        doc_content += f"{chunk.content}\n"
                    content_parts.append(doc_content)
                    document_names.append(document.name)
                    logger.info(f"Retrieved {len(chunks)} chunks from document '{document.name}'")
                else:
                    logger.warning(f"No chunks found for document {doc_uuid}")

        combined_content = "\n".join(content_parts) if content_parts else ""
        logger.info(f"Retrieved content from {len(document_names)} documents, total {len(combined_content)} characters")
        return combined_content, document_names

    except Exception as e:
        logger.error(f"Error retrieving document content for style guide: {e}", exc_info=True)
        return "", []


def _wrap_conversation_context(messages: List[Message]) -> str:
    """
    Wrap conversation messages for context in style guide checks.
    Includes the full conversation history so the LLM understands what's being checked.

    Args:
        messages: List of Message objects in the conversation

    Returns:
        Formatted conversation context string
    """
    if not messages:
        return ""

    # Use the same preparation as other features
    messages_prep: list[dict] = prepare_message_objects_for_llm(messages)

    context = "\n<ConversationContext>\n"
    for msg in messages_prep:
        role = msg['role']
        content = msg['content']
        context += f"<{role}>\n{content}\n</{role}>\n"
    context += "</ConversationContext>\n"

    return context


async def determine_style_guide_content_type(
    messages: List[Message],
    has_documents: bool,
    llm_model: str = STYLE_GUIDE_LLM_MODEL
) -> StyleGuideContentType:
    """
    Router agent that decides what content the user wants checked against the style guide.

    For initial messages and follow-ups, examines the conversation to decide:
    - QUERY_TEXT: User wants to check the text they've provided in their message
    - DOCUMENTS: User wants to check their uploaded/selected documents
    - SIMPLE_RESPONSE: User is asking a question or requesting modifications (no new check needed)

    Args:
        messages: Full conversation history (must include the latest user message)
        has_documents: Whether the user has documents selected in this chat
        llm_model: LLM model to use for routing decision

    Returns:
        StyleGuideContentType indicating what should be checked
    """
    if not messages:
        # No messages means we can't determine intent - default to query text
        return StyleGuideContentType.QUERY_TEXT

    # For initial messages, skip the LLM router entirely - use deterministic logic.
    # The user has just arrived at the style guide checker, so:
    # - If they have documents attached, check those
    # - Otherwise, check the text they typed
    # The LLM router is only needed for follow-ups where intent is ambiguous.
    if len(messages) == 1:
        if has_documents:
            logger.info("Initial message with documents - routing to CHECK_DOCUMENTS")
            return StyleGuideContentType.DOCUMENTS
        logger.info("Initial message without documents - routing to CHECK_QUERY_TEXT")
        return StyleGuideContentType.QUERY_TEXT

    try:
        llm_obj = LLMTable().get_by_model(llm_model)
        if not llm_obj:
            logger.warning(f"LLM model {llm_model} not found, defaulting to QUERY_TEXT")
            return StyleGuideContentType.QUERY_TEXT

        # Get conversation context with the entire history
        conversation_context = _wrap_conversation_context(messages)

        # Get the latest user message
        messages_prep = prepare_message_objects_for_llm(messages)
        latest_user_message = None
        for msg in reversed(messages_prep):
            if msg['role'] == 'user':
                latest_user_message = msg['content']
                break

        if not latest_user_message:
            return StyleGuideContentType.QUERY_TEXT

        # Build document context info for the prompt
        document_info = ""
        if has_documents:
            document_info = """
IMPORTANT: The user has uploaded/selected documents in this chat session.
They may want to check EITHER their uploaded documents OR the text in their message."""
        else:
            document_info = """
NOTE: The user has NOT selected any documents in this chat session.
If they want a style guide check, it will be on the text they've written in their message."""

        # For follow-up messages, check if this is a simple response first
        is_follow_up = len(messages) > 1

        # Create the router prompt
        router_prompt = f"""You are a router that decides what the user wants checked against the GOV.UK style guide.

{document_info}

Given the conversation history and the user's latest message, decide:

DECISION OPTIONS:
1. CHECK_QUERY_TEXT: The user wants you to check/analyze the text content in their message
   - Examples: "check this text: [text]", "is this paragraph OK?", "review: [content]"
   - The user has written or pasted text they want checked

2. CHECK_DOCUMENTS: The user wants you to check their uploaded/selected documents
   - Examples: "check my document", "is my uploaded file compliant?", "analyze the document I selected"
   - Only valid if the user has documents selected (see IMPORTANT note above)
   - If no documents are selected, treat as CHECK_QUERY_TEXT instead

3. SIMPLE_RESPONSE: The user is NOT requesting a new style guide check
   - Examples: "make it shorter", "what does that rule mean?", "fix the second paragraph", "thanks"
   - The user is asking a question, requesting modifications to previous results, or just chatting
   {"- This is more likely for follow-up messages after an initial check" if is_follow_up else ""}

{conversation_context}

User's latest message: {latest_user_message}

You MUST respond with ONLY one of these exact phrases: "CHECK_QUERY_TEXT", "CHECK_DOCUMENTS", or "SIMPLE_RESPONSE"

{"If uncertain between CHECK options and SIMPLE_RESPONSE, default to SIMPLE_RESPONSE for follow-ups."
 if is_follow_up else "If uncertain, default to CHECK_QUERY_TEXT for initial messages."}

Your answer (one phrase only):"""

        # Call the LLM for routing decision
        bedrock_handler = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)
        response = await bedrock_handler.invoke_async(
            messages=[{
                "role": "user",
                "content": router_prompt
            }]
        )

        response_text = response.content[0].text.strip().upper()
        logger.info(f"Style guide content type router decision: {response_text}")

        # Parse the decision
        if "CHECK_DOCUMENTS" in response_text:
            if has_documents:
                return StyleGuideContentType.DOCUMENTS
            # User asked for documents but none selected - fall back to query text
            logger.info("User requested document check but no documents selected, falling back to QUERY_TEXT")
            return StyleGuideContentType.QUERY_TEXT
        if "CHECK_QUERY_TEXT" in response_text:
            return StyleGuideContentType.QUERY_TEXT
        return StyleGuideContentType.SIMPLE_RESPONSE

    except Exception as e:
        logger.error(f"Error in style guide content type router: {e}", exc_info=True)
        # On error, default to query text for initial, simple response for follow-ups
        if messages and len(messages) > 1:
            return StyleGuideContentType.SIMPLE_RESPONSE
        return StyleGuideContentType.QUERY_TEXT


def _filter_violations_for_chunk(violations: List[Dict], chunk: str) -> List[Dict]:
    """Filter violations to those whose sentences/occurrences appear in the given chunk.

    Narrows each violation's ``sentences``/``occurrences`` lists to only entries
    that are present (as substrings) in *chunk*.  Violations with no locating
    information are always included, as are violations where at least one
    sentence or occurrence matches.

    Args:
        violations: All violations found across the full document.
        chunk: A single text chunk from the original document.

    Returns:
        A copy of the violations list filtered to this chunk, with location
        lists narrowed to only the entries present in the chunk.
    """
    chunk_violations: List[Dict] = []
    for violation in violations:
        sentences = violation.get("sentences", [])
        occurrences = violation.get("occurrences", [])

        relevant_sentences = [s for s in sentences if s and s in chunk]
        relevant_occurrences = [o for o in occurrences if o and o in chunk]

        has_sentences = bool(sentences)
        has_occurrences = bool(occurrences)

        # Keep if: no locating info (structural/doc-level rule), OR at least one match.
        if (not has_sentences and not has_occurrences) or relevant_sentences or relevant_occurrences:
            v_copy = dict(violation)
            if has_sentences:
                v_copy["sentences"] = relevant_sentences
            if has_occurrences:
                v_copy["occurrences"] = relevant_occurrences
            chunk_violations.append(v_copy)

    return chunk_violations


async def _run_llm_validation_on_chunks(
    chunks: List[str],
    llm_model: str,
) -> List[Dict]:
    """Run LLM validation rules across multiple document chunks.

    Deduplicates violations by ``rule_id`` so the same rule is not reported
    multiple times simply because it was triggered in more than one chunk.

    Args:
        chunks: Text chunks to validate.
        llm_model: LLM model identifier.

    Returns:
        Deduplicated list of LLM violations found across all chunks.
    """
    seen_rule_ids: set = set()
    llm_violations: List[Dict] = []

    for idx, chunk in enumerate(chunks):
        logger.info(f"Running LLM validation on chunk {idx + 1}/{len(chunks)}")
        chunk_violations = await check_llm_validation_rules(chunk, RULES, llm_model=llm_model)
        for violation in chunk_violations:
            rule_id = violation.get("rule_id")
            if rule_id not in seen_rule_ids:
                llm_violations.append(violation)
                if rule_id:
                    seen_rule_ids.add(rule_id)

    return llm_violations


async def _generate_chunked_summary_and_fix(
    chunks: List[str],
    all_violations: List[Dict],
    llm_model: str,
    output_dir: Path,
    conversation_context: str = "",
) -> Optional[Dict]:
    """Generate a style guide summary and corrected document across multiple chunks.

    Each chunk is processed independently via :func:`generate_summary_and_fix`.
    The ``fixed_document`` outputs are concatenated and summaries are joined
    into a single combined result.

    Args:
        chunks: Non-overlapping text chunks from the original document.
        all_violations: All violations found across the full document.
        llm_model: LLM model identifier.
        output_dir: Directory passed through to :func:`generate_summary_and_fix`.
        conversation_context: Optional conversation history (only passed to
            the first chunk to avoid repetition).

    Returns:
        Dict with combined ``summary`` and ``fixed_document``, or ``None``
        if every chunk failed.
    """
    fixed_parts: List[str] = []
    summaries: List[str] = []

    for idx, chunk in enumerate(chunks):
        # Narrow violations to those whose text appears in this chunk;
        # fall back to all violations for structural/doc-level rules.
        chunk_violations = _filter_violations_for_chunk(all_violations, chunk)
        violations_to_use = chunk_violations if chunk_violations else all_violations

        result = await generate_summary_and_fix(
            document=chunk,
            violations=violations_to_use,
            llm_model=llm_model,
            output_dir=output_dir,
            # Only pass conversation context to the first chunk.
            conversation_context=conversation_context if idx == 0 else "",
        )

        if result:
            fixed_parts.append(result.get("fixed_document") or chunk)
            if result.get("summary"):
                summaries.append(result["summary"])
        else:
            logger.warning(f"Failed to generate summary/fix for chunk {idx + 1}/{len(chunks)}")
            fixed_parts.append(chunk)

    if not fixed_parts:
        return None

    combined_fixed = "\n\n".join(fixed_parts)
    combined_summary = " ".join(summaries) if summaries else None

    if not combined_summary:
        return None

    return {"summary": combined_summary, "fixed_document": combined_fixed}


async def check_content_against_style_guide(
    content: str,
    llm_model: str = STYLE_GUIDE_LLM_MODEL,
    messages: Optional[List[Message]] = None,
    document_uuids: Optional[List[str]] = None,
    user_id: Optional[int] = None,
) -> Optional[str]:
    """
    Check content against GOV.UK style guide rules.

    This function runs style guide checks and returns a formatted prompt segment
    that will be passed to the main LLM along with the user's query.
    The main LLM then generates a response based on the style guide analysis.

    Args:
        content: The text content from the user's query
        llm_model: LLM model to use for validation
        messages: Optional list of Message objects for conversation context (for follow-ups)
        document_uuids: Optional list of document UUIDs the user has selected
        user_id: Optional user ID (integer) for document access verification

    Returns:
        Formatted prompt segment with style guide analysis, or None if no analysis needed
    """
    try:
        logger.info(
            f"Style guide check started - content length: {len(content)} chars, "
            f"documents: {len(document_uuids) if document_uuids else 0}"
        )

        # Build conversation context if messages are provided
        conversation_context = ""
        if messages:
            conversation_context = _wrap_conversation_context(messages)

        # Determine if user has documents available
        has_documents = bool(document_uuids and user_id)

        # Use the router to determine what content type to check
        content_type = await determine_style_guide_content_type(messages, has_documents, llm_model)
        logger.debug(f"Router decision: {content_type.value}")

        # For simple responses (follow-up questions, modifications), don't add style guide context
        # Let the main LLM handle it naturally with the conversation history
        if content_type == StyleGuideContentType.SIMPLE_RESPONSE:
            logger.debug("Simple response mode - no style guide analysis needed, main LLM will handle")
            return None

        # Determine what content to check based on router decision
        content_to_check = ""
        content_source_info = ""

        if content_type == StyleGuideContentType.DOCUMENTS:
            # Retrieve document content
            document_content, document_names = await _get_document_content_for_style_guide(
                document_uuids, user_id
            )

            if not document_content:
                # No document content retrieved - fall back to query text
                logger.warning("Failed to retrieve document content, falling back to query text")
                content_to_check = content
                content_source_info = "Checking the text you provided"
            else:
                content_to_check = document_content
                content_source_info = f"Checking your document(s): {', '.join(document_names)}"
                logger.debug(f"Content from {len(document_names)}, ({len(document_content)} chars)")
        else:
            # Check query text
            content_to_check = content
            content_source_info = "Checking the text you provided"

        # Max document size guard – reject before any LLM calls are made.
        if len(content_to_check) > STYLE_GUIDE_MAX_DOCUMENT_CHARS:
            char_count = len(content_to_check)
            logger.warning(
                f"Document too large for style guide check: {char_count:,} chars "
                f"(limit: {STYLE_GUIDE_MAX_DOCUMENT_CHARS:,})"
            )
            prompt_segment = (
                "<style-guide-analysis>\n"
                "<instructions>\n"
                "The document provided is too long to check in a single request. "
                "Inform the user clearly and suggest they split it into smaller sections.\n"
                "</instructions>\n"
                f"<error>The document is too large to check in one go "
                f"({char_count:,} characters). "
                f"Please split your document into sections of under "
                f"{STYLE_GUIDE_MAX_DOCUMENT_CHARS:,} characters "
                f"and check each section separately.</error>\n"
                "</style-guide-analysis>"
            )
            return prompt_segment

        if not content_to_check.strip():
            logger.warning("No content to check against style guide")
            prompt_segment = (
                "<style-guide-analysis>\n"
                "<instructions>\n"
                "No content was provided to check. Inform the user they need to provide text or select documents.\n"
                "</instructions>\n"
                "<message>No content was provided to check against the style guide. "
                "Please provide text or select documents to analyze.</message>\n"
                "</style-guide-analysis>"
            )
            return prompt_segment

        # Full analysis flow
        logger.debug(f"Running style guide analysis on {len(content_to_check)} chars of content")

        # Determine if document needs to be chunked for LLM calls.
        needs_chunking = len(content_to_check) > STYLE_GUIDE_MAX_CHUNK_CHARS
        chunks: List[str] = []
        if needs_chunking:
            chunks = split_text_into_chunks(content_to_check, STYLE_GUIDE_MAX_CHUNK_CHARS)
            logger.debug(
                f"Document ({len(content_to_check):,} chars) split into "
                f"{len(chunks)} chunk(s) for LLM processing"
            )

        # Run deterministic checks on the full document (regex – fast regardless of size).
        case_insensitive_violations = check_case_insensitive_rules(content_to_check, RULES)
        case_sensitive_violations = check_case_sensitive_rules(content_to_check, RULES)

        # Run LLM validation – per chunk if document was split, otherwise on full document.
        if needs_chunking:
            llm_violations = await _run_llm_validation_on_chunks(chunks, llm_model)
        else:
            # Run LLM validation (includes deterministic:true with pass_to_llm AND deterministic:false)
            llm_violations = await check_llm_validation_rules(content_to_check, RULES, llm_model=llm_model)

        # Combine all violations
        all_violations = case_insensitive_violations + case_sensitive_violations + llm_violations
        logger.info(f"Style guide check complete: {len(all_violations)} total violations found")

        if not all_violations:
            prompt_segment = (
                "<style-guide-analysis>\n"
                "<instructions>\n"
                "A style guide check has been performed and NO violations were found. "
                "Format your response with the heading: # GOV.UK Style Guide Analysis\n"
                "Then state what was checked and that no violations were detected.\n"
                "</instructions>\n"
                f"<content_source>{content_source_info}</content_source>\n"
                "<result>No style guide violations were detected in the submitted content. "
                "The content appears to follow GOV.UK style guide principles.</result>\n"
                "</style-guide-analysis>"
            )
            return prompt_segment

        # Call LLM to generate summary and fixed document.
        if needs_chunking:
            summary_result = await _generate_chunked_summary_and_fix(
                chunks=chunks,
                all_violations=all_violations,
                llm_model=llm_model,
                output_dir=SCRIPT_DIR,
                conversation_context=conversation_context,
            )
        else:
            summary_result = await generate_summary_and_fix(
                document=content_to_check,
                violations=all_violations,
                llm_model=llm_model,
                output_dir=SCRIPT_DIR,
                conversation_context=conversation_context,
            )

        if not summary_result:
            # Fallback to simple violation list if summary generation fails
            logger.warning("Failed to generate summary, falling back to violation list")
            prompt_segment = "<style-guide-analysis>\n"
            prompt_segment += "<instructions>\n"
            prompt_segment += "A style guide check has been performed on the user's content. "
            prompt_segment += "You MUST format your response as follows:\n"
            prompt_segment += "1. Start with a heading: # GOV.UK Style Guide Analysis\n"
            prompt_segment += "2. State what was checked\n"
            prompt_segment += "3. List the violations found\n"
            prompt_segment += "</instructions>\n\n"
            prompt_segment += f"<content_source>{content_source_info}</content_source>\n\n"
            prompt_segment += (
                f"<violations>\nThe following {len(all_violations)} style guide violations were detected:\n\n"
            )
            for i, violation in enumerate(all_violations, 1):
                rule_id = violation.get("rule_id", "unknown")
                rule_title = violation.get("rule_title", "No title")
                prompt_segment += f"{i}. {rule_title} (Rule ID: {rule_id})\n"
            prompt_segment += "</violations>\n"
            prompt_segment += "</style-guide-analysis>"
            return prompt_segment

        # Format the LLM-generated summary and fixed document as prompt segment
        # Include instructions for the LLM to format its response appropriately
        prompt_segment = "<style-guide-analysis>\n"
        prompt_segment += "<instructions>\n"
        prompt_segment += "A style guide check has been performed on the user's content. "
        prompt_segment += "You MUST format your response as follows:\n"
        prompt_segment += "1. Start with a heading: # GOV.UK Style Guide Analysis\n"
        prompt_segment += (
            "2. State what was checked (e.g., '**Checking the text you provided**'"
            " or '**Checking your document(s): [names]**')\n"
        )
        prompt_segment += "3. Show the Summary of Violations under heading: **Summary of Violations:**\n"
        prompt_segment += (
            "4. Show the corrected document under heading:"
            " **Corrected Document (GOV.UK Style Guide compliant):**\n"
        )
        prompt_segment += "Present the information from the analysis below in this format.\n"
        prompt_segment += "</instructions>\n\n"
        prompt_segment += f"<content_source>{content_source_info}</content_source>\n\n"
        prompt_segment += f"<violation_summary>{summary_result.get('summary', 'N/A')}</violation_summary>\n\n"
        prompt_segment += f"<corrected_document>\n{summary_result.get('fixed_document', '')}\n</corrected_document>\n"
        prompt_segment += "</style-guide-analysis>"

        return prompt_segment

    except Exception as e:
        logger.error(f"Error checking style guide: {e}", exc_info=True)
        # Return None rather than failing the entire chat
        return None



