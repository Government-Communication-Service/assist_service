"""Style guide checks for the chat agent loop's check_style_guide tool."""

import logging
from typing import List, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.bedrock.call_log import LlmCallContext
from app.config import STYLE_GUIDE_MAX_DOCUMENT_CHARS, StyleGuideThoroughness, settings
from app.database.models import Document, DocumentChunk, DocumentUserMapping
from app.database.table import async_db_session
from app.style_guide.engine import run_check
from app.style_guide.report import format_result, tagged
from app.style_guide.rules import RULES, rules_for

logger = logging.getLogger(__name__)


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
    logger.info(f"Retrieved {len(chunks)} chunks from document {document.id}")
    return document.name, "".join(f"{chunk.content}\n" for chunk in chunks)


async def analyse_content_for_style_guide(
    content_to_check: str,
    checked: str,
    checker_context: str | None,
    context: LlmCallContext,
    thoroughness: StyleGuideThoroughness | None = None,
) -> Tuple[str, bool]:
    """Check some content against the style guide and list what was found.

    The result is data for the chat model: what was checked and each issue found, with no corrected
    version and no instructions. How to present it is up to the chat model, guided by the tool
    description. If some model calls fail, the issues found by the rest are returned with a warning.

    Args:
        content_to_check: Text to check.
        checked: What was checked, in words, e.g. "the user's latest message".
        checker_context: Guidance from the chat model for the model calls, or None.
        context: What the model calls are for, recorded on their llm_internal_response rows.
        thoroughness: Which rules to run; defaults to the style_guide_default_thoroughness setting.

    Returns:
        The <style-guide-check> segment, and whether the check couldn't be run.
    """
    if len(content_to_check) > STYLE_GUIDE_MAX_DOCUMENT_CHARS:
        char_count = len(content_to_check)
        logger.warning(f"Text too long for a style guide check: {char_count:,} chars")
        return tagged(
            f"<checked>{checked}</checked>\n"
            f"<error>The text is too long to check in one go ({char_count:,} characters). It can be checked "
            f"in sections of under {STYLE_GUIDE_MAX_DOCUMENT_CHARS:,} characters.</error>"
        ), True

    if not content_to_check.strip():
        return tagged(f"<checked>{checked}</checked>\n<error>There was no text to check.</error>"), True

    thoroughness = thoroughness or settings.style_guide_default_thoroughness
    rules = rules_for(thoroughness)
    try:
        result = await run_check(content_to_check, checker_context, context, rules=rules)
    except Exception:
        logger.exception("Style guide check failed")
        return tagged(
            f"<checked>{checked}</checked>\n<error>The check could not be completed. Try again.</error>"
        ), True
    return format_result(result, content_to_check, checked, thoroughness, len(RULES) - len(rules))


async def check_style_guide_for_tool(
    text: str | None,
    document_id: str | None,
    checker_context: str | None,
    document_uuids: List[str],
    user_id: int,
    context: LlmCallContext,
    thoroughness: StyleGuideThoroughness | None = None,
) -> Tuple[str, bool]:
    """Run a style guide check chosen by the chat model, on either some text or one attached document.

    Writes nothing, so it takes no session: a document is read in a short session of its own, which is
    closed before the model calls, so no DB connection is held while they run.

    Args:
        text: The text to check, or None to check a document.
        document_id: The document to check, or None to check text; must be attached to the chat.
        checker_context: Optional guidance from the chat model for the model calls.
        document_uuids: Documents attached to the chat, from the request context.
        user_id: The chat owner's id, for document access checks.
        context: What the model calls are for, recorded on their llm_internal_response rows.
        thoroughness: Which rules to run, or None for the default.

    Returns:
        The tool result text, and whether the check couldn't be run.
    """
    if document_id is not None:
        attached = {uuid.lower(): uuid for uuid in document_uuids}
        document_uuid = attached.get(document_id.strip().lower())
        if document_uuid is None:
            return tagged(
                f"<error>Unknown document id: {document_id}. Use an id from the list of attached documents: "
                f"{', '.join(document_uuids) or 'none are attached'}.</error>"
            ), True
        async with async_db_session() as read_session:
            document = await _get_document_text(document_uuid, user_id, read_session)
        if document is None:
            return tagged("<error>This document could not be read.</error>"), True
        document_name, content_to_check = document
        if not content_to_check.strip():
            return tagged(
                "<error>This document has no readable text yet: it may still be processing, or have none.</error>"
            ), True
        checked = f"the attached document {document_name}"
    else:
        content_to_check = text or ""
        checked = "the text passed in"

    logger.info(f"Style guide check of {len(content_to_check):,} chars")
    return await analyse_content_for_style_guide(content_to_check, checked, checker_context, context, thoroughness)
