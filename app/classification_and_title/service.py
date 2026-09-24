"""Service layer for combined chat title generation and classification.

Extracts title + classification logic from app/chat/service.py so the latter
can shed two functions. The single Haiku forced-tool call returns title, category, task_type
and discipline simultaneously — no extra LLM call beyond what title
generation already made.
"""

from dataclasses import dataclass

from anthropic.types import ToolUseBlock
from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.bedrock import BedrockHandler, RunMode
from app.chat.schemas import ChatRequest, ChatSuccessResponse, ChatTitleRequest
from app.classification_and_title.prompts import (
    TOOL_NAME_TITLE_AND_CLASSIFICATION,
    build_title_and_classification_system_prompt,
    build_title_and_classification_tool,
)
from app.classification_and_title.schemas import ClassificationInput
from app.config import LLM_CHAT_TITLE_MODEL, settings
from app.database.db_operations import DbOperations
from app.database.models import Chat, ChatClassification, ChatClassificationMapping, User
from app.database.table import LLMTable, async_db_session
from app.error_messages import ErrorMessages
from app.logs.logs_handler import logger


@dataclass
class TitleAndClassificationResult:
    title: str
    classification_id: int | None
    task_type: str | None
    discipline: str | None
    llm_internal_response_id: int | None


def _extract_tool_input(response, tool_name: str) -> dict | None:
    """Return the named ToolUseBlock's input dict from a Bedrock response, or None if absent.

    Mirrors the signature of app.bedrock.tool_schema.extract_tool_input (from
    fix/central-guidance-tool-schema-validation) so this can be swapped for that once it lands.
    """
    for block in response.content:
        if isinstance(block, ToolUseBlock) and block.name == tool_name:
            return block.input
    return None


async def create_title_and_classification(
    db_session: AsyncSession,
    chat,
    data: ChatTitleRequest,
) -> TitleAndClassificationResult:
    """Generate a chat title and infer category in a single forced-tool Haiku call.

    Replaces chat_create_title in app/chat/service.py. The full user query is sent
    (no truncation) so both title quality and classification signal are maximised.

    Returns a TitleAndClassificationResult. If classification fields cannot be parsed
    the result still contains a valid title so the caller can safely save it.
    """
    try:
        classifications = await DbOperations.get_classifications(db_session)

        documents_used_in_chat = await DbOperations.fetch_undeleted_chat_documents(db_session, chat.user_id, chat.id)
        document_names = [doc.name for doc in documents_used_in_chat] if documents_used_in_chat else []

        user_result = await db_session.execute(select(User).where(User.id == chat.user_id))
        user = user_result.scalar_one_or_none()
        job_title = user.job_title if user and user.job_title else None

        system_prompt = build_title_and_classification_system_prompt(classifications, document_names, job_title)
        tool_schema = build_title_and_classification_tool(classifications)

        llm_obj = LLMTable().get_by_model(LLM_CHAT_TITLE_MODEL)
        handler = BedrockHandler(system=system_prompt, mode=RunMode.ASYNC, llm=llm_obj)

        formatted_query = f"<human-query>{data.query}</human-query>"
        messages = handler.format_content_for_chat_title(formatted_query)

        max_attempts = settings.classification_title_max_attempts
        response = None
        tool_input = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = await handler.invoke_async(
                    messages,
                    db_session=db_session,
                    tools=[tool_schema],
                    tool_choice={"type": "tool", "name": TOOL_NAME_TITLE_AND_CLASSIFICATION},
                )
            except Exception as error:
                logger.warning(f"Classification attempt {attempt}/{max_attempts} failed for chat_id={chat.id}: {error}")
                continue

            tool_input = _extract_tool_input(response, TOOL_NAME_TITLE_AND_CLASSIFICATION)
            if tool_input is not None:
                break
            logger.warning(
                f"Classification attempt {attempt}/{max_attempts} returned no tool_use block for chat_id={chat.id}"
            )

        if tool_input is None:
            logger.error(f"Classification failed after {max_attempts} attempts for chat_id={chat.id}")
            tool_input = {}

        llm_internal_response_id = response.llm_internal_response_id if response is not None else None

        title = tool_input.get("title", "")
        task_type = tool_input.get("task_type")
        discipline = tool_input.get("discipline")
        classification_id = None
        category_name = tool_input.get("category")
        if category_name and category_name != "Other":
            matched = next((c for c in classifications if c.title == category_name), None)
            if matched:
                classification_id = matched.id

        if len(title) > 255:
            logger.warning(f"Generated title exceeds 255 characters. Truncating: {title}")
            title = title[:252] + "..."

        logger.info(
            f"Chat title created: {title!r}, classification_id={classification_id}, "
            f"task_type={task_type!r}, discipline={discipline!r}"
        )
        return TitleAndClassificationResult(
            title=title,
            classification_id=classification_id,
            task_type=task_type,
            discipline=discipline,
            llm_internal_response_id=llm_internal_response_id,
        )

    except Exception as error:
        logger.error(f"Error in create_title_and_classification: {error}", exc_info=True)
        raise Exception(ErrorMessages.CHAT_TITLE_NOT_CREATED, error) from error


async def _insert_chat_classification_mapping(
    db_session: AsyncSession,
    chat_id: int,
    classification_id: int | None,
    task_type: str | None,
    discipline: str | None,
    llm_internal_response_id: int | None,
) -> ChatClassificationMapping:
    stmt = (
        insert(ChatClassificationMapping)
        .values(
            chat_id=chat_id,
            classification_id=classification_id,
            task_type=task_type,
            discipline=discipline,
            llm_internal_response_id=llm_internal_response_id,
        )
        .returning(ChatClassificationMapping)
    )
    result = await db_session.execute(stmt)
    return result.scalars().unique().one()


def _chat_success_response(chat: Chat) -> ChatSuccessResponse:
    return ChatSuccessResponse(uuid=chat.uuid, created_at=chat.created_at, updated_at=chat.updated_at, title=chat.title)


async def update_chat_title(chat: Chat, data: ChatRequest) -> ChatSuccessResponse:
    """Generate title + classification and persist both, each in its own transaction.

    Transactions:
        1. Generate: reads and the LLM call, which writes llm_internal_response.
        2. Title: writes the chat title. Failure propagates.
        3. Classification: writes the mapping. Failure is logged and swallowed.

    Args:
        chat: The chat to title and classify.
        data: The chat request; its query is used to generate the title.

    Returns:
        ChatSuccessResponse for the chat, with the new title if one was generated.
    """
    async with async_db_session() as db_session:
        result = await create_title_and_classification(db_session, chat, ChatTitleRequest(**data.to_dict()))

    if result.title:
        try:
            async with async_db_session() as db_session:
                chat_result = await DbOperations.chat_update_title(db_session, chat, result.title)
        except Exception as error:
            logger.error(f"Failed to save chat title for chat_id={chat.id}: {error}", exc_info=True)
            raise
    else:
        chat_result = chat
    response = _chat_success_response(chat_result)

    try:
        async with async_db_session() as db_session:
            await _insert_chat_classification_mapping(
                db_session=db_session,
                chat_id=chat.id,
                classification_id=result.classification_id,
                task_type=result.task_type,
                discipline=result.discipline,
                llm_internal_response_id=result.llm_internal_response_id,
            )
    except Exception:
        logger.exception(f"Failed to persist chat classification for chat_id={chat.id}")

    return response


async def patch_chat_title(db_session: AsyncSession, chat: Chat, title: str) -> ChatSuccessResponse:
    """Set a user-supplied chat title.

    Args:
        db_session: The active database session for performing the update.
        chat: The chat to update.
        title: The new title.

    Returns:
        ChatSuccessResponse for the updated chat.
    """
    chat_result = await DbOperations.chat_update_title(db_session, chat, title)
    return _chat_success_response(chat_result)


async def get_classifications(db_session: AsyncSession) -> list[dict]:
    rows = await DbOperations.get_classifications(db_session)
    return [{"uuid": r.uuid, "title": r.title, "description": r.description} for r in rows]


async def bulk_sync_classifications(
    db_session: AsyncSession,
    inputs: list[ClassificationInput],
) -> dict:
    """Sync the canonical classification list into the DB.

    - New title: created.
    - Existing title, description changed: description overwritten.
    - Existing title, previously soft-deleted: revived.
    - Title absent from incoming list, has FK references: soft-deleted (deprecated).
    - Title absent from incoming list, no FK references: soft-deleted.
    """
    try:
        return await _bulk_sync_classifications(db_session, inputs)
    except Exception as error:
        logger.error(f"Error during bulk classification sync: {error}", exc_info=True)
        raise


async def _bulk_sync_classifications(
    db_session: AsyncSession,
    inputs: list[ClassificationInput],
) -> dict:
    incoming_by_title = {c.title: c for c in inputs}

    existing_result = await db_session.execute(select(ChatClassification))
    existing_by_title = {c.title: c for c in existing_result.scalars().all()}

    created = updated = deprecated = 0

    for title, inp in incoming_by_title.items():
        if title not in existing_by_title:
            await db_session.execute(insert(ChatClassification).values(title=title, description=inp.description))
            created += 1
            logger.info(f"Classification created: {title!r}")
            continue

        row = existing_by_title[title]
        changes = {}
        if row.description != inp.description:
            changes["description"] = inp.description
            updated += 1
        if row.deleted_at is not None:
            changes["deleted_at"] = None
            logger.info(f"Classification revived: {title!r}")

        if not changes:
            continue

        await db_session.execute(update(ChatClassification).where(ChatClassification.id == row.id).values(**changes))
        if "description" in changes:
            logger.info(f"Classification description updated: {title!r}")

    absent_ids = [
        row.id for title, row in existing_by_title.items() if title not in incoming_by_title and row.deleted_at is None
    ]
    mapped_classification_ids = set()
    if absent_ids:
        mapped_result = await db_session.execute(
            select(ChatClassificationMapping.classification_id)
            .where(ChatClassificationMapping.classification_id.in_(absent_ids))
            .distinct()
        )
        mapped_classification_ids = {row[0] for row in mapped_result.all()}

    for title, row in existing_by_title.items():
        if title in incoming_by_title or row.deleted_at is not None:
            continue
        await db_session.execute(
            update(ChatClassification).where(ChatClassification.id == row.id).values(deleted_at=func.now())
        )
        deprecated += 1
        reason = "has FK references" if row.id in mapped_classification_ids else "no FK references"
        logger.info(f"Classification deprecated ({reason}): {title!r}")

    return {"created": created, "updated": updated, "deprecated": deprecated}
