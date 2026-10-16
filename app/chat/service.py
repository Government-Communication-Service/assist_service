# ruff: noqa: B008
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional
from uuid import UUID

import sqlalchemy
from fastapi import Body, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import cast, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.constants import USER_GROUPS_ALIAS
from app.auth.verify_service import (
    verify_and_get_auth_session_from_header,
    verify_and_get_user_from_path_and_header,
    verify_and_parse_uuid,
)
from app.bedrock import BedrockHandler, RunMode
from app.bedrock.bedrock_types import BedrockError, BedrockErrorType
from app.bedrock.thinking import thinking_kwargs
from app.central_guidance.schemas import RagRequest
from app.chat.agent_loop import (
    AgentLoopEvent,
    AgentLoopResult,
    LoopFinished,
    SourcesUpdated,
    TextDelta,
    ToolCallRecord,
    ToolStatus,
    run_agent_loop,
)
from app.chat.cache_control import apply_compaction_aware_cache_control
from app.chat.constants import DELETION_NOTICE
from app.chat.prompts import build_chat_system_prompt, build_turn_context_block
from app.chat.schemas import (
    ChatBasicResponse,
    ChatCreateInput,
    ChatCreateMessageInput,
    ChatPost,
    ChatRequestData,
    ChatShareResponse,
    ChatSuccessResponse,
    ChatWithAllMessages,
    ChatWithLatestMessage,
    DocumentAccessError,
    DocumentSchema,
    MessageBasicResponse,
    MessageDefaults,
    RoleEnum,
    Sources,
    UserChatsResponse,
)
from app.chat.tools import ToolContext, build_available_tools
from app.chat.utils import prepare_message_objects_for_llm
from app.compaction.service import (
    estimate_prefix_tokens,
    find_last_assistant_message_id,
    get_latest_compaction,
    is_compaction_locked,
    messages_since_compaction,
    schedule_compaction,
    should_compact,
)
from app.config import (
    CHAT_THINKING_LEVEL,
    LLM_CHAT_RESPONSE_MODEL,
    TEST_USER_GROUPS,
)
from app.database.db_operations import DbOperations
from app.database.models import (
    LLM,
    Chat,
    ChatDocumentMapping,
    Document,
    DocumentUserMapping,
    Message,
    ToolCall,
    User,
)
from app.database.table import (
    ChatTable,
    LLMTable,
    MessageTable,
    MessageUserGroupMappingTable,
    UseCaseTable,
    UserGroupTable,
    async_db_session,
)
from app.logs.logs_handler import logger


async def get_all_user_chats(db_session: AsyncSession, user: User):
    chats = await DbOperations.get_chats_by_user(db_session=db_session, user_id=user.id)

    return UserChatsResponse(chats=[ChatBasicResponse.model_validate(chat) for chat in chats], **user.client_response())


def chat_user_group_mapping(message: Message, user_group_ids: list[int]):
    user_group_mapping_table = MessageUserGroupMappingTable()

    for user_group_id in user_group_ids:
        user_group_mapping_table.create({"message_id": message.id, "user_group_id": user_group_id})


def chat_stream_message(chat: Chat, message_uuid: str, content: str, citations: str, sources: Sources) -> Dict:
    response = {
        **chat.client_response(),
        "message_streamed": {
            "uuid": str(message_uuid),
            "role": RoleEnum.assistant,
            "content": content,
            "citations": citations,
            "sources": sources.model_dump_json(exclude_none=True),
        },
    }

    logger.debug(f"API chat_stream_message: {json.dumps(response, indent=5)}")

    return response


def chat_stream_error_message(chat: Chat, ex: Exception, has_documents: bool, is_initial_call: bool) -> str:
    if isinstance(ex, BedrockError) and ex.error_type == BedrockErrorType.INPUT_TOO_LONG:
        if has_documents:
            error_message = "Input is too long, too many documents selected, select fewer documents"
        else:
            if is_initial_call:
                error_message = "Input is too long, reduce input text"
            else:
                error_message = "Input is too long, reduce input text or start a new chat with reduced input text"

        response = {
            **chat.client_response(),
            "error_code": "BEDROCK_SERVICE_INPUT_TOO_LONG_ERROR",
            "error_message": error_message,
        }
    else:
        response = {**chat.client_response(), "error_code": "BEDROCK_SERVICE_ERROR", "error_message": str(ex)}

    logger.debug(f"API chat_stream_message: {json.dumps(response, indent=5)}")

    return json.dumps(response)


def get_user_groups(
    user_groups_string=Header(
        default=TEST_USER_GROUPS,
        alias=USER_GROUPS_ALIAS,
        description="A comma-separated list of groups assigned to the user from GCS Connect which can be used to "
        "filter queries.",
    ),
):
    user_group_ids = []

    if user_groups_string:
        user_groups = user_groups_string.split(",")
        ug_table = UserGroupTable()
        for group_name in user_groups:
            group = ug_table.upsert_by_name(group_name)
            user_group_ids.append(group.id)

    return user_group_ids


def chat_request_data(
    user=Depends(verify_and_get_user_from_path_and_header),
    auth_session=Depends(verify_and_get_auth_session_from_header),
    data: ChatPost = Body(...),
    user_group_ids=Depends(get_user_groups),
):
    use_case_id = data.use_case_id
    del data.use_case_id
    data_dict = data.to_dict()

    try:
        if use_case_id:
            use_case_id = verify_and_parse_uuid(use_case_id)

            data_dict["use_case_id"] = UseCaseTable().get_by_uuid(use_case_id).id

        return ChatRequestData(
            user_id=user.id, auth_session_id=auth_session.id, user_group_ids=user_group_ids, **data_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


async def chat_create_stream(data: ChatCreateInput):
    data.stream = True
    response = await chat_create(data)

    return StreamingResponse(response, media_type="text/event-stream")


async def _chat_message_with_documents(chat: Chat, db_session: AsyncSession, request_input: Dict):
    """
    Checks if the chat item has documents referenced, then associates those documents under document_uuids key
     to the request input.
    Merges existing chat documents with new documents from request input.
    Args:
        chat (Chat): The Chat object containing the chat information (e.g., `id`).
        db_session (AsyncSession): The asynchronous database session to be used for the query.
        request_input (Dict): The dictionary containing the initial request input.

    Returns:
        Dict: The updated request input dictionary, with merged "document_uuids" key.
    """
    # Get existing chat documents
    chat_document_mappings = await DbOperations.fetch_undeleted_chat_documents(db_session, chat.user_id, chat.id)
    existing_document_uuids = [str(doc.uuid) for doc in chat_document_mappings] if chat_document_mappings else []

    # Get NEW documents from request (these are the ones user added to chat message)
    new_document_uuids = request_input.get("document_uuids") or []

    # Combine existing + new documents for RAG processing
    all_document_uuids = existing_document_uuids + new_document_uuids

    # Store combined list for RAG and new_documents_uuids for saving to DB
    request_input["document_uuids"] = all_document_uuids
    request_input["new_document_uuids"] = new_document_uuids

    return request_input


async def chat_add_message_stream(chat: Chat, data):
    chat_message = data.to_dict()
    # check if there are chat documents, then fetch and append them to the request
    async with async_db_session() as db_session:
        chat_message = await _chat_message_with_documents(chat, db_session, chat_message)
        turn = await chat_create_message(chat, ChatCreateMessageInput(**chat_message, stream=True), db_session)
    return StreamingResponse(turn.stream(), media_type="text/event-stream")


async def chat_add_message(chat: Chat, data):
    request_input = data.to_dict()
    # check if there are chat documents, then fetch and append them to the request
    async with async_db_session() as db_session:
        request_input = await _chat_message_with_documents(chat, db_session, request_input)
        turn = await chat_create_message(chat, ChatCreateMessageInput(**request_input), db_session)
    # Run the turn once the request's writes are committed, so their locks and connection aren't held across it.
    message = await turn.run()
    return ChatWithLatestMessage(
        **chat.client_response(),
        message=message.client_response(),
    )


async def patch_chat_favourite(db_session: AsyncSession, chat: Chat, favourite: bool) -> ChatSuccessResponse:
    """
    Updates the favourite status of a chat.

    Args:
        db_session (AsyncSession): The active database session for performing the update.
        chat (Chat): The Chat instance representing the chat to be favourited.
        favourite (bool): The new favourite status to be assigned to the chat.

    Returns:
        ChatSuccessResponse: Response object containing the updated chat details.

    Raises:
        Exception: If the database operation fails, the underlying DatabaseError will be propagated.
    """
    chat_result = await DbOperations.chat_update_favourite(db_session, chat, favourite)
    return ChatSuccessResponse(**chat_result.client_response())


async def patch_chat_share(
    db_session: AsyncSession, chat: Chat, share: bool, share_private: Optional[bool] = None
) -> ChatShareResponse:
    """
    Updates the share status of a chat.

    Args:
        db_session (AsyncSession): The active database session for performing the update.
        chat (Chat): The Chat instance representing the chat to be shared.
        share (bool): The new share status to be assigned to the chat.
        share_private (Optional[bool]): The new private share status. When None the existing
            value is left unchanged, keeping clients that predate private shares working.

    Returns:
        ChatShareResponse: Response object containing the updated chat details including share_code.

    Raises:
        Exception: If the database operation fails, the underlying DatabaseError will be propagated.
    """
    chat_result = await DbOperations.chat_update_share(db_session, chat, share, share_private)
    return ChatShareResponse(**chat_result.client_response())


async def get_chat_share_users(db_session: AsyncSession, chat: Chat) -> list[dict]:
    """
    Lists the users allowed to view a privately shared chat, with their notification state.

    Args:
        db_session (AsyncSession): The active database session.
        chat (Chat): The Chat instance whose private share list is requested.

    Returns:
        list[dict]: One dict per shared user with "uuid" and "notified_at" keys.
    """
    rows = await DbOperations.get_chat_share_users(db_session, chat)
    return [{"uuid": str(row.uuid), "notified_at": row.notified_at} for row in rows]


async def add_chat_share_user(db_session: AsyncSession, chat: Chat, shared_user_uuid: UUID) -> list[dict]:
    """
    Grants a user access to a chat's private share. The user is created if they are not
    yet known to this service (their details live in the PHP frontend's database; this
    API only tracks their UUID). Adding a user who already has access is a no-op.

    Args:
        db_session (AsyncSession): The active database session.
        chat (Chat): The Chat instance being shared.
        shared_user_uuid (UUID): The UUID of the user being granted access.

    Returns:
        list[dict]: The updated shared-user list, as returned by get_chat_share_users.
    """
    shared_user = await DbOperations.upsert_user_by_uuid(db_session, str(shared_user_uuid))
    await DbOperations.add_chat_share_user(db_session, chat, shared_user)
    return await get_chat_share_users(db_session, chat)


async def remove_chat_share_user(db_session: AsyncSession, chat: Chat, shared_user_uuid: UUID) -> Optional[list[dict]]:
    """
    Revokes a user's access to a chat's private share. This deletes the mapping record
    including its notification state, so re-adding the user starts fresh.

    Args:
        db_session (AsyncSession): The active database session.
        chat (Chat): The Chat instance being shared.
        shared_user_uuid (UUID): The UUID of the user whose access is being revoked.

    Returns:
        Optional[list[dict]]: The updated shared-user list, or None when the user was
            not part of the share.
    """
    shared_user = await DbOperations.get_user_by_uuid(db_session, str(shared_user_uuid))
    if shared_user is None:
        return None

    removed_count = await DbOperations.remove_chat_share_user(db_session, chat, shared_user)
    if removed_count == 0:
        return None

    return await get_chat_share_users(db_session, chat)


async def set_chat_share_user_notified(
    db_session: AsyncSession, chat: Chat, shared_user_uuid: UUID, notified: bool
) -> Optional[list[dict]]:
    """
    Records (notified=True) or clears (notified=False) when a user was notified that a
    chat was privately shared with them. The timestamp is set server-side.

    Args:
        db_session (AsyncSession): The active database session.
        chat (Chat): The Chat instance being shared.
        shared_user_uuid (UUID): The UUID of the user whose notification state is updated.
        notified (bool): True to stamp the notification time, False to clear it.

    Returns:
        Optional[list[dict]]: The updated shared-user list, or None when the user was
            not part of the share.
    """
    shared_user = await DbOperations.get_user_by_uuid(db_session, str(shared_user_uuid))
    if shared_user is None:
        return None

    notified_at = datetime.now() if notified else None
    updated_count = await DbOperations.set_chat_share_user_notified(db_session, chat, shared_user, notified_at)
    if updated_count == 0:
        return None

    return await get_chat_share_users(db_session, chat)


async def chat_archive(db_session: AsyncSession, chat: Chat) -> ChatSuccessResponse:
    """
    Archives a chat by setting the deleted_at timestamp.

    Args:
        db_session (AsyncSession): The active database session for performing the update.
        chat (Chat): The Chat instance representing the chat to be archived.

    Returns:
        ChatSuccessResponse: Response object containing the updated chat details.

    Raises:
        Exception: If the database operation fails, the underlying DatabaseError will be propagated.
    """
    chat_result = await DbOperations.chat_archive(db_session, chat)
    return ChatSuccessResponse(**chat_result.client_response())


async def chat_get_messages(chat: Chat):
    """
    Retrieves all messages for a specific chat, including related documents.
    Args:
        chat (Chat): The Chat instance representing the chat for which messages
                      are to be retrieved.

    Returns:
        ChatWithAllMessages: A response model containing the chat details and
                             a list of messages, including document details.

    """
    async with async_db_session() as db_session:
        chat = await DbOperations.get_chat_with_messages(db_session, chat.id, chat.user_id)
        chat_response = ChatWithAllMessages(
            uuid=chat.uuid,
            created_at=chat.created_at,
            updated_at=chat.updated_at,
            title=chat.title,
            favourite=chat.favourite,
            share=chat.share,
            share_code=chat.share_code,
            share_private=chat.share_private,
            from_open_chat=chat.from_open_chat,
            use_rag=chat.use_rag,
            use_gov_uk_search_api=chat.use_gov_uk_search_api,
            use_smart_targets=chat.use_smart_targets,
            documents=[
                DocumentSchema(
                    uuid=d.document_uuid,
                    name=d.document.name,
                    created_at=d.document.user_mappings[0].created_at,
                    expired_at=d.document.user_mappings[0].expired_at,
                    deleted_at=d.document.user_mappings[0].deleted_at,
                    last_used=d.document.user_mappings[0].last_used,
                )
                for d in chat.chat_document_mapping
            ],
            messages=[
                MessageBasicResponse(
                    uuid=m.uuid,
                    created_at=m.created_at,
                    updated_at=m.updated_at,
                    content=m.content,
                    role=m.role,
                    interrupted=m.interrupted,
                    citation=m.citation or "",
                    sources=m.sources or "",
                )
                for m in chat.messages
            ],
        )
        logger.debug(f"{chat_response=}")
        return chat_response


async def chat_create(input_data: ChatCreateInput) -> ChatWithLatestMessage:
    """
    Takes a chat creation request and creates a new chat with the message.
    Creates chat and message objects and saves them in the database,
    marks the message as the initial call (e.g. parent is null).
    if input request contains a use_case_id flag, then the chat is created from an open chat and use_case_id is set.

    Return a ChatWithLatestMessage instance, wrapping chat and the message details.
    """
    logger.debug(f"{input_data=}")
    title = "New chat"

    from_open_chat = True
    if input_data.use_case_id:
        from_open_chat = False

    logger.debug("starting creating db item")
    chat_repo = ChatTable()
    chat_data = {
        "user_id": input_data.user_id,
        "from_open_chat": from_open_chat,
        "use_case_id": input_data.use_case_id,
        "title": title,
        "use_rag": input_data.use_rag,
        "use_gov_uk_search_api": input_data.use_gov_uk_search_api,
        "use_smart_targets": input_data.use_smart_targets,
    }
    chat_obj = chat_repo.create(chat_data)

    logger.debug("starting create message")
    logger.debug(f"input_data.to_dict(): {input_data.to_dict()}")

    async with async_db_session() as db_session:
        db_session.add(chat_obj)
        await db_session.flush()
        await db_session.refresh(chat_obj)

        turn = await chat_create_message(
            chat=chat_obj,
            input_data=ChatCreateMessageInput(**input_data.to_dict(), initial_call=True),
            db_session=db_session,
        )
        await db_session.commit()

    # Run the turn once the request's writes are committed, so their locks and connection aren't held across it.
    if input_data.stream:
        return turn.stream()

    message = await turn.run()

    return ChatWithLatestMessage(**chat_obj.dict(), message=message.dict())


async def chat_create_message(chat: Chat, input_data: ChatCreateMessageInput, db_session: AsyncSession) -> "_AgentTurn":
    """Save the user message and the request-scope writes, and prepare the agent turn.

    The caller runs the returned turn (`stream()` or `run()`) after `db_session` is committed and
    closed, so no request transaction or row lock is held while the loop runs.
    """
    if not chat:
        raise Exception("Chat not found")

    chat_id = chat.id
    llm_obj = LLMTable().get_by_model(LLM_CHAT_RESPONSE_MODEL)

    if not llm_obj:
        raise Exception("LLM not found with name: " + LLM_CHAT_RESPONSE_MODEL)

    message_defaults = {
        "chat_id": chat_id,
        "auth_session_id": input_data.auth_session_id,
        "llm_id": llm_obj.id,
    }

    message_repo = MessageTable()
    messages = []
    parent_message_id = None

    if not input_data.initial_call:
        messages = message_repo.get_by_chat(chat_id)
        parent_message_id = messages[-1].id

    # If the previous request created this exact user message but never got a reply (e.g. the
    # client timed out and resubmitted), reuse that row instead of writing a duplicate one.
    pending_duplicate = messages and messages[-1].role == RoleEnum.user and messages[-1].content == input_data.query
    if pending_duplicate:
        m_user = messages[-1]
        messages = messages[:-1]
        if m_user.content_enhanced_with_rag is not None:
            # Written by the pre-agent-loop flow; replay the plain query instead.
            m_user = message_repo.update(m_user, {"content_enhanced_with_rag": None})
    else:
        m_user = message_repo.create(
            {
                "content": input_data.query,
                "role": RoleEnum.user,
                "parent_message_id": parent_message_id,
                **MessageDefaults(**message_defaults).dict(),
            }
        )

    ai_message = MessageDefaults(**message_defaults)

    system = await build_chat_system_prompt(db_session)

    context_ctx = ToolContext(
        chat=chat,
        user_id=chat.user_id,
        user_message=m_user,
        history=messages,
        input_data=input_data,
        document_uuids=list(input_data.document_uuids or []),
    )
    context_block = await build_turn_context_block(
        context_ctx, db_session, audience_segment_uuids=input_data.audience_segment_uuids
    )
    m_user = message_repo.update(
        m_user,
        {
            "raw_messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": context_block},
                        {"type": "text", "text": input_data.query},
                    ],
                }
            ]
        },
    )

    if input_data.user_group_ids:
        chat_user_group_mapping(m_user, input_data.user_group_ids)

    llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC, system=system)

    rag_request = RagRequest(
        use_central_rag=input_data.use_rag,
        user_id=chat.user_id,
        query=input_data.query,
        document_uuids=input_data.document_uuids,
    )
    # Save documents to chat_document_mapping
    new_document_uuids = getattr(input_data, "new_document_uuids", [])

    if new_document_uuids:
        documents_to_save = new_document_uuids
    elif input_data.initial_call:
        documents_to_save = rag_request.document_uuids
    else:
        documents_to_save = []

    if documents_to_save:
        save_rag_request = RagRequest(
            use_central_rag=False,
            user_id=chat.user_id,
            query="",
            document_uuids=documents_to_save,
        )
        await _check_document_access(db_session, save_rag_request)
        await _save_chat_documents(db_session, m_user, save_rag_request)

    # extend document expiry time and update last_used time
    if rag_request.document_uuids:
        await _extend_document_expiry_and_last_used_time(rag_request, db_session)

    compaction = await get_latest_compaction(chat_id, db_session)
    formatted_messages = prepare_message_objects_for_llm(messages + [m_user], compaction=compaction)

    # Excludes m_user: compaction can't touch the current turn, so it shouldn't count toward the threshold.
    prefix_before_current_turn = prepare_message_objects_for_llm(messages, compaction=compaction)
    estimated_prefix_tokens = estimate_prefix_tokens(prefix_before_current_turn)
    last_assistant_message_id = find_last_assistant_message_id(messages)
    over_compaction_threshold = last_assistant_message_id is not None and should_compact(
        estimated_prefix_tokens, messages_since_compaction(messages, compaction)
    )

    if over_compaction_threshold and await is_compaction_locked(chat_id, db_session):
        logger.debug(f"Chat {chat_id} is already being compacted; not scheduling another compaction this turn")
        over_compaction_threshold = False

    effective_thinking_level = input_data.thinking_level or CHAT_THINKING_LEVEL
    llm_thinking_kwargs = thinking_kwargs(effective_thinking_level)

    return _AgentTurn(
        chat=chat,
        input_data=input_data,
        llm=llm,
        llm_obj=llm_obj,
        system=system,
        user_message=m_user,
        history=messages,
        formatted_messages=formatted_messages,
        prefix_before_current_turn=prefix_before_current_turn,
        ai_message_defaults=ai_message,
        should_compact=over_compaction_threshold,
        last_assistant_message_id=last_assistant_message_id,
        request_kwargs=llm_thinking_kwargs,
    )


def chat_stream_status(chat: Chat, status: ToolStatus) -> Dict:
    """Stream object reporting a tool call's progress (stream contract v1)."""
    return {
        **chat.client_response(),
        "status": {"id": status.id, "tool": status.tool, "state": status.state, "label": status.label},
    }


@dataclass
class _AgentTurn:
    """Runs the agent loop for one chat turn, streams or returns the answer, and saves it."""

    chat: Chat
    input_data: ChatCreateMessageInput
    llm: BedrockHandler
    llm_obj: LLM
    system: list
    user_message: Message
    history: list[Message]
    formatted_messages: list[dict]
    # formatted_messages without the current turn, for compaction. It can end with a tool_result
    # user message that formatted_messages merged with this turn's user message.
    prefix_before_current_turn: list[dict]
    ai_message_defaults: MessageDefaults
    should_compact: bool
    last_assistant_message_id: int | None
    request_kwargs: dict
    _tool_specs: list[dict] = field(default_factory=list, init=False)
    saved_message: Message | None = field(default=None, init=False)

    async def _events(self) -> AsyncIterator[AgentLoopEvent]:
        """Run the loop and save the turn before passing on LoopFinished.

        No DB session is held across the loop: each tool call opens and commits its own (see
        `_run_tool_call`), so tool writes are kept even if a later model call fails, and the turn's
        tool_call rows are written in another short session once the answer is saved.
        """
        ctx = ToolContext(
            chat=self.chat,
            user_id=self.chat.user_id,
            user_message=self.user_message,
            history=self.history,
            input_data=self.input_data,
            document_uuids=list(self.input_data.document_uuids or []),
        )
        tools = build_available_tools()
        self._tool_specs = [tool.tool_def.spec for tool in tools]
        async for event in run_agent_loop(
            self.llm,
            history=self.formatted_messages,
            tools=tools,
            ctx=ctx,
            should_compact=self.should_compact,
            request_kwargs=self.request_kwargs,
        ):
            if isinstance(event, LoopFinished):
                self.saved_message = self._save(event.result)
                if self.saved_message is not None:
                    await _record_tool_calls(self.saved_message.id, event.result.tool_calls)
            yield event

    async def run(self) -> Message | None:
        """Run the turn to completion and return the saved assistant message."""
        async for _ in self._events():
            pass
        return self.saved_message

    async def stream(self) -> AsyncIterator[str]:
        """Run the turn, yielding stream contract JSON objects, and save the answer at the end."""
        content = ""
        citations: list[dict] = []
        sources = Sources()
        try:
            async for event in self._events():
                if isinstance(event, TextDelta):
                    content += event.text
                    yield self._stream_message(content, citations, sources)
                elif isinstance(event, SourcesUpdated):
                    citations, sources = event.citations, event.sources
                elif isinstance(event, ToolStatus):
                    yield json.dumps(chat_stream_status(self.chat, event), indent=4)
                elif isinstance(event, LoopFinished):
                    yield self._stream_message(event.result.display_text, event.result.citations, event.result.sources)
        except Exception as e:
            logger.exception(f"Agent loop failed for chat {self.chat.id}: {e}")
            yield chat_stream_error_message(
                self.chat,
                e,
                has_documents=bool(self.input_data.document_uuids),
                is_initial_call=self.input_data.initial_call,
            )

    def _stream_message(self, content: str, citations: list[dict], sources: Sources) -> str:
        message = chat_stream_message(
            chat=self.chat,
            message_uuid=self.ai_message_defaults.uuid,
            content=content,
            citations=json.dumps(citations or None),
            sources=sources,
        )
        return json.dumps(message, indent=4)

    def _save(self, result: AgentLoopResult) -> Message | None:
        """Save the turn's usage and answer, and schedule compaction if due.

        Returns:
            The assistant message, or None if it couldn't be saved.
        """
        citation = json.dumps(result.citations or None)
        sources = result.sources.model_dump_json(exclude_none=True)
        message_repo = MessageTable()
        try:
            message_repo.update(
                self.user_message,
                {
                    "tokens": result.input_tokens,
                    "cache_read_tokens": result.cache_read_tokens,
                    "cache_write_tokens": result.cache_write_tokens,
                    "citation": citation,
                    "sources": sources,
                },
            )
        except Exception as e:
            logger.error(f"Failed to update user message ID {self.user_message.id}: {e}")

        try:
            ai_message = message_repo.create(
                {
                    **self.ai_message_defaults.dict(),
                    "parent_message_id": self.user_message.id,
                    "completion_cost": result.completion_cost,
                    "role": RoleEnum.assistant,
                    "content": result.display_text,
                    "raw_messages": result.raw_messages,
                    "tokens": result.output_tokens,
                    "citation": citation,
                    "sources": sources,
                }
            )
        except Exception as e:
            logger.exception(f"Failed to create AI message: {e}")
            return None

        try:
            chat_repo = ChatTable()
            chat_repo.update(chat_repo.get(ai_message.chat_id), {"updated_at": datetime.now()})
        except Exception as e:
            logger.error(f"Failed to update chat timestamp for chat ID {ai_message.chat_id}: {e}")

        logger.info(
            f"AI message created: message_id={ai_message.id}, model_calls={len(result.model_calls)}, "
            f"tool_calls={len(result.tool_calls)}, stop_reason={result.stop_reason}, "
            f"completion_cost={ai_message.completion_cost}"
        )

        if self.should_compact and result.model_calls:
            tools_kwargs = {"tools": self._tool_specs} if self._tool_specs else {}
            schedule_compaction(
                chat_id=self.chat.id,
                llm_obj=self.llm_obj,
                system=self.system,
                formatted_messages=apply_compaction_aware_cache_control(self.prefix_before_current_turn, True),
                up_to_message_id=self.last_assistant_message_id,
                prefix_tokens=result.model_calls[0].total_input_tokens,
                extra_api_kwargs={**self.request_kwargs, **tools_kwargs},
            )
        return ai_message


async def _record_tool_calls(message_id: int, records: list[ToolCallRecord]) -> None:
    """Write one tool_call row per record, in a short session of its own. Failures are logged, not raised."""
    if not records:
        return
    try:
        async with async_db_session() as db_session:
            db_session.add_all(
                ToolCall(
                    message_id=message_id,
                    llm_internal_response_id=record.llm_internal_response_id,
                    iteration=record.iteration,
                    tool_use_id=record.tool_use_id,
                    tool_name=record.tool_name,
                    args=record.args,
                    is_error=record.is_error,
                    duration_ms=record.duration_ms,
                    result_chars=record.result_chars,
                )
                for record in records
            )
    except Exception as e:
        logger.error(f"Failed to record tool calls for message ID {message_id}: {e!r}")


async def _save_chat_documents(db_session: AsyncSession, user_message: Message, rag_request: RagRequest):
    """
    Saves documents referenced in chat message to the database.
    """
    insert_records = [
        {"chat_id": user_message.chat_id, "document_uuid": doc_uuid} for doc_uuid in rag_request.document_uuids
    ]
    await DbOperations.save_records(db_session, ChatDocumentMapping, insert_records)


async def _extend_document_expiry_and_last_used_time(rag_request: RagRequest, db_session: AsyncSession):
    """
    Extends the expiry date of the documents referenced in the rag_request parameter by 90 days from today onwards.
    And updates the last used time of the documents referenced.
    """
    document_uuids = rag_request.document_uuids
    user_id = rag_request.user_id
    await DbOperations.update_document_expiry_and_last_used_time(db_session, document_uuids, user_id)


async def _check_document_access(db_session: AsyncSession, rag_request: RagRequest):
    """
    Checks if the user has access to the specified documents.

    Raises:
        DocumentAccessError: Raised if the user does not have access to the requested documents
    """
    result = await db_session.execute(
        select(cast(Document.uuid, sqlalchemy.String))
        .select_from(DocumentUserMapping)
        .join(Document, Document.id == DocumentUserMapping.document_id)
        .where(
            DocumentUserMapping.user_id == rag_request.user_id,
            Document.uuid.in_(rag_request.document_uuids),
            DocumentUserMapping.deleted_at.is_(None),
            Document.deleted_at.is_(None),
        )
    )
    docs_allowed = result.scalars().all()
    docs_not_allowed = [requested for requested in rag_request.document_uuids if requested not in docs_allowed]
    if docs_not_allowed:
        raise DocumentAccessError(
            "User not allowed for access",
            document_uuids=docs_not_allowed,
        )


async def clean_expired_message_content(db_session: AsyncSession):
    """
    Removes content from messages older than 1 year for data protection compliance.
    This function cleans message content by replacing it with a deletion notice, updating
    the deleted_at timestamp for messages that are older than 365 days, and marks
    associated chats as deleted ONLY if ALL messages in those chats are now deleted.

    Note: We only mark chats as deleted when ALL their messages are deleted to protect
    against scenarios where someone adds a new message to an old chat (e.g. a chat
    started 364 days ago that gets a new message). We don't want such chats to suddenly
    become unavailable.

    Args:
        db_session (AsyncSession): The database session for executing queries.

    Returns:
        dict: Simple data structure containing cleanup results.
    """

    # Calculate cutoff date (1 year ago)
    cutoff_date = datetime.now() - timedelta(days=365)

    # Update messages older than 1 year and get their chat IDs
    stmt = (
        update(Message)
        .where(
            Message.created_at < cutoff_date,
            Message.deleted_at.is_(None),
        )
        .values(
            content=DELETION_NOTICE,
            content_enhanced_with_rag=DELETION_NOTICE,
            raw_messages=None,
            deleted_at=datetime.now(),
        )
        .returning(Message.id, Message.chat_id)
    )

    result = await db_session.execute(stmt)
    cleaned_rows = result.fetchall()
    cleaned_count = len(cleaned_rows)

    # Get unique chat IDs that had messages cleaned
    chat_ids_with_cleaned_messages = {row.chat_id for row in cleaned_rows}

    # Tool arguments can hold user-derived text (e.g. search queries)
    cleaned_message_ids = [row.id for row in cleaned_rows]
    if cleaned_message_ids:
        await db_session.execute(update(ToolCall).where(ToolCall.message_id.in_(cleaned_message_ids)).values(args=None))

    # Only mark chats as deleted if ALL messages in those chats are now deleted
    chats_to_mark_deleted = []
    for chat_id in chat_ids_with_cleaned_messages:
        # Check if there are any remaining undeleted messages in this chat
        remaining_stmt = select(Message.id).where(Message.chat_id == chat_id, Message.deleted_at.is_(None)).limit(1)
        remaining_result = await db_session.execute(remaining_stmt)
        remaining_message = remaining_result.scalar()

        # If no remaining undeleted messages, mark this chat for deletion
        if remaining_message is None:
            chats_to_mark_deleted.append(chat_id)

    # Mark chats as deleted where ALL messages are now deleted
    deleted_chats_count = 0
    if chats_to_mark_deleted:
        chat_stmt = (
            update(Chat)
            .where(Chat.id.in_(chats_to_mark_deleted), Chat.deleted_at.is_(None))
            .values(deleted_at=datetime.now())
        )
        await db_session.execute(chat_stmt)
        deleted_chats_count = len(chats_to_mark_deleted)

    # Don't commit here - let the API layer handle transaction management

    logger.info(
        f"Cleaned content from {cleaned_count} expired messages and marked {deleted_chats_count} "
        "chats as deleted (only chats where ALL messages are deleted)"
    )

    return {
        "cleaned_count": cleaned_count,
        "cleaned_chats": deleted_chats_count,
    }
