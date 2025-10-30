# ruff: noqa: B008
import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

import sqlalchemy
from anthropic.types import TextBlock
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
from app.bedrock.schemas import LLMResponse
from app.bedrock.service import llm_transaction
from app.central_guidance.schemas import RagRequest
from app.central_guidance.service_rag import search_central_guidance
from app.chat.actions import get_response_system_prompt
from app.chat.config import SLEEP_TIME_MESSAGE_DELETION
from app.chat.constants import DELETION_NOTICE
from app.chat.schemas import (
    CentralGuidanceSource,
    ChatBasicResponse,
    ChatCreateInput,
    ChatCreateMessageInput,
    ChatPost,
    ChatRequestData,
    ChatSuccessResponse,
    ChatTitleRequest,
    ChatWithAllMessages,
    ChatWithLatestMessage,
    DocumentAccessError,
    DocumentSchema,
    GovUkSearchSource,
    MessageBasicResponse,
    MessageDefaults,
    RoleEnum,
    SmartTargetsSource,
    Sources,
    UserChatsResponse,
    UserDocumentSource,
)
from app.chat.utils import prepare_message_objects_for_llm
from app.compaction.service import trigger_compaction_if_needed
from app.config import (
    LLM_CHAT_RESPONSE_MODEL,
    LLM_CHAT_TITLE_MODEL,
)
from app.database.db_operations import DbOperations
from app.database.models import (
    LLM,
    Chat,
    ChatDocumentMapping,
    Document,
    DocumentUserMapping,
    Message,
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
from app.document_upload.service import search_uploaded_documents
from app.error_messages import ErrorMessages
from app.gov_uk_search.service import assess_if_next_message_should_use_gov_uk_search, enhance_user_prompt
from app.logs.logs_handler import logger
from app.smart_targets.service import SmartTargetsService


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


def chat_save_llm_output(
    ai_message_defaults: MessageDefaults,
    llm_response: LLMResponse,
    user_message: Message,
    llm: LLM,
) -> Optional[Message]:
    try:
        transaction = llm_transaction(llm, llm_response)
        logger.debug(
            f"LLM transaction created: input_cost={transaction.input_cost}, output_cost={transaction.output_cost}",
        )

        message_repo = MessageTable()
        try:
            message_repo.update(
                user_message,
                {
                    "tokens": llm_response.input_tokens,
                },
            )
            logger.info(
                f"User message updated successfully. Tokens: {llm_response.input_tokens}, "
                f"Completion cost: {transaction.input_cost}",
            )
        except Exception as e:
            logger.error(f"Failed to update user message ID {user_message.id}: {str(e)}")
            # Continue execution, as this is not a critical error

        try:
            content_list = [text_block.text for text_block in llm_response.content if text_block.type == "text"]
            if content_list:
                content = ", ".join(content_list)
            else:
                content = ""
            ai_message = message_repo.create(
                {
                    **ai_message_defaults.dict(),
                    "parent_message_id": user_message.id,
                    "completion_cost": transaction.completion_cost,
                    "role": RoleEnum.assistant,
                    "content": content,
                    "tokens": llm_response.output_tokens,
                    "citation": user_message.citation,
                    "sources": user_message.sources,
                },
            )

            # Update chat updated_at timestamp when AI message is created
            try:
                chat_repo = ChatTable()
                chat_obj = chat_repo.get(ai_message.chat_id)
                chat_repo.update(chat_obj, {"updated_at": datetime.now()})
            except Exception as e:
                # Log error but continue execution, as not a critical error
                logger.error(f"Failed to update chat timestamp for chat ID {ai_message.chat_id}: {str(e)}")

            logger.info(
                "AI message created succesfully: "
                f"message_id={ai_message.id}, "
                f"tokens={ai_message.tokens}, "
                f"completion_cost={ai_message.completion_cost}",
            )
        except Exception as e:
            logger.exception(f"Failed to create AI message: {e}")
            return None
        return ai_message

    except Exception as e:
        logger.error(f"An unexpected error occurred in chat_save_llm_output: {str(e)}")
        return None


def get_user_groups(
    user_groups_string=Header(
        default=os.getenv("TEST_USER_GROUPS", ""),
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
    Args:
        chat (Chat): The Chat object containing the chat information (e.g., `id`).
        db_session (AsyncSession): The asynchronous database session to be used for the query.
        request_input (Dict): The dictionary containing the initial request input.

    Returns:
        Dict: The updated request input dictionary, with "document_uuids" key if there are chat
        documents in the database.
    """
    chat_document_mappings = await DbOperations.fetch_undeleted_chat_documents(db_session, chat.user_id, chat.id)
    if chat_document_mappings:
        document_uuids = [str(doc.uuid) for doc in chat_document_mappings]
        request_input["document_uuids"] = document_uuids
    return request_input


async def chat_add_message_stream(chat: Chat, data):
    chat_message = data.to_dict()
    # check if there are chat documents, then fetch and append them to the request
    async with async_db_session() as db_session:
        chat_message = await _chat_message_with_documents(chat, db_session, chat_message)
        response = await chat_create_message(chat, ChatCreateMessageInput(**chat_message, stream=True), db_session)
        return StreamingResponse(response, media_type="text/event-stream")


async def chat_add_message(chat: Chat, data):
    request_input = data.to_dict()
    # check if there are chat documents, then fetch and append them to the request
    async with async_db_session() as db_session:
        request_input = await _chat_message_with_documents(chat, db_session, request_input)
        message = await chat_create_message(chat, ChatCreateMessageInput(**request_input), db_session)
        return ChatWithLatestMessage(
            **chat.client_response(),
            message=message.client_response(),
        )


async def update_chat_title(db_session: AsyncSession, chat: Chat, data) -> ChatSuccessResponse:
    # update chat title
    title = await chat_create_title(ChatTitleRequest(**data.to_dict()))
    chat_result = await DbOperations.chat_update_title(db_session, chat, title)

    return ChatSuccessResponse(**chat_result.client_response())


async def patch_chat_title(db_session: AsyncSession, chat: Chat, title) -> ChatSuccessResponse:
    """
    Updates the title of a chat.

    Args:
        db_session (AsyncSession): The active database session for performing the update.
        chat (Chat): The chat object to be updated.
        title (str): The new title to be assigned to the chat.

    Returns:
        ChatSuccessResponse: Response object containing:
            - uuid: The chat's unique identifier
            - created_at: Original creation timestamp
            - updated_at: Last update timestamp
            - title: The updated chat title
            - status: Success status
            - status_message: Success message

    Note:
        This method is wrapped with api_wrapper decorator for consistent error handling
        and uses existing DbOperations.chat_update_title for the actual database update.
    """
    chat_result = await DbOperations.chat_update_title(db_session, chat, title)

    return ChatSuccessResponse(**chat_result.client_response())


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
            from_open_chat=chat.from_open_chat,
            use_rag=chat.use_rag,
            use_gov_uk_search_api=chat.use_gov_uk_search_api,
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
        logger.info(f"{chat_response=}")
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

        message = await chat_create_message(
            chat=chat_obj,
            input_data=ChatCreateMessageInput(**input_data.to_dict(), initial_call=True),
            db_session=db_session,
        )
        await db_session.commit()

    if input_data.stream:
        return message

    return ChatWithLatestMessage(**chat_obj.dict(), message=message.dict())


async def chat_create_title(data: ChatTitleRequest):
    try:
        system_prompt_title = """The assistant is a title generator called Title Bot. \
Title Bot creates short titles with a maximum of 5 words. \
Title Bot creates titles that are useful for identifying the subject of the provided human query. \
The human query is provided between XML tags as shown: \
<human-query>This is an example message from the human.</human-query>. \
When responding, Title Bot only provides the title. \
Title Bot does not provide ANY chain of thought in it's response. Title Bot ONLY provides the title. \
Title Bot formats it's response using title case. Title Bot does not use a full stop at the end of the title. \
Title Bot does not enclose the title in quotes. \
If Title Bot does not have enough information to generate a title, Title Bot gives it's best attempt. \
The next message received is the human's query.
    """

        logger.debug(f"Constructed title_system: {system_prompt_title}")
        llm_obj = LLMTable().get_by_model(LLM_CHAT_TITLE_MODEL)
        chat = BedrockHandler(system=system_prompt_title, mode=RunMode.ASYNC, llm=llm_obj)

        user_query_for_title_generation = (
            data.query if len(data.query) < 200 else data.query[0:100] + data.query[-100:-1]
        )
        logger.debug(f"Query extract for title generation: {user_query_for_title_generation}")

        formatted_user_query_for_title_generation = f"<human-query>{user_query_for_title_generation}</human-query>"
        messages = chat.format_content_for_chat_title(formatted_user_query_for_title_generation)

        result = await chat.create_chat_title(messages)

        logger.debug(f"Raw LLM result: {result.content}")
        if isinstance(result.content, list):
            content = [m.text for m in result.content if isinstance(m, TextBlock)]
            if len(content) > 0:
                title = content[0].split("\n")[0]
            else:
                title = ""
        else:
            title = result.content

        if len(title) > 255:
            logger.warning(f"Title exceeds 255 characters. Truncating: {title}")
            title = title[:252] + "..."
            logger.debug(f"Truncated title: {title}")

        logger.info(f"Chat title created: {title}")
        return title
    except Exception as error:
        logger.error(f"Error in chat_create_title: {str(error)}", exc_info=True)
        raise Exception(ErrorMessages.CHAT_TITLE_NOT_CREATED, error) from error


async def chat_create_message(chat: Chat, input_data: ChatCreateMessageInput, db_session: AsyncSession):
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

    m_user = message_repo.create(
        {
            "content": input_data.query,
            "role": RoleEnum.user,
            "parent_message_id": parent_message_id,
            **MessageDefaults(**message_defaults).dict(),
        }
    )
    all_messages_pre_retrieval: list[Message] = messages + [m_user]

    ai_message = MessageDefaults(**message_defaults)

    system = await get_response_system_prompt(db_session)

    if input_data.user_group_ids:
        chat_user_group_mapping(m_user, input_data.user_group_ids)

    llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC, system=system)

    prompt_segment_gov_uk_search = None
    citations_gov_uk_search = []
    gov_uk_search_wrapped_documents = None

    # Default to None so it writes as Null into the database if the RAG errors out of the try block.
    query_enhanced_with_rag = None
    citations = None
    sources = Sources()
    prompt_segment_central_guidance = None
    prompt_segment_document_upload = None
    prompt_segment_smart_targets = None
    rag_request = RagRequest(
        use_central_rag=input_data.use_rag,
        user_id=chat.user_id,
        query=input_data.query,
        document_uuids=input_data.document_uuids,
    )

    # check if there are documents referenced in the chat request
    if rag_request.document_uuids and input_data.initial_call:
        await _check_document_access(db_session, rag_request)
        await _save_chat_documents(db_session, m_user, rag_request)

    # extend document expiry time and update last_used time
    if rag_request.document_uuids:
        await _extend_document_expiry_and_last_used_time(rag_request, db_session)

    ## Condition for running enhance_user_prompt
    # IF this is the first message
    # AND the user has ticked 'use_gov_uk_search_api'
    # THEN always search GOV.UK
    if input_data.initial_call:
        # logger.info("### --- Fist message trigger --- ###")
        should_enhance = input_data.use_gov_uk_search_api or input_data.enable_web_browsing
    # ELSE for subsequent messages, let the LLM decide if GOV.UK Search is appropriated
    else:
        # logger.info("### --- Followup message trigger --- ###")
        llm_assesment_if_we_should_use_gov_uk_search_again = await assess_if_next_message_should_use_gov_uk_search(
            messages=messages,
            new_user_message_content=input_data.query,
            new_user_message_id=m_user.id,
            db_session=db_session,
        )
        if llm_assesment_if_we_should_use_gov_uk_search_again is True:
            should_enhance = True
        else:
            should_enhance = False

    tasks = []
    enhance_task = None
    search_central_guidance_task = None
    search_uploaded_documents_task = None

    # Condition for searching GOV.UK
    if should_enhance:
        enhance_task = asyncio.create_task(
            enhance_user_prompt(chat=chat, input_data=input_data, m_user_id=m_user.id, db_session=db_session)
        )
        tasks.append(enhance_task)

    # Condition for searching central guidance
    if input_data.use_rag:
        search_central_guidance_task = asyncio.create_task(
            search_central_guidance(input_data.query, m_user.id, db_session)
        )
        tasks.append(search_central_guidance_task)

    # Condition for searching uploaded documents
    if rag_request.document_uuids:
        search_uploaded_documents_task = asyncio.create_task(search_uploaded_documents(rag_request, m_user, db_session))
        tasks.append(search_uploaded_documents_task)

    # Condition for using Smart Targets
    smart_targets_task = None
    if input_data.use_smart_targets:
        smart_targets_task = asyncio.create_task(
            SmartTargetsService().use_smart_targets_tool(messages=all_messages_pre_retrieval)
        )
        tasks.append(smart_targets_task)

    # Run tasks concurrently if any were created
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, checking for exceptions
        enhance_result_index = tasks.index(enhance_task) if enhance_task else -1
        search_central_guidance_result_index = (
            tasks.index(search_central_guidance_task) if search_central_guidance_task else -1
        )
        search_uploaded_documents_result_index = (
            tasks.index(search_uploaded_documents_task) if search_uploaded_documents_task else -1
        )
        smart_targets_result_index = tasks.index(smart_targets_task) if smart_targets_task else -1

        if enhance_task and enhance_result_index != -1:
            enhance_res = results[enhance_result_index]
            if isinstance(enhance_res, Exception):
                logger.exception(
                    f"Error in enhance_user_prompt - {type(enhance_res).__name__}: {str(enhance_res)}",
                    exc_info=enhance_res,
                )
            else:
                gov_uk_search_wrapped_documents, citations_gov_uk_search = enhance_res
                prompt_segment_gov_uk_search = (
                    f"<results-from-gov-uk-search>\n{gov_uk_search_wrapped_documents}\n</results-from-gov-uk-search>"
                )
                if citations is None:
                    citations = citations_gov_uk_search
                else:
                    citations.extend(citations_gov_uk_search)
                sources.gov_uk_search_sources = [
                    GovUkSearchSource(pretty_name=c["docname"], url=c["docurl"]) for c in citations_gov_uk_search
                ]

        if search_central_guidance_task and search_central_guidance_result_index != -1:
            search_central_guidance_result = results[search_central_guidance_result_index]
            if isinstance(search_central_guidance_result, Exception):
                logger.exception(
                    f"Error when searching central guidance - "
                    f"{type(search_central_guidance_result).__name__}: {str(search_central_guidance_result)}",
                    exc_info=search_central_guidance_result,
                )
            else:
                prompt_segment_central_guidance, citations_central_guidance = search_central_guidance_result
                if citations is None:
                    citations = citations_central_guidance
                else:
                    citations.extend(citations_central_guidance)
                sources.central_guidance_sources = [
                    CentralGuidanceSource(pretty_name=c["docname"], url=c["docurl"]) for c in citations_central_guidance
                ]

        if search_uploaded_documents_task and search_uploaded_documents_result_index != -1:
            search_uploaded_documents_result = results[search_uploaded_documents_result_index]
            if isinstance(search_uploaded_documents_result, Exception):
                logger.exception(
                    f"Error when searching user-uploaded documents - "
                    f"{type(search_uploaded_documents_result).__name__}: {str(search_uploaded_documents_result)}",
                    exc_info=search_uploaded_documents_result,
                )
            else:
                prompt_segment_document_upload, citations_uploaded_documents = search_uploaded_documents_result
                if citations is None:
                    citations = citations_uploaded_documents
                else:
                    citations.extend(citations_uploaded_documents)
                sources.user_document_sources = [
                    UserDocumentSource(pretty_name=c["docname"], url=c["docurl"]) for c in citations_uploaded_documents
                ]

        if smart_targets_task and smart_targets_result_index != -1:
            smart_targets_result = results[smart_targets_result_index]
            if isinstance(smart_targets_result, Exception):
                logger.exception(
                    f"Error when using Smart Targets tool - "
                    f"{type(smart_targets_result).__name__}: {str(smart_targets_result)}",
                    exc_info=smart_targets_result,
                )
            elif smart_targets_result is None:
                prompt_segment_smart_targets = None
            else:
                context, citations_smart_targets = smart_targets_result
                prompt_segment_smart_targets = (
                    f"<results-from-smart-targets-tool>\n{context}\n</results-from-smart-targets-tool>"
                )
                # Smart Targets results are handled separately on the frontend and not handled by the 'citations' array
                sources.smart_targets_sources = [
                    SmartTargetsSource(pretty_name=c["docname"], url=c["docurl"]) for c in citations_smart_targets
                ]

    # Compile the final query to be passed to the LLM
    query_parts = [input_data.query]
    # Add each segment only if it has content
    if prompt_segment_gov_uk_search:
        query_parts.append(prompt_segment_gov_uk_search)
    if prompt_segment_central_guidance:
        query_parts.append(prompt_segment_central_guidance)
    if prompt_segment_document_upload:
        query_parts.append(prompt_segment_document_upload)
    if prompt_segment_smart_targets:
        query_parts.append(prompt_segment_smart_targets)

    query_enhanced_with_rag = "\n\n".join(query_parts)

    # Check if compaction is needed before generating the final message
    compaction_triggered = False
    try:
        compaction_triggered = await trigger_compaction_if_needed(chat_id, query_enhanced_with_rag, db_session)
        if compaction_triggered:
            logger.info(f"Compaction triggered for chat {chat_id} before message generation")
            # Reload messages from database to get updated summaries
            messages = message_repo.get_by_chat(chat_id)
    except Exception as e:
        logger.exception(f"Error during compaction check for chat {chat_id}: {e}")

    m_user = message_repo.update(
        m_user,
        {
            "content_enhanced_with_rag": query_enhanced_with_rag,
            "citation": json.dumps(citations),
            "sources": sources.model_dump_json(exclude_none=True),
        },
    )

    all_messages_post_retrieval = messages + [m_user]

    formatted_messages = prepare_message_objects_for_llm(all_messages_post_retrieval)

    def on_complete(response):
        formatted_response = llm.format_response(response)
        result = chat_save_llm_output(
            ai_message_defaults=ai_message,
            user_message=m_user,
            llm=llm_obj,
            llm_response=formatted_response,
        )
        return result

    if input_data.stream:

        def parse_data(text, citations):
            return chat_stream_message(
                chat=chat,
                message_uuid=ai_message.uuid,
                content=text,
                citations=citations,
                sources=sources,
            )

        def on_error(ex: Exception):
            has_documents = bool(input_data.document_uuids)
            return chat_stream_error_message(
                chat, ex, has_documents=has_documents, is_initial_call=input_data.initial_call
            )

        stream_res = llm.stream(
            formatted_messages,
            on_error=on_error,
            user_message=m_user,
            system=system,
            parse_data=parse_data,
            on_complete=on_complete,
        )

        return stream_res

    response = await llm.invoke_async(formatted_messages)
    result = on_complete(response)

    return result


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
    # logger.info(
    #     "Checking document access for user %s for documents %s", rag_request.user_id, rag_request.document_uuids
    # )
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
        .values(content=DELETION_NOTICE, content_enhanced_with_rag=DELETION_NOTICE, deleted_at=datetime.now())
        .returning(Message.id, Message.chat_id)
    )

    result = await db_session.execute(stmt)
    cleaned_rows = result.fetchall()
    cleaned_count = len(cleaned_rows)

    # Get unique chat IDs that had messages cleaned
    chat_ids_with_cleaned_messages = {row.chat_id for row in cleaned_rows}

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


async def schedule_expired_messages_deletion():
    """
    Schedules the periodic execution of the expired message deletion process.
    The process runs every hour and checks if there are expired documents to delete from database and opensearch
    """
    while True:
        try:
            logger.info("Running scheduled expired messages deletion process")
            async with async_db_session() as db_session:
                await clean_expired_message_content(db_session)
                await db_session.commit()
                logger.info("Successfully committed expired messages deletion changes")
        except Exception as e:
            logger.exception("An error occurred during expired messages deletion: %s", e)
        logger.info("Sleeping for %s seconds before the next message deletion run.", SLEEP_TIME_MESSAGE_DELETION)
        await asyncio.sleep(SLEEP_TIME_MESSAGE_DELETION)
