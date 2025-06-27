# ruff: noqa: B008
import logging
from typing import Any

from fastapi import Body, Depends
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.status import (
    HTTP_204_NO_CONTENT,
    HTTP_401_UNAUTHORIZED,
    HTTP_404_NOT_FOUND,
    HTTP_502_BAD_GATEWAY,
)

from app.api import api_wrapper, get_current_session
from app.api.paths import ApiPaths
from app.database.db_operations import DbOperations
from app.database.models import User, UserPrompt
from app.personal_prompts.schemas import (
    ItemUserPromptResponse,
    ItemUserPromptResponseWithDescription,
    ListUserPromptResponse,
    NoBodyUserPromptRequestData,
    UserPromptInput,
    UserPromptRequestBody,
    UserPromptRequestData,
)

logger = logging.getLogger(__name__)


def user_prompts_request_data(
    user: User = ApiPaths.USER_UUID,
    session: Any = Depends(get_current_session),
    data: UserPromptRequestBody = Body(...),
) -> UserPromptRequestData:
    data_dict = data.model_dump()

    return UserPromptRequestData(
        user_id=user.id,
        user_uuid=user.uuid,
        auth_session_id=session.id,
        uuid=None,
        **data_dict,
    )


def user_prompt_request_data(
    user: User = ApiPaths.USER_UUID,
    user_prompt: UserPrompt = ApiPaths.USER_PROMPT_UUID,
    session: Any = Depends(get_current_session),
    data: UserPromptRequestBody = Body(...),
) -> UserPromptRequestData:
    data_dict = data.model_dump()

    return UserPromptRequestData(
        user_id=user.id,
        user_uuid=user.uuid,
        auth_session_id=session.id,
        uuid=user_prompt.uuid,
        **data_dict,
    )


def no_body_user_prompt_request_data(
    user: User = ApiPaths.USER_UUID,
    user_prompt: UserPrompt = ApiPaths.USER_PROMPT_UUID,
    session: Any = Depends(get_current_session),
) -> NoBodyUserPromptRequestData:
    return NoBodyUserPromptRequestData(user_id=user.id, auth_session_id=session.id, uuid=user_prompt.uuid)


@api_wrapper(task="get_all_user_prompts")
async def get_all_user_prompts(db_session: AsyncSession, user: User) -> ListUserPromptResponse:
    """
    Fetch a user's user prompts by their ID. Expandable down the line to include filters / recent slices.
    """

    user_prompts = await DbOperations.get_user_prompts(db_session=db_session, user=user)
    user_prompts_list = [
        ItemUserPromptResponseWithDescription(
            id=prompt.id,
            uuid=prompt.uuid,
            user_id=prompt.user_id,
            title=prompt.title,
            description=(
                prompt.description
                if prompt.description
                else " ".join(prompt.content.split()[:10]) + ("..." if len(prompt.content.split()) > 10 else "")
            ),
            content="",
            created_at=prompt.created_at,
            updated_at=prompt.updated_at,
            deleted_at=prompt.deleted_at,
        )
        for prompt in user_prompts
    ]
    return ListUserPromptResponse(user_prompts=user_prompts_list)


@api_wrapper(task="post_create_user_prompt")
async def post_create_user_prompt(
    db_session: AsyncSession,
    user_prompt_input: UserPromptRequestData,
) -> ItemUserPromptResponse:
    """
    Create a new user prompt by their ID.
    """

    user_prompt = await DbOperations.insert_single_user_prompt(
        db_session=db_session,
        user_id=user_prompt_input.user_id,
        title=user_prompt_input.title,
        content=user_prompt_input.content,
    )
    return ItemUserPromptResponse(**user_prompt.client_response())


@api_wrapper(task="get_user_prompt")
async def get_existing_user_prompt(
    db_session: AsyncSession,
    user_prompt_input: UserPromptInput,
) -> ItemUserPromptResponse | Response:
    """
    Get an existing user prompt with the given ID.
    """

    user_prompt = await DbOperations.get_single_user_prompt(
        db_session=db_session,
        user_id=user_prompt_input.user_id,
        user_prompt_uuid=user_prompt_input.uuid,
    )

    if not user_prompt:
        return Response(status_code=HTTP_404_NOT_FOUND)

    if user_prompt_input.user_id != user_prompt.user_id:
        return Response(status_code=HTTP_401_UNAUTHORIZED)

    return ItemUserPromptResponse(**user_prompt.client_response())


@api_wrapper(task="update_user_prompt")
async def patch_update_existing_user_prompt(
    db_session: AsyncSession,
    user_prompt_req_data: UserPromptRequestData,
) -> ItemUserPromptResponse | Response:
    """
    Update an existing user prompt by post UUID.
    """

    user_prompt = await DbOperations.get_single_user_prompt(
        db_session=db_session,
        user_id=user_prompt_req_data.user_id,
        user_prompt_uuid=user_prompt_req_data.uuid,
    )

    if not user_prompt:
        return Response(status_code=HTTP_404_NOT_FOUND)

    try:
        updated_user_prompt = await DbOperations.update_single_user_prompt(
            db_session=db_session,
            user_id=user_prompt_req_data.user_id,
            user_prompt_uuid=user_prompt_req_data.uuid,
            title=user_prompt_req_data.title,
            content=user_prompt_req_data.content,
        )
    except Exception as e:
        logger.exception(msg=f"Could not update user prompt: {user_prompt.uuid} ", exc_info=e)
        return Response(status_code=HTTP_502_BAD_GATEWAY)
    else:
        return ItemUserPromptResponse(**updated_user_prompt.client_response())


@api_wrapper(task="delete_user_prompt")
async def delete_existing_user_prompt(
    db_session: AsyncSession,
    user_prompt_input: NoBodyUserPromptRequestData,
) -> Response:
    """
    Delete an existing user prompt for the user with the given ID.
    """

    user_prompt = await DbOperations.get_single_user_prompt(
        db_session=db_session,
        user_id=user_prompt_input.user_id,
        user_prompt_uuid=user_prompt_input.uuid,
    )

    if not user_prompt:
        return Response(status_code=HTTP_404_NOT_FOUND)

    await DbOperations.delete_single_user_prompt(
        db_session=db_session,
        user_id=user_prompt_input.user_id,
        user_prompt_uuid=user_prompt_input.uuid,
    )

    return Response(status_code=HTTP_204_NO_CONTENT)
