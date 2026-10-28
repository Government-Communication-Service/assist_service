from fastapi import APIRouter, Depends, Response

from app.api.endpoints import ENDPOINTS
from app.auth.verify_service import (
    verify_and_get_auth_session_from_header,
    verify_and_get_user_from_path_and_header,
    verify_auth_token,
)
from app.database.table import async_db_session
from app.personal_prompts.schemas import ItemUserPromptResponse, ListUserPromptResponse
from app.personal_prompts.service import (
    delete_existing_user_prompt,
    get_all_user_prompts,
    get_existing_user_prompt,
    no_body_user_prompt_request_data,
    patch_update_existing_user_prompt,
    post_create_user_prompt,
    user_prompt_request_data,
    user_prompts_request_data,
)
from app.personal_prompts.utils import verify_and_get_user_prompt_by_uuid

router = APIRouter()


@router.get(
    path=ENDPOINTS.USER_PROMPTS,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_auth_session_from_header),
    ],
    response_model=ListUserPromptResponse,
)
async def get_user_prompts(
    user=Depends(verify_and_get_user_from_path_and_header),
) -> ListUserPromptResponse | Response:
    """
    Fetch a user's user prompts by their ID. Expandable down the line to include filters / recent slices.
    """

    async with async_db_session() as db_session:
        return await get_all_user_prompts(db_session=db_session, user=user)


@router.post(
    path=ENDPOINTS.USER_PROMPTS,
    response_model=ItemUserPromptResponse,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
    ],
)
async def create_user_prompt(data=Depends(user_prompts_request_data)) -> ItemUserPromptResponse | Response:
    """
    Create a new user prompt for the user with the given ID.
    """

    async with async_db_session() as db_session:
        return await post_create_user_prompt(db_session=db_session, user_prompt_input=data)


@router.get(
    path=ENDPOINTS.USER_PROMPT,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
        Depends(verify_and_get_user_prompt_by_uuid),
    ],
    response_model=ItemUserPromptResponse,
)
async def get_user_prompt(data=Depends(no_body_user_prompt_request_data)) -> ItemUserPromptResponse | Response:
    """
    Get an existing user prompt for the user with the given ID.
    """

    async with async_db_session() as db_session:
        return await get_existing_user_prompt(db_session=db_session, user_prompt_input=data)


@router.patch(
    path=ENDPOINTS.USER_PROMPT,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
        Depends(verify_and_get_user_prompt_by_uuid),
    ],
    response_model=ItemUserPromptResponse,
)
async def patch_user_prompt(data=Depends(user_prompt_request_data)) -> ItemUserPromptResponse | Response:
    """
    Update an existing user prompt for the user with the given ID.
    """

    async with async_db_session() as db_session:
        return await patch_update_existing_user_prompt(db_session=db_session, user_prompt_req_data=data)


@router.delete(
    path=ENDPOINTS.USER_PROMPT,
    dependencies=[
        Depends(verify_auth_token),
        Depends(verify_and_get_user_from_path_and_header),
        Depends(verify_and_get_auth_session_from_header),
        Depends(verify_and_get_user_prompt_by_uuid),
    ],
    status_code=204,
    response_model=None,
)
async def delete_user_prompt(data=Depends(no_body_user_prompt_request_data)) -> Response:
    """
    Delete an existing user prompt for the user with the given ID.
    """

    async with async_db_session() as db_session:
        return await delete_existing_user_prompt(db_session=db_session, user_prompt_input=data)
