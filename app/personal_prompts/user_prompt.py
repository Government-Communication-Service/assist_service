from fastapi import APIRouter, Depends, Response

from app.api import ENDPOINTS, endpoint_defaults
from app.api.paths import ApiPaths
from app.database.table import async_db_session
from app.personal_prompts.lib import (
    delete_existing_user_prompt,
    get_all_user_prompts,
    get_existing_user_prompt,
    no_body_user_prompt_request_data,
    patch_update_existing_user_prompt,
    post_create_user_prompt,
    user_prompt_request_data,
    user_prompts_request_data,
)
from app.personal_prompts.schemas import ItemUserPromptResponse, ListUserPromptResponse

router = APIRouter()


user_prompts_endpoint_defaults = {
    **endpoint_defaults(
        extra_dependencies=[
            ApiPaths.USER_UUID,
        ],
    ),
}


user_prompt_endpoint_defaults = {
    **endpoint_defaults(
        extra_dependencies=[
            ApiPaths.USER_UUID,
            ApiPaths.USER_PROMPT_UUID,
        ],
    ),
}

user_prompt_endpoint_defaults_delete = {
    **endpoint_defaults(
        status_code=204,
        extra_dependencies=[
            ApiPaths.USER_UUID,
            ApiPaths.USER_PROMPT_UUID,
        ],
    ),
}


@router.get(ENDPOINTS.USER_PROMPTS, response_model=ListUserPromptResponse, **endpoint_defaults())
async def get_user_prompts(
    user=ApiPaths.USER_UUID,
) -> ListUserPromptResponse | Response:
    """
    Fetch a user's user prompts by their ID. Expandable down the line to include filters / recent slices.
    """

    async with async_db_session() as db_session:
        return await get_all_user_prompts(db_session=db_session, user=user)


@router.post(ENDPOINTS.USER_PROMPTS, response_model=ItemUserPromptResponse, **user_prompts_endpoint_defaults)
async def create_user_prompt(data=Depends(user_prompts_request_data)) -> ItemUserPromptResponse | Response:
    """
    Create a new user prompt for the user with the given ID.
    """

    async with async_db_session() as db_session:
        return await post_create_user_prompt(db_session=db_session, user_prompt_input=data)


@router.get(ENDPOINTS.USER_PROMPT, response_model=ItemUserPromptResponse, **user_prompt_endpoint_defaults)
async def get_user_prompt(data=Depends(no_body_user_prompt_request_data)) -> ItemUserPromptResponse | Response:
    """
    Get an existing user prompt for the user with the given ID.
    """

    async with async_db_session() as db_session:
        return await get_existing_user_prompt(db_session=db_session, user_prompt_input=data)


@router.patch(ENDPOINTS.USER_PROMPT, response_model=ItemUserPromptResponse, **user_prompt_endpoint_defaults)
async def patch_user_prompt(data=Depends(user_prompt_request_data)) -> ItemUserPromptResponse | Response:
    """
    Update an existing user prompt for the user with the given ID.
    """

    async with async_db_session() as db_session:
        return await patch_update_existing_user_prompt(db_session=db_session, user_prompt_req_data=data)


@router.delete(ENDPOINTS.USER_PROMPT, response_model=None, **user_prompt_endpoint_defaults_delete)
async def delete_user_prompt(data=Depends(no_body_user_prompt_request_data)) -> Response:
    """
    Delete an existing user prompt for the user with the given ID.
    """

    async with async_db_session() as db_session:
        return await delete_existing_user_prompt(db_session=db_session, user_prompt_input=data)
