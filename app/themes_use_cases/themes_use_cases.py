from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends

from app.api import ENDPOINTS, endpoint_defaults
from app.auth.auth_token import auth_token_validator_no_user
from app.chat.schemas import (
    PrebuiltPromptsResponse,
    SuccessResponse,
    ThemeResponse,
    ThemesResponse,
    UseCaseResponse,
    UseCasesResponse,
)
from app.database.table import async_db_session
from app.themes_use_cases.lib import (
    create_theme,
    create_use_case,
    fetch_all_prompts,
    fetch_theme,
    fetch_themes,
    fetch_use_case,
    fetch_use_case_without_requiring_theme,
    fetch_use_cases,
    soft_delete_theme,
    soft_delete_use_case,
    update_theme,
    update_use_case,
    upload_prompts_in_bulk,
)
from app.themes_use_cases.schemas import PrebuiltPrompt, ThemeInput, UseCaseInputPost, UseCaseInputPut

router = APIRouter()


### Endpoint definitions ###
@router.post(
    ENDPOINTS.PROMPTS_THEMES,
    dependencies=[Depends(auth_token_validator_no_user)],
    response_model=ThemeResponse,
)
async def post_theme(theme_input: ThemeInput) -> ThemeResponse:
    """Creates or revives a theme.

    Args:
        theme_input (ThemeInput): The theme data containing title, subtitle and position.

    Returns:
        ThemeResponse: The created or revived theme details.
    """
    async with async_db_session() as db_session:
        return await create_theme(db_session=db_session, theme_input=theme_input)


@router.get(
    ENDPOINTS.PROMPTS_THEME,
    dependencies=[Depends(auth_token_validator_no_user)],
    response_model=ThemeResponse,
)
async def get_theme(theme_uuid: UUID) -> ThemeResponse:
    """Gets a theme by UUID.

    Args:
        theme_uuid (UUID): The theme's UUID.

    Returns:
        ThemeResponse: The theme details.
    """
    async with async_db_session() as db_session:
        return await fetch_theme(db_session=db_session, theme_uuid=theme_uuid)


@router.get(ENDPOINTS.PROMPTS_THEMES, **endpoint_defaults(), response_model=ThemesResponse)
async def get_themes() -> ThemesResponse:
    """Gets all themes.

    Returns:
        ThemesResponse: List of all themes.
    """
    async with async_db_session() as db_session:
        return await fetch_themes(db_session=db_session)


@router.put(
    ENDPOINTS.PROMPTS_THEME,
    dependencies=[Depends(auth_token_validator_no_user)],
    response_model=ThemeResponse,
)
async def put_theme(theme_uuid: UUID, theme_input: ThemeInput) -> ThemeResponse:
    """Updates a theme.

    Args:
        theme_uuid (UUID): The theme's UUID.
        theme_input (ThemeInput): The updated theme data.

    Returns:
        ThemeResponse: The updated theme details.
    """
    async with async_db_session() as db_session:
        return await update_theme(db_session=db_session, theme_uuid=theme_uuid, theme_input=theme_input)


@router.delete(
    ENDPOINTS.PROMPTS_THEME,
    dependencies=[Depends(auth_token_validator_no_user)],
    response_model=SuccessResponse,
)
async def delete_theme(theme_uuid: UUID) -> SuccessResponse:
    """Soft deletes a theme.

    Args:
        theme_uuid (UUID): The theme's UUID.

    Returns:
        SuccessResponse: Success response.
    """
    async with async_db_session() as db_session:
        return await soft_delete_theme(db_session=db_session, theme_uuid=theme_uuid)


@router.post(
    ENDPOINTS.PROMPTS_THEMES_USE_CASES,
    dependencies=[Depends(auth_token_validator_no_user)],
    response_model=UseCaseResponse,
)
async def post_use_case(theme_uuid: UUID, use_case_input: UseCaseInputPost) -> UseCaseResponse:
    """Creates or revives a use case under a theme.

    Args:
        theme_uuid (UUID): The parent theme's UUID.
        use_case_input (UseCaseInputPost): The use case data.

    Returns:
        UseCaseResponse: The created or revived use case details.
    """
    async with async_db_session() as db_session:
        return await create_use_case(db_session=db_session, theme_uuid=theme_uuid, use_case_input=use_case_input)


@router.get(ENDPOINTS.PROMPTS_THEMES_USE_CASE, **endpoint_defaults())
async def get_use_case(theme_uuid: UUID, use_case_uuid: UUID) -> UseCaseResponse:
    """Gets a use case under a specific theme.

    Args:
        theme_uuid (UUID): The parent theme's UUID.
        use_case_uuid (UUID): The use case's UUID.

    Returns:
        UseCaseResponse: The use case details.
    """
    async with async_db_session() as db_session:
        return await fetch_use_case(db_session=db_session, theme_uuid=theme_uuid, use_case_uuid=use_case_uuid)


@router.get(ENDPOINTS.PROMPTS_USE_CASE, **endpoint_defaults())
async def get_use_case_without_requiring_theme(use_case_uuid: UUID) -> UseCaseResponse:
    """Gets a use case without requiring its parent theme.

    Args:
        use_case_uuid (UUID): The use case's UUID.

    Returns:
        UseCaseResponse: The use case details.
    """
    async with async_db_session() as db_session:
        return await fetch_use_case_without_requiring_theme(db_session=db_session, use_case_uuid=use_case_uuid)


@router.put(
    ENDPOINTS.PROMPTS_THEMES_USE_CASE,
    dependencies=[Depends(auth_token_validator_no_user)],
    response_model=UseCaseResponse,
)
async def put_use_case(theme_uuid: UUID, use_case_uuid: UUID, use_case_input: UseCaseInputPut) -> UseCaseResponse:
    """Updates a use case.

    Args:
        theme_uuid (UUID): The parent theme's UUID.
        use_case_uuid (UUID): The use case's UUID.
        use_case_input (UseCaseInputPut): The updated use case data.

    Returns:
        UseCaseResponse: The updated use case details.
    """
    async with async_db_session() as db_session:
        return await update_use_case(
            db_session=db_session, theme_uuid=theme_uuid, use_case_uuid=use_case_uuid, use_case_input=use_case_input
        )


@router.delete(
    ENDPOINTS.PROMPTS_THEMES_USE_CASE,
    dependencies=[Depends(auth_token_validator_no_user)],
    response_model=SuccessResponse,
)
async def delete_use_case(theme_uuid: UUID, use_case_uuid: UUID) -> SuccessResponse:
    """Soft deletes a use case.

    Args:
        theme_uuid (UUID): The parent theme's UUID.
        use_case_uuid (UUID): The use case's UUID.

    Returns:
        SuccessResponse: Success response.
    """
    async with async_db_session() as db_session:
        return await soft_delete_use_case(db_session=db_session, theme_uuid=theme_uuid, use_case_uuid=use_case_uuid)


@router.get(
    ENDPOINTS.PROMPTS_THEMES_USE_CASES,
    **endpoint_defaults(),
    response_model=UseCasesResponse,
)
async def get_use_cases_by_theme(theme_uuid: UUID) -> UseCasesResponse:
    """Gets all use cases under a theme.

    Args:
        theme_uuid (str): The parent theme's UUID.

    Returns:
        UseCasesResponse: List of use cases under the theme.
    """
    async with async_db_session() as db_session:
        return await fetch_use_cases(db_session=db_session, theme_uuid=theme_uuid)


# Bulk get prompts (both themes and use cases - may be used when preparing for a bulk upload)
@router.get(
    ENDPOINTS.PROMPTS_BULK,
    dependencies=[Depends(auth_token_validator_no_user)],
    response_model=PrebuiltPromptsResponse,
)
async def bulk_get_prompts() -> PrebuiltPromptsResponse:
    """Gets all prompts (themes and use cases) for bulk upload preparation.

    Returns:
        PrebuiltPromptsResponse: List of all prompts in the database.
    """
    async with async_db_session() as db_session:
        return await fetch_all_prompts(db_session=db_session)


# Bulk upload
@router.post(
    ENDPOINTS.PROMPTS_BULK,
    dependencies=[Depends(auth_token_validator_no_user)],
    response_model=SuccessResponse,
)
async def bulk_upload_prompts(prebuilt_prompts: List[PrebuiltPrompt]) -> SuccessResponse:
    """Bulk uploads prompts (themes and use cases).

    Handles existing prompts as follows:
    - Skips non-deleted existing prompts
    - Revives deleted existing prompts
    - Deletes prompts not included in upload

    Args:
        prebuilt_prompts (List[PrebuiltPrompt]): The prompts to upload.

    Returns:
        SuccessResponse: Success response.
    """
    async with async_db_session() as db_session:
        return await upload_prompts_in_bulk(db_session=db_session, prompts=prebuilt_prompts)
