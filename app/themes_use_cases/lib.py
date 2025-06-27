from typing import List
from uuid import UUID

from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.status import (
    HTTP_404_NOT_FOUND,
)

from app.api import api_wrapper
from app.chat.schemas import (
    PrebuiltPromptsResponse,
    SuccessResponse,
    ThemeResponse,
    ThemesResponse,
    UseCaseResponse,
    UseCasesResponse,
)
from app.database.db_operations import DbOperations
from app.database.table import DatabaseError, DatabaseExceptionErrorCode, UseCaseTable
from app.logs.logs_handler import logger
from app.themes_use_cases.schemas import (
    PrebuiltPrompt,
    ThemeInput,
    UseCaseInputPost,
    UseCaseInputPut,
)


async def success_response():
    response = SuccessResponse()
    return response


# CRUD operations on themes
@api_wrapper(task="create_theme")
async def create_theme(db_session: AsyncSession, theme_input: ThemeInput) -> ThemeResponse:
    """
    Creates a new theme or revives a soft-deleted theme with the given details.

    If a theme with the same title already exists but is soft-deleted, it will be revived
    with the new details. Otherwise, a new theme will be created.

    Args:
        db_session: sqlalchemy async session
        theme_input: ThemeInput object containing:
            - title: Title of the theme
            - subtitle: Subtitle/description of the theme
            - position: Display position/order of the theme

    Returns:
        ThemeResponse containing the created/revived theme details
    """
    theme = await DbOperations.theme_create_or_revive(db_session=db_session, theme_input=theme_input)
    logger.info(f"Theme created or revived with title: {theme_input.title}")
    theme_read = await DbOperations.get_theme(db_session=db_session, theme_uuid=theme.uuid)
    if theme:
        return ThemeResponse(**theme_read.client_response())
    logger.info(f"Failed to created or revive theme with title: {theme_input.title}")
    return Response(status_code=HTTP_404_NOT_FOUND)


@api_wrapper(task="fetch_theme")
async def fetch_theme(db_session: AsyncSession, theme_uuid: UUID) -> ThemeResponse | Response:
    """
    Retrieves a single active theme by its UUID.

    Args:
        theme_uuid: UUID of the theme to fetch

    Returns:
        ThemeResponse containing the theme details

    Raises:
        DatabaseError: If theme not found or is soft-deleted
    """
    theme = await DbOperations.get_theme(db_session=db_session, theme_uuid=theme_uuid, include_deleted_records=False)
    if theme is None:
        logger.info(f"Fail to fetch theme with UUID: {theme_uuid}")
        return Response(status_code=HTTP_404_NOT_FOUND)

    logger.info(f"Fetched theme with UUID: {theme_uuid}")
    return ThemeResponse(**theme.client_response())


@api_wrapper(task="update_theme")
async def update_theme(db_session: AsyncSession, theme_uuid: UUID, theme_input: ThemeInput) -> ThemeResponse:
    """
    Updates an existing theme's details.

    Args:
        db_session: sqlalchemy async session
        theme_uuid: UUID of the theme to update
        theme_input: ThemeInput object containing:
            - title: New title for the theme
            - subtitle: New subtitle/description
            - position: New display position

    Returns:
        ThemeResponse containing the updated theme details

    Raises:
        DatabaseError: If theme not found
    """
    theme = await DbOperations.theme_update(db_session=db_session, theme_uuid=theme_uuid, theme_input=theme_input)
    logger.info(f"Updated theme with UUID: {theme_uuid}")
    return ThemeResponse(**theme.client_response())


@api_wrapper(task="soft_delete_theme")
async def soft_delete_theme(db_session: AsyncSession, theme_uuid: UUID) -> SuccessResponse:
    """
    Marks a theme as deleted without removing it from the database.

    Args:
        db_session: sqlalchemy async session
        theme_uuid: UUID of the theme to soft delete

    Returns:
        SuccessResponse indicating successful deletion

    Raises:
        DatabaseError: If theme not found
    """
    await DbOperations.theme_soft_delete_by_uuid(db_session=db_session, theme_uuid=theme_uuid)
    logger.info(f"Soft deleted theme with UUID: {theme_uuid}")
    return SuccessResponse()


@api_wrapper(task="fetch_themes")
async def fetch_themes(db_session: AsyncSession) -> ThemesResponse:
    """
    Retrieves all active themes ordered by position or ID.

    Returns:
        ThemesResponse containing list of all active themes
    """
    themes = await DbOperations.get_themes(db_session=db_session)
    logger.info("Fetched all themes")
    return ThemesResponse(themes=[i.client_response() for i in themes])


# CRUD operations on use cases


@api_wrapper(task="create_use_case")
async def create_use_case(
    db_session: AsyncSession, theme_uuid: UUID, use_case_input: UseCaseInputPost
) -> UseCaseResponse:
    """
    Creates a new use case or revives a soft-deleted one under a theme.

    Args:
        db_session: sqlalchemy async session
        theme_uuid: UUID of the parent theme
        use_case_input: UseCaseInputPost object containing:
            - title: Title of the use case
            - instruction: Instructions for the use case
            - user_input_form: Form fields for user input
            - position: Display position within theme

    Returns:
        UseCaseResponse containing the created/revived use case details

    Raises:
        DatabaseError: If parent theme not found
    """
    theme = await DbOperations.get_theme(db_session=db_session, theme_uuid=theme_uuid, include_deleted_records=False)
    use_case = await DbOperations.use_case_create_or_revive(
        db_session=db_session, theme_uuid=theme_uuid, use_case_input=use_case_input
    )
    use_case_new = await DbOperations.get_use_case(
        db_session=db_session, theme_uuid=theme_uuid, use_case_uuid=use_case.uuid
    )
    response = UseCaseResponse(
        theme_uuid=theme.uuid,
        **use_case_new[1].client_response(),
    )
    logger.info(f"Use case created or revived: {response=}")
    return response


@api_wrapper(task="fetch_use_case")
async def fetch_use_case(db_session: AsyncSession, theme_uuid: UUID, use_case_uuid: UUID) -> UseCaseResponse | Response:
    """
    Retrieves a single use case, verifying it belongs to the specified theme.

    Args:
        theme_uuid: UUID of the parent theme
        use_case_uuid: UUID of the use case to fetch

    Returns:
        UseCaseResponse containing the use case details

    Raises:
        DatabaseError: If use case not found or doesn't belong to theme
    """
    theme, use_case = await DbOperations.get_use_case(
        db_session=db_session, theme_uuid=theme_uuid, use_case_uuid=use_case_uuid
    )
    logger.info(f"Fetched use case with UUID: {use_case_uuid}")

    if use_case:
        return UseCaseResponse(**use_case.client_response(), theme_uuid=theme.uuid)
    logger.info(f"Fail to fetch use case with UUID: {use_case_uuid}")
    return Response(status_code=HTTP_404_NOT_FOUND)


@api_wrapper(task="fetch_use_case_without_requiring_theme")
async def fetch_use_case_without_requiring_theme(db_session: AsyncSession, use_case_uuid: UUID) -> UseCaseResponse:
    """
    Retrieves a single use case without theme verification.

    Args:
        use_case_uuid: UUID of the use case to fetch

    Returns:
        UseCaseResponse containing the use case details and its theme UUID

    Raises:
        DatabaseError: If use case not found
    """
    use_case = await DbOperations.use_case_get_by_uuid_no_theme(
        db_session=db_session, use_case_uuid=use_case_uuid, include_deleted_records=False
    )
    theme = await DbOperations.theme_get_by_id(db_session=db_session, theme_id=use_case.theme_id)
    logger.info(f"Fetched use case with UUID: {use_case_uuid}")
    return UseCaseResponse(**use_case.client_response(), theme_uuid=theme.uuid)


@api_wrapper(task="fetch_use_cases")
async def fetch_use_cases(db_session: AsyncSession, theme_uuid: UUID) -> UseCasesResponse:
    """
    Retrieves all active use cases belonging to a theme.

    Args:
        theme_id: UUID of the theme to fetch use cases for

    Returns:
        UseCasesResponse containing the theme details and list of its use cases

    Raises:
        DatabaseError: If theme not found
    """
    theme = await DbOperations.get_theme(db_session=db_session, theme_uuid=theme_uuid, include_deleted_records=False)
    use_cases = UseCaseTable().get_by_theme(theme.id)
    use_cases = await DbOperations.use_case_get_by_theme(db_session=db_session, theme_uuid=theme.id)

    logger.info(f"Fetched use cases for theme UUID: {theme_uuid}")
    return UseCasesResponse(
        use_cases=[i.client_response(theme.uuid) for i in use_cases],
        **theme.client_response(),
    )


@api_wrapper(task="update_use_case")
async def update_use_case(
    db_session: AsyncSession, theme_uuid: UUID, use_case_uuid: UUID, use_case_input: UseCaseInputPut
) -> UseCaseResponse:
    """
    Updates an existing use case's details.

    Args:
        theme_uuid: UUID of the current parent theme
        use_case_uuid: UUID of the use case to update
        use_case_input: UseCaseInputPut object containing:
            - theme_uuid: UUID of new parent theme (can be different)
            - title: New title
            - instruction: New instructions
            - user_input_form: New form fields
            - position: New display position

    Returns:
        UseCaseResponse containing the updated use case details

    Raises:
        DatabaseError: If use case not found or doesn't belong to theme
    """
    # Verify that the use case belongs to this theme UUID

    use_case = await DbOperations.use_case_get_by_uuid_no_theme(db_session=db_session, use_case_uuid=use_case_uuid)
    theme = await DbOperations.get_theme(db_session=db_session, theme_uuid=theme_uuid)

    if use_case.theme_id != theme.id:
        raise DatabaseError(
            code=DatabaseExceptionErrorCode.USE_CASE_NOT_UNDER_THIS_THEME_ERROR,
            message=f"The use case UUID '{use_case_uuid}'" + f"is not associated with the theme UUID '{theme_uuid}'",
        )

    theme_to_update_to = await DbOperations.get_theme(db_session=db_session, theme_uuid=use_case_input.theme_uuid)

    # db_session: AsyncSession, theme: Theme, use_case_uuid: UUID, use_case_input: UseCaseInputPut
    use_case = await DbOperations.use_case_update_by_uuid(
        db_session=db_session, theme=theme_to_update_to, use_case_uuid=use_case.uuid, use_case_input=use_case_input
    )

    logger.info(f"Updated use case with UUID: {use_case_uuid}")
    return UseCaseResponse(**use_case.client_response(), theme_uuid=theme_to_update_to.uuid)


@api_wrapper(task="soft_delete_use_case")
async def soft_delete_use_case(db_session: AsyncSession, theme_uuid: UUID, use_case_uuid: UUID) -> SuccessResponse:
    """
    Marks a use case as deleted without removing it from the database.

    Args:
        theme_uuid: UUID of the parent theme
        use_case_uuid: UUID of the use case to soft delete

    Returns:
        SuccessResponse indicating successful deletion

    Raises:
        DatabaseError: If use case not found or doesn't belong to theme
    """
    # Verify that the use case belongs to this theme UUID

    use_case = await DbOperations.use_case_get_by_uuid_no_theme(db_session=db_session, use_case_uuid=use_case_uuid)
    theme = await DbOperations.get_theme(db_session=db_session, theme_uuid=theme_uuid)

    if use_case.theme_id != theme.id:
        raise DatabaseError(
            code=DatabaseExceptionErrorCode.USE_CASE_NOT_UNDER_THIS_THEME_ERROR,
            message=f"The use case UUID '{use_case_uuid}'" + f"is not associated with the theme UUID '{theme_uuid}'",
        )

    await DbOperations.use_case_soft_delete_by_uuid(db_session=db_session, use_case_uuid=use_case.uuid)
    logger.info(f"Soft deleted use case with UUID: {use_case_uuid}")
    return SuccessResponse()


# Bulk upload of prompts
@api_wrapper(task="upload_prompts_in_bulk")
async def upload_prompts_in_bulk(db_session: AsyncSession, prompts: List[PrebuiltPrompt]) -> SuccessResponse:
    """
    Replaces all existing prompts with a new set of themes and use cases.

    This operation:
    1. Soft deletes all existing themes and use cases
    2. Creates or revives themes and use cases from the provided list
    3. Ensures only the uploaded prompts remain active in the database

    Args:
        prompts: List of PrebuiltPrompt objects, each containing:
            - theme_title: Title of the theme
            - theme_subtitle: Theme subtitle/description
            - theme_position: Theme display position
            - use_case_title: Title of the use case
            - use_case_instruction: Use case instructions
            - use_case_user_input_form: Form fields for user input
            - use_case_position: Use case display position

    Returns:
        SuccessResponse indicating successful bulk upload

    Raises:
        DatabaseError: If any database operation fails
    """

    # Delete all themes and use cases
    await DbOperations.theme_delete_all(db_session=db_session)
    await DbOperations.use_case_delete_all(db_session=db_session)

    # Create or revive all themes and use cases defined by the user in their bulk upload
    for prompt in prompts:
        theme_input = ThemeInput(
            title=prompt.theme_title,
            subtitle=prompt.theme_subtitle,
            position=prompt.theme_position,
        )
        # theme_input.title = prompt.theme_title
        # theme_input.subtitle = prompt.theme_subtitle
        # theme_input.position = prompt.theme_position
        theme = await DbOperations.theme_create_or_revive(db_session=db_session, theme_input=theme_input)

        use_case_input = UseCaseInputPost(
            title=prompt.use_case_title,
            instruction=prompt.use_case_instruction,
            user_input_form=prompt.use_case_user_input_form,
            theme_id=theme.id,
            position=prompt.use_case_position,
        )
        await DbOperations.use_case_create_or_revive(
            db_session=db_session, theme_uuid=theme.uuid, use_case_input=use_case_input
        )

    import asyncio

    return await asyncio.create_task(success_response())


@api_wrapper(task="fetch_all_prompts")
async def fetch_all_prompts(db_session: AsyncSession) -> PrebuiltPromptsResponse:
    """
    Retrieves all active themes and their use cases in a format suitable for bulk editing.

    Returns:
        PrebuiltPromptsResponse containing a list of PrebuiltPrompt objects,
        each representing a theme-use case pair with all their attributes
    """
    themes = await DbOperations.theme_fetch_all_ordered_by_position_or_id(db_session=db_session)
    theme_use_cases = []

    for theme in themes:
        use_cases = await DbOperations.use_case_get_by_theme(db_session=db_session, theme_uuid=theme.id)
        for use_case in use_cases:
            theme_use_case = PrebuiltPrompt(
                theme_title=theme.title,
                theme_subtitle=theme.subtitle,
                theme_position=theme.position,
                use_case_title=use_case.title,
                use_case_instruction=use_case.instruction,
                use_case_user_input_form=use_case.user_input_form,
                use_case_position=use_case.position,
            )
            theme_use_cases.append(theme_use_case)

    logger.info("Fetched all prompts for bulk upload")
    return PrebuiltPromptsResponse(prompts=theme_use_cases)
