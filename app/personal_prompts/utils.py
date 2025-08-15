from fastapi import Depends, Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.exceptions import (
    UuidInvalidError,
    UuidMissingError,
)
from app.auth.utils import verify_and_parse_uuid
from app.auth.verify_service import verify_and_get_user_from_path_and_header
from app.database.db_session import get_db_session
from app.database.models import User, UserPrompt
from app.personal_prompts.exceptions import (
    PromptUuidInvalidError,
    PromptUuidMissingError,
    UserPromptMissingError,
)


# When the user makes a request for a prompt, this dependency is used
async def verify_and_get_user_prompt_by_uuid(
    user: User = Depends(verify_and_get_user_from_path_and_header),
    user_prompt_uuid: str = Path(...),
    db_session: AsyncSession = Depends(get_db_session),
) -> UserPrompt:
    try:
        user_prompt_uuid = verify_and_parse_uuid(user_prompt_uuid)
    except UuidMissingError as e:
        raise PromptUuidMissingError("user_prompt_uuid is missing from the path") from e
    except UuidInvalidError as e:
        raise PromptUuidInvalidError(f"user_prompt_uuid in the path was invalid: {user_prompt_uuid}") from e

    stmt = select(UserPrompt).where(
        UserPrompt.user_id == user.id,
        UserPrompt.uuid == user_prompt_uuid,
        UserPrompt.deleted_at.is_(None),
    )
    user_prompt_obj = await db_session.scalar(stmt)
    if not user_prompt_obj:
        raise UserPromptMissingError("No prompt for this user with user_prompt_uuid: {user_prompt_uuid}")
    return user_prompt_obj
