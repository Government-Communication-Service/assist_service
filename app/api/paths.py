from fastapi import Depends, HTTPException, Path

from app.api.config import ApiConfig
from app.auth.session_user import session_user
from app.database.models import User, UserPrompt
from app.database.table import UserPromptTable


def endpoint_user_uuid(
    user_uuid: str = Path(..., description=f"Default: `{ApiConfig.PARAMS.USER_UUID}`"),
    user_header=ApiConfig.USER_KEY_UUID,
) -> User:
    if user_uuid != user_header:
        raise HTTPException(status_code=400, detail="user UUIDs do not match")

    return session_user('"user_uuid" path variable', user_uuid)


def endpoint_user_prompt_uuid(
    user_prompt_uuid: str = Path(..., description=f"Default: `{ApiConfig.PARAMS.USER_PROMPT_UUID}`"),
) -> UserPrompt:
    user_prompt = UserPromptTable().get_by_uuid(uuid=user_prompt_uuid)

    return user_prompt


class ApiPaths:
    USER_UUID: User = Depends(endpoint_user_uuid)
    USER_PROMPT_UUID: UserPrompt = Depends(endpoint_user_prompt_uuid)
