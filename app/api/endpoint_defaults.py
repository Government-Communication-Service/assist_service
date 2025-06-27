# ruff: noqa: B008

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.config import ApiConfig
from app.auth.auth_token import auth_token_validator
from app.auth.session_request import SessionRequest
from app.auth.session_user import session_user
from app.config import BYPASS_SESSION_VALIDATOR
from app.database.table import AuthSessionTable
from app.error_messages import ErrorMessages
from app.logs import LogsHandler
from app.verify_uuid import verify_uuid

router = APIRouter()


def get_current_session(
    auth_session=ApiConfig.SESSION_AUTH,
    user_key_uuid=ApiConfig.USER_KEY_UUID,
):
    auth_session = verify_uuid(ApiConfig.SESSION_AUTH_ALIAS, auth_session)

    try:
        user = session_user(ApiConfig.USER_KEY_UUID, user_key_uuid)

        session_repo = AuthSessionTable()
        if BYPASS_SESSION_VALIDATOR:
            session = session_repo.create({"user_id": user.id})
        else:
            session = session_repo.get_by_uuid(auth_session)
            if not session:
                e = HTTPException(
                    status_code=403,
                    detail=ErrorMessages.item_not_found("auth session", "UUID", auth_session),
                )
                LogsHandler.error(e, "retrieving auth session")

                raise e

        return SessionRequest(id=session.id, user_id=user.id)
    except HTTPException as e:
        LogsHandler.error(e, f'validating "{ApiConfig.SESSION_AUTH_ALIAS}" header')
        raise e


def endpoint_defaults(
    status_code: int = status.HTTP_200_OK,
    dependencies: Optional[List] = None,  # Default to None
    extra_dependencies: Optional[List] = None,
):
    # Initialize dependencies if None
    if dependencies is None:
        dependencies = [Depends(get_current_session), Depends(auth_token_validator)]

    if extra_dependencies is None:
        extra_dependencies = []

    return {
        "status_code": status_code,
        "dependencies": dependencies + extra_dependencies,
    }
