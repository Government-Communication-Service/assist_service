import logging
from typing import Annotated

from fastapi import Header, Path

from app.auth.constants import SESSION_AUTH_ALIAS, USER_KEY_UUID_ALIAS
from app.auth.verify_service import (
    verify_and_get_auth_session_from_header,
    verify_and_get_user_from_header,
    verify_and_get_user_from_path,
    verify_and_get_user_from_path_and_header,
)
from app.config import USER_UPDATE_REQUIRES_SESSION_AUTH
from app.database.db_session import DbSession

logger = logging.getLogger(__name__)


async def verify_user_update_access(
    db_session: DbSession,
    user_uuid: Annotated[str, Path()],
    user_key_uuid: Annotated[str | None, Header(alias=USER_KEY_UUID_ALIAS)] = None,
    session_auth: Annotated[str | None, Header(alias=SESSION_AUTH_ALIAS)] = None,
) -> None:
    """
    Checks that the caller may update the profile of the user in the path.

    With USER_UPDATE_REQUIRES_SESSION_AUTH on, the User-Key-UUID header must match the path user and Session-Auth
    must belong to that user. With it off, only the Auth-Token is checked, as before, so the frontend can start
    sending both headers before the check is enforced.
    """
    if not USER_UPDATE_REQUIRES_SESSION_AUTH:
        missing = [
            name
            for name, value in ((USER_KEY_UUID_ALIAS, user_key_uuid), (SESSION_AUTH_ALIAS, session_auth))
            if not value
        ]
        if missing:
            logger.warning("PUT user %s called without the %s header(s)", user_uuid, " and ".join(missing))
        return

    user_path = await verify_and_get_user_from_path(user_uuid=user_uuid, db_session=db_session)
    user_header = await verify_and_get_user_from_header(user_key_uuid=user_key_uuid, db_session=db_session)
    user = await verify_and_get_user_from_path_and_header(user_path=user_path, user_header=user_header)
    await verify_and_get_auth_session_from_header(session_auth=session_auth, user_obj=user, db_session=db_session)
