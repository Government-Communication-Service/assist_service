from logging import getLogger
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header, Path
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.config import AUTH_TOKEN, AUTH_TOKEN_2, DEFAULT_AUTH_TOKEN, DEFAULT_USER_KEY_UUID
from app.auth.constants import AUTH_TOKEN_ALIAS, SESSION_AUTH_ALIAS, USER_KEY_UUID_ALIAS
from app.auth.exceptions import (
    AddNewUserError,
    AuthTokenInvalidError,
    AuthTokenMissingError,
    SessionUuidMalformedError,
    SessionUuidMissingError,
    SessionUuidNotInDatabaseError,
    UserKeyUuidMalformedError,
    UserKeyUuidMissingError,
    UserUuidNotMatchingError,
    UuidInvalidError,
    UuidMissingError,
)
from app.auth.utils import verify_and_parse_uuid
from app.database.db_session import get_db_session
from app.database.models import AuthSession, User

logger = getLogger(__name__)


async def _upsert_user_by_uuid(db_session: AsyncSession, user_key_uuid: UUID) -> User:
    # If the user exists in the database they are an existing user.
    user_obj = await db_session.scalar(select(User).where(User.uuid == user_key_uuid))

    # If the user is not in the database they are interpreted as a new user.
    # A new user UUID is always accepted.
    if not user_obj:
        try:
            user_obj = User(uuid=user_key_uuid)
            db_session.add(user_obj)
            await db_session.flush()  # Get the ID
            await db_session.refresh(user_obj)  # Refresh the object with the DB state
        except SQLAlchemyError as e:
            raise AddNewUserError("An error occurred while adding a new user to the database") from e
    return user_obj


def verify_auth_token(auth_token: Annotated[str, Header(alias=AUTH_TOKEN_ALIAS)] = DEFAULT_AUTH_TOKEN) -> bool:
    """Checks that the provided Auth-Token header is valid."""
    if not auth_token:
        raise AuthTokenMissingError("Auth-Token header is missing.")
    if auth_token not in [AUTH_TOKEN, AUTH_TOKEN_2]:
        raise AuthTokenInvalidError("Auth-Token header is invalid.")
    return True


async def verify_and_get_user_from_header(
    user_key_uuid: Annotated[str, Header(alias=USER_KEY_UUID_ALIAS)] = DEFAULT_USER_KEY_UUID,
    db_session: AsyncSession = Depends(get_db_session),
):
    # Verify that the header value is a valid UUID
    # Add the user if they are not there already
    try:
        user_key_uuid = verify_and_parse_uuid(user_key_uuid)
        return await _upsert_user_by_uuid(db_session, user_key_uuid)
    except UuidMissingError as e:
        raise UserKeyUuidMissingError("User-Key-UUID header was missing.") from e
    except UuidInvalidError as e:
        raise UserKeyUuidMalformedError(f"User-Key-UUID header was provided but malformed: {user_key_uuid}") from e


async def verify_and_get_user_from_path(
    user_uuid: Annotated[str, Path()],
    db_session: AsyncSession = Depends(get_db_session),
):
    try:
        user_uuid = verify_and_parse_uuid(user_uuid)
        return await _upsert_user_by_uuid(db_session, user_uuid)
    except UuidMissingError as e:
        raise UserKeyUuidMissingError("user_uuid path parameter was missing.") from e
    except UuidInvalidError as e:
        raise UserKeyUuidMalformedError(f"user_uuid path parameter was provided but malformed: {user_uuid}") from e


async def verify_and_get_user_from_path_and_header(
    user_path: User = Depends(verify_and_get_user_from_path),
    user_header: User = Depends(verify_and_get_user_from_header),
) -> User:
    """Checks that the provided User-Key-UUID header and user_uuid path parameter are valid."""
    # Verify that header and path parameter match
    if user_path.uuid != user_header.uuid:
        raise UserUuidNotMatchingError(
            f"The header User-Key-UUID and path parameter user_uuid do not match: "
            f"header={user_header.uuid}, path={user_path.uuid}"
        )
    return user_path


async def verify_and_get_auth_session_from_header(
    session_auth: Annotated[str, Header(alias=SESSION_AUTH_ALIAS)],
    user_obj: User = Depends(verify_and_get_user_from_header),
    db_session: AsyncSession = Depends(get_db_session),
) -> AuthSession:
    """Checks that the provided Session-Auth header is valid"""

    # Start by checking for a malformed UUID.
    try:
        session_auth = verify_and_parse_uuid(session_auth)
    except UuidMissingError as e:
        raise SessionUuidMissingError("Session-Auth header was missing.") from e
    except UuidInvalidError as e:
        raise SessionUuidMalformedError(f"Session-Auth header was provided but malformed: {session_auth}") from e

    session_auth_obj = await db_session.scalar(select(AuthSession).where(AuthSession.uuid == session_auth))

    if not session_auth_obj:
        raise SessionUuidNotInDatabaseError(
            f"Could not find the provided Session-Auth header in the database: {session_auth}"
        )

    return session_auth_obj
