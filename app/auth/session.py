from fastapi import APIRouter, Depends, status

from app.api import ENDPOINTS, ApiConfig
from app.auth.auth_sessions import AuthSessions
from app.auth.auth_token import auth_token_validator
from app.database.table import async_db_session

router = APIRouter()


@router.post(
    ENDPOINTS.SESSIONS,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(auth_token_validator)],
)
async def create_auth_session_endpoint(
    user_key_uuid=ApiConfig.USER_KEY_UUID,
):
    """
    Generates the session item in the database that will be provided as a header token to validate
    the rest of the endpoints.
    Only the session UUID should be returned to the client to be used elsewhere.
    """

    async with async_db_session() as db_session:
        return await AuthSessions.create(db_session=db_session, user_key_uuid=user_key_uuid)
