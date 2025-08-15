from fastapi import APIRouter, Depends

from app.api.endpoints import ENDPOINTS
from app.auth.constants import SESSION_AUTH_ALIAS
from app.auth.create_auth_session_service import create_auth_session
from app.auth.verify_service import verify_and_get_user_from_header, verify_auth_token
from app.database.db_session import get_db_session

router = APIRouter()


@router.post(
    path=ENDPOINTS.SESSIONS,
    dependencies=[Depends(verify_auth_token)],
)
async def create_auth_session_endpoint(
    user=Depends(verify_and_get_user_from_header),
    db_session=Depends(get_db_session),
):
    """
    Generates the session item in the database that will be provided as a header token to validate
    the rest of the endpoints.
    Only the session UUID should be returned to the client to be used elsewhere.
    """
    session = await create_auth_session(db_session=db_session, user=user)
    return {SESSION_AUTH_ALIAS: str(session.uuid)}
