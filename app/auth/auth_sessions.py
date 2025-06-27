from sqlalchemy.ext.asyncio import AsyncSession

from app.api import ApiConfig
from app.database.db_operations import DbOperations
from app.verify_uuid import verify_uuid


class AuthSessions:
    @staticmethod
    async def create(
        db_session: AsyncSession,
        user_key_uuid: str,
    ):
        user_key_uuid = verify_uuid(ApiConfig.USER_KEY_UUID_ALIAS, user_key_uuid)

        user = await DbOperations.upsert_user_by_uuid(db_session=db_session, user_key_uuid=user_key_uuid)
        session = await DbOperations.create_auth_session(db_session=db_session, user=user)

        return {ApiConfig.SESSION_AUTH_ALIAS: str(session.uuid)}
