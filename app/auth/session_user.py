from app.database.models import User
from app.database.table import UserTable
from app.verify_uuid import verify_uuid


def session_user(input_type=str, input_value=str) -> User:
    uuid = verify_uuid(input_type, input_value)

    user_repo = UserTable()
    user = user_repo.upsert_by_uuid(uuid)

    return user
