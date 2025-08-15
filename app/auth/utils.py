from uuid import UUID

from app.auth.exceptions import UuidInvalidError, UuidMissingError


def verify_and_parse_uuid(uuid: str) -> UUID:
    if not uuid:
        raise UuidMissingError("uuid is missing")
    try:
        return UUID(uuid)
    except ValueError as e:
        raise UuidInvalidError(f"Provided uuid was invalid: '{uuid}'") from e
