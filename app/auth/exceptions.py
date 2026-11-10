from app.exceptions.app_error import AppError


class UuidMissingError(Exception):
    """Triggered when a required UUID is missing from the header of a request."""

    pass


class UuidInvalidError(Exception):
    """Triggered when a provided string which is meant to be a UUID cannot be parsed to a UUID type."""

    pass


class AuthTokenMissingError(AppError):
    """Triggered when the Auth-Token header is missing from a request."""

    status_code = 401
    detail = "Auth token missing"


class AuthTokenInvalidError(AppError):
    """Triggered when the Auth-Token header was provided but did not match any of the expected Auth-Tokens"""

    status_code = 401
    detail = "Auth token invalid"


class UserKeyUuidMissingError(AppError):
    """Triggered when the User-Key-UUID header is missing."""

    status_code = 400
    detail = "User key UUID missing"
    expected = True


class UserKeyUuidMalformedError(AppError):
    """Triggered when the User-Key-UUID header is provided but invalid."""

    status_code = 400
    detail = "User key UUID malformed"
    expected = True


class UserUuidNotMatchingError(AppError):
    """Triggered when the User-Key-UUID in the header does not match the user_uuid provided as a path parameter."""

    status_code = 403
    detail = "User UUID mismatch"


class AddNewUserError(AppError):
    """Triggered when attempting to add a new user to the database."""

    status_code = 401
    detail = "Unable to add user"


class SessionUuidMissingError(AppError):
    """Triggered when the Session-Auth header is missing."""

    status_code = 400
    detail = "Session UUID missing"
    expected = True


class SessionUuidMalformedError(AppError):
    """Triggered when the Session-Auth header is provided but invalid."""

    status_code = 400
    detail = "Session UUID malformed"
    expected = True


class SessionUuidNotInDatabaseError(AppError):
    """Triggered when the Session-Auth provided was a valid UUID but could not be found in the database."""

    status_code = 404
    detail = "Session not found"
    expected = True
