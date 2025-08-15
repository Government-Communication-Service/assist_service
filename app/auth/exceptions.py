class UuidMissingError(Exception):
    """Triggered when a required UUID is missing from the header of a request."""

    pass


class UuidInvalidError(Exception):
    """Triggered when a provided string which is meant to be a UUID cannot be parsed to a UUID type."""

    pass


class AuthTokenMissingError(Exception):
    """Triggered when the Auth-Token header is missing from a request."""

    pass


class AuthTokenInvalidError(Exception):
    """Triggered when the Auth-Token header was provided but did not match any of the expected Auth-Tokens"""

    pass


class UserKeyUuidMissingError(Exception):
    """Triggered when the User-Key-UUID header is missing."""

    pass


class UserKeyUuidMalformedError(Exception):
    """Triggered when the User-Key-UUID header is provided but invalid."""

    pass


class UserUuidNotMatchingError(Exception):
    """Triggered when the User-Key-UUID in the header does not match the user_uuid provided as a path parameter."""


class AddNewUserError(Exception):
    """Triggered when attempting to add a new user to the database."""

    pass


class SessionUuidMissingError(Exception):
    """Triggered when the Session-Auth header is missing."""

    pass


class SessionUuidMalformedError(Exception):
    """Triggered when the Session-Auth header is provided but invalid."""

    pass


class SessionUuidNotInDatabaseError(Exception):
    """Triggered when the Session-Auth provided was a valid UUID but could not be found in the database."""
