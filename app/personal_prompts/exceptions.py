from app.exceptions.app_error import AppError


class PromptUuidMissingError(AppError):
    """Triggered when the prompt UUID is missing."""

    status_code = 400
    detail = "User prompt UUID missing"
    expected = True


class PromptUuidInvalidError(AppError):
    """Triggered when the prompt UUID is invalid."""

    status_code = 400
    detail = "User prompt UUID malformed"
    expected = True


class UserPromptMissingError(AppError):
    """Triggered when the prompt UUID did not return a user prompt"""

    status_code = 404
    detail = "User prompt not found"
    expected = True
