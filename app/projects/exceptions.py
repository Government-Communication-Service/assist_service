from app.exceptions.app_error import AppError


class ProjectNotFoundError(AppError):
    """Triggered when the project does not exist, has been deleted, or belongs to another user."""

    status_code = 404
    detail = "Project not found"
    expected = True


class ProjectChatNotFoundError(AppError):
    """Triggered when the chat does not exist, has been archived, belongs to another user, or is not in the project."""

    status_code = 404
    detail = "Chat not found"
    expected = True


class ProjectNameConflictError(AppError):
    """Triggered when the user already has a live project with the same name, ignoring case."""

    status_code = 409
    detail = "A project with this name already exists"
    expected = True
