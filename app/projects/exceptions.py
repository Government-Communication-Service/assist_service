class ProjectNotFoundError(Exception):
    """Triggered when the project does not exist, has been deleted, or belongs to another user."""

    pass


class ProjectChatNotFoundError(Exception):
    """Triggered when the chat does not exist, has been archived, belongs to another user, or is not in the project."""

    pass


class ProjectNameConflictError(Exception):
    """Triggered when the user already has a live project with the same name, ignoring case."""

    pass
