class PromptUuidMissingError(Exception):
    """Triggered when the prompt UUID is missing."""

    pass


class PromptUuidInvalidError(Exception):
    """Triggered when the prompt UUID is invalid."""

    pass


class UserPromptMissingError(Exception):
    """Triggered when the prompt UUID did not return a user prompt"""

    pass
