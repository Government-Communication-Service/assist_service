class ChatNotFoundError(Exception):
    """Raised when a chat is not found."""

    pass


class LLMNotFoundError(Exception):
    """Raised when a required LLM model is not found."""

    pass
