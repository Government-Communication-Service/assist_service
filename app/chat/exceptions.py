from app.exceptions.app_error import AppError


class ChatNotFoundError(Exception):
    """Raised when a chat is not found."""

    pass


class LLMNotFoundError(Exception):
    """Raised when a required LLM model is not found."""

    pass


class DocumentAccessError(AppError):
    """Raised when a chat request names documents the user doesn't have access to."""

    status_code = 401
    expected = True

    def __init__(self, *args, document_uuids: list[str]):
        super().__init__(*args)
        self.document_uuids = document_uuids

    def content(self) -> dict:
        return {"error": "DOCUMENT_ACCESS_ERROR", "documents_uuids": self.document_uuids}


class UseCaseIdMalformedError(AppError):
    """Raised when a chat request's use_case_id is not a valid UUID."""

    status_code = 400
    detail = "use_case_id malformed"
    expected = True


class UseCaseNotFoundError(AppError):
    """Raised when a chat request's use_case_id doesn't match a use case."""

    status_code = 400
    detail = "Use case not found"
    expected = True
