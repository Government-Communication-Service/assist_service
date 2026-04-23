from typing import Protocol


class UserProtocol(Protocol):
    """Structural type for the Locust HttpUser + AuthMixin combination."""

    user_uuid: str
    auth_headers: dict[str, str]
    uploaded_document_uuids: list[str]
    audience_document_uuids: list[str]
    chat_uuids: list[str]
    last_analysis_result: dict | None
    last_audience_filename: str | None

    def client(self) -> object: ...
