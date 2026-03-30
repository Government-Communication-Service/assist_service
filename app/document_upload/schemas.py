from pydantic import BaseModel


class SuccessResponse(BaseModel):
    status: str = "success"
    status_message: str = "success"


class DocumentCleanupResponse(SuccessResponse):
    """Response schema for document cleanup operation."""

    message: str
    deleted_count: int
