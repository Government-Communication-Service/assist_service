import bugsnag
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response

from app.auth.exceptions import (
    AddNewUserError,
    AuthTokenInvalidError,
    AuthTokenMissingError,
    SessionUuidMalformedError,
    SessionUuidMissingError,
    SessionUuidNotInDatabaseError,
    UserKeyUuidMalformedError,
    UserKeyUuidMissingError,
    UserUuidNotMatchingError,
)
from app.bedrock.bedrock_types import BedrockError
from app.chat.schemas import DocumentAccessError
from app.database.database_exception import DatabaseError, DatabaseExceptionErrorCode
from app.logs.logs_handler import logger
from app.personal_prompts.exceptions import UserPromptMissingError


def log_and_raise_http_exception(status_code: int, request: Request, exc: Exception) -> None:
    logger.error(f"Error in endpoint {request.url.path}: {exc}\n", exc_info=exc)
    raise HTTPException(status_code=status_code, detail=str(exc))


async def auth_token_missing_handler(request: Request, exc: AuthTokenMissingError) -> None:
    log_and_raise_http_exception(status_code=401, request=request, exc=exc)


def auth_token_invalid_handler(request: Request, exc: AuthTokenInvalidError) -> None:
    log_and_raise_http_exception(status_code=401, request=request, exc=exc)


def add_new_user_error_handler(request: Request, exc: AddNewUserError) -> None:
    log_and_raise_http_exception(status_code=401, request=request, exc=exc)


def session_uuid_missing_handler(request: Request, exc: SessionUuidMissingError) -> None:
    log_and_raise_http_exception(status_code=400, request=request, exc=exc)


def user_key_uuid_missing_handler(request: Request, exc: UserKeyUuidMissingError) -> None:
    log_and_raise_http_exception(status_code=400, request=request, exc=exc)


def user_uuid_not_matching_handler(request: Request, exc: UserUuidNotMatchingError) -> None:
    log_and_raise_http_exception(status_code=403, request=request, exc=exc)


def session_uuid_malformed_handler(request: Request, exc: SessionUuidMalformedError) -> None:
    log_and_raise_http_exception(status_code=400, request=request, exc=exc)


def user_key_uuid_malformed_handler(request: Request, exc: UserKeyUuidMalformedError) -> None:
    log_and_raise_http_exception(status_code=400, request=request, exc=exc)


def session_uuid_not_in_database_handler(request: Request, exc: SessionUuidNotInDatabaseError) -> None:
    log_and_raise_http_exception(status_code=404, request=request, exc=exc)


def user_prompt_missing_handler(request: Request, exc: UserPromptMissingError) -> None:
    log_and_raise_http_exception(status_code=404, request=request, exc=exc)


def handle_document_access_error(request: Request, ex: DocumentAccessError) -> Response:
    detail = {"error": "DOCUMENT_ACCESS_ERROR", "documents_uuids": ex.document_uuids}
    return JSONResponse(status_code=401, content=detail)


async def database_exception_handler(request: Request, exc: DatabaseError) -> Response:
    logger.error(f"Database Error: code={exc.code}, message={exc.message}")
    bugsnag.notify(exc)

    if exc.code == DatabaseExceptionErrorCode.GET_BY_UUID_ERROR:
        return JSONResponse(content="Record not found", status_code=404)
    if exc.code == DatabaseExceptionErrorCode.USE_CASE_NOT_UNDER_THIS_THEME_ERROR:
        return JSONResponse(content=f"{exc.message}", status_code=404)
    return JSONResponse(content="An internal error occurred.", status_code=500)


async def bedrock_exception_handler(request: Request, exc: BedrockError) -> Response:
    return JSONResponse(
        status_code=503,
        content={"status": "failed", "error_code": "BEDROCK_SERVICE_ERROR", "status_message": str(exc)},
    )


def register_exception_handlers(app):
    """Register all exception handlers with the FastAPI app."""
    app.exception_handler(AuthTokenMissingError)(auth_token_missing_handler)
    app.exception_handler(AuthTokenInvalidError)(auth_token_invalid_handler)
    app.exception_handler(AddNewUserError)(add_new_user_error_handler)
    app.exception_handler(SessionUuidMissingError)(session_uuid_missing_handler)
    app.exception_handler(UserKeyUuidMissingError)(user_key_uuid_missing_handler)
    app.exception_handler(UserUuidNotMatchingError)(user_uuid_not_matching_handler)
    app.exception_handler(SessionUuidMalformedError)(session_uuid_malformed_handler)
    app.exception_handler(UserKeyUuidMalformedError)(user_key_uuid_malformed_handler)
    app.exception_handler(SessionUuidNotInDatabaseError)(session_uuid_not_in_database_handler)
    app.exception_handler(UserPromptMissingError)(user_prompt_missing_handler)
    app.exception_handler(DocumentAccessError)(handle_document_access_error)
    app.exception_handler(DatabaseError)(database_exception_handler)
    app.exception_handler(BedrockError)(bedrock_exception_handler)
