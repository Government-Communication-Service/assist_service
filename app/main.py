import asyncio
from asyncio import exceptions
from contextlib import asynccontextmanager

import bugsnag
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse

import app.chat.routes as chat
import app.feedback.routes as feedback
import app.user.user as user
from app.api import healthcheck
from app.auth import session
from app.bedrock.bedrock_types import BedrockError
from app.central_guidance import routes
from app.central_guidance.service_index import sync_central_index
from app.chat.schemas import DocumentAccessError
from app.chat.service import schedule_expired_messages_deletion
from app.config import IS_DEV, URL_HOSTNAME, load_environment_variables
from app.database.database_exception import (
    DatabaseError,
    DatabaseExceptionErrorCode,
)
from app.database.db_session import async_db_session
from app.database.table import AsyncEngineProvider
from app.document_upload.document_management import schedule_expired_files_deletion
from app.logs import BUGSNAG_ENABLED, BugsnagLogger
from app.logs.logs_handler import logger, session_id_var
from app.opensearch.service import verify_connection_to_opensearch
from app.personal_prompts import user_prompt
from app.themes_use_cases import themes_use_cases
from app.themes_use_cases.sync_service import sync_themes_use_cases


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan function that runs code before startup and on shutdown.

    Handles startup tasks like verifying OpenSearch connection and syncing indexes.
    On shutdown, closes database connections.

    Args:
        app (FastAPI): The FastAPI application instance

    Yields:
        None: Yields control to the main API code
    """
    # Startup code is written here

    # Sync with OpenSearch
    verify_connection_to_opensearch()
    async with async_db_session() as s:
        # Sync themes and use cases
        await sync_themes_use_cases(s)

        # Sync the central RAG documents (OpenSearch)
        await sync_central_index(s)

        # schedule deleting expired messages
        asyncio.create_task(schedule_expired_messages_deletion(s))

    # schedule deleting expired documents
    asyncio.create_task(schedule_expired_files_deletion())

    # Now yield to the main API code
    yield

    # Shutdown code can be written here
    logger.info("Closing DB connections")
    await AsyncEngineProvider.get().dispose()


load_environment_variables()

app = FastAPI(title="GCS Assist API", version="0.1.0", lifespan=lifespan)
app.openapi_version = "3.0.2"
REQUEST_TIMEOUT_SECS = 120

# Configure CORS
if IS_DEV:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins in development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[URL_HOSTNAME],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
# Setup Bugsnag logger
logger.info(f"BUGSNAG_ENABLED: {BUGSNAG_ENABLED}")
if BUGSNAG_ENABLED:
    bugsnag_logger = BugsnagLogger()
    bugsnag_logger.setup_bugsnag(app)


@app.middleware("http")
async def add_session_id(request: Request, call_next):
    session_id = request.headers.get("Session-Auth")
    if session_id:
        session_id_var.set(session_id)

    response = await call_next(request)
    return response


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    """
    Returns robots.txt content to prevent web crawlers from indexing the API.

    Returns:
        str: robots.txt content disallowing all crawlers
    """
    return "User-agent: *\nDisallow: /"


@app.get("/", include_in_schema=False)
def root():
    """
    Redirects root URL to API documentation.

    Returns:
        RedirectResponse: Redirect to /docs endpoint
    """
    return RedirectResponse(url="/docs")


# Include routers
app.include_router(healthcheck.router, prefix="/healthcheck", tags=["Health Check"])
app.include_router(session.router, prefix="/v1", tags=["Auth Sessions"])
app.include_router(chat.router, prefix="/v1", tags=["Chat Sessions"])
app.include_router(feedback.router, prefix="/v1", tags=["Message Feedback"])
app.include_router(user.router, prefix="/v1", tags=["User Data"])
app.include_router(user_prompt.router, prefix="/v1", tags=["User Prompts"])
app.include_router(themes_use_cases.router, prefix="/v1", tags=["Themes / Use Cases"])
app.include_router(routes.router, prefix="/v1", tags=["Central RAG"])


# exception handlers
@app.exception_handler(DocumentAccessError)
def handle_document_access_error(request: Request, ex: DocumentAccessError):
    """
    Handles document access errors by returning a 401 unauthorized response.

    Args:
        request (Request): The incoming request
        ex (DocumentAccessError): The document access error

    Returns:
        JSONResponse: Error response with 401 status code
    """
    detail = {"error": "DOCUMENT_ACCESS_ERROR", "documents_uuids": ex.document_uuids}
    return JSONResponse(status_code=401, content=detail)


@app.exception_handler(DatabaseError)
async def database_exception_handler(request: Request, exc: DatabaseError):
    """
    Handles database errors by returning appropriate HTTP error responses.

    Args:
        request (Request): The incoming request
        exc (DatabaseError): The database error

    Returns:
        JSONResponse: With appropriate status code and message based on error type
    """
    # Report the detailed bug to Bugsnag
    bugsnag.notify(f"Database Error: {exc}")

    if exc.code == DatabaseExceptionErrorCode.GET_BY_UUID_ERROR:
        return JSONResponse(content="Record not found", status_code=404)
    if exc.code == DatabaseExceptionErrorCode.USE_CASE_NOT_UNDER_THIS_THEME_ERROR:
        return JSONResponse(content=f"{exc.message}", status_code=404)
    # Handle all other DatabaseErrors generically
    return JSONResponse(content="An internal error occurred.", status_code=500)


@app.exception_handler(BedrockError)
async def bedrock_exception_handler(request: Request, exc: BedrockError):
    """
    Handles BedrockError  by returning HTTP 503 status code and BEDROCK_SERVICE_ERROR error code

    Args:
        request (Request): The incoming request
        exc (BedrockError): The Bedrock error

    Returns:
        JSONResponse: With 503 status code and BEDROCK_SERVICE_ERROR error code
    """
    return JSONResponse(
        status_code=503,
        content={"status": "failed", "error_code": "BEDROCK_SERVICE_ERROR", "status_message": str(exc)},
    )


@app.middleware("http")
async def set_global_timeout(request: Request, call_next):
    try:
        response = await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_SECS)
        return response
    except exceptions.TimeoutError:
        return JSONResponse(
            status_code=503,
            content={
                "status": "failed",
                "error_code": "REQUEST_TIMED_OUT",
                "status_message": "Server failed to process the request on time",
            },
        )
