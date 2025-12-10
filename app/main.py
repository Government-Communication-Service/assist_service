# ruff: noqa: F401

import asyncio
from asyncio import exceptions
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.central_guidance.service_index import sync_central_index
from app.chat.service import schedule_expired_messages_deletion
from app.config import IS_DEV, URL_HOSTNAME, SMART_TARGETS_SERVICE_DISABLED, load_environment_variables
from app.database.db_session import async_db_session
from app.database.table import AsyncEngineProvider
from app.document_upload.document_management import schedule_expired_files_deletion
from app.exceptions.handlers import register_exception_handlers
from app.logs import BUGSNAG_ENABLED, BugsnagLogger
from app.logs.logs_handler import logger, session_id_var
from app.opensearch.service import verify_connection_to_opensearch
from app.routers import routers
from app.smart_targets.service import SmartTargetsService
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
    verify_connection_to_opensearch()

    # Verify connection with the GCS Data API
    if SMART_TARGETS_SERVICE_DISABLED and IS_DEV:
        logger.info("Skipping Smart Targets Service connection verification")
    else:
        await SmartTargetsService().verify_connection()

    if not IS_DEV:
        # Sync with OpenSearch
        async with async_db_session() as s:
            # Sync themes and use cases
            # await sync_themes_use_cases(s)

            # Sync the central RAG documents (OpenSearch)
            await sync_central_index(s)

        # schedule deleting expired messages
        asyncio.create_task(schedule_expired_messages_deletion())

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
async def request_middleware(request: Request, call_next):
    # Set session ID from header
    session_id = request.headers.get("Session-Auth")
    if session_id:
        session_id_var.set(session_id)

    # Apply global timeout
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


for router in routers:
    app.include_router(router.router, prefix=router.prefix, tags=list(router.tags))

register_exception_handlers(app)
