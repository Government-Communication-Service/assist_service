import logging
import os

import bugsnag
import bugsnag.handlers
from bugsnag.asgi import BugsnagMiddleware
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class BugsnagLogger:
    """
    BugsnagLogger is responsible for configuring Bugsnag integration, capturing ERROR level logs and sending
    to Bugnsag and setting up error handling middleware for FastAPI applications.

    It logs errors to Bugsnag based on configuration and provides exception handlers to handle
    common application errors such as 500 (Internal Server Error), 404 (Not Found) and 422 (Validation Error).
    """

    def __init__(self):
        """
        Initialize the BugsnagLogger class by loading the Bugsnag API key, release stage, and
        other configurations from environment variables.

        Raises:
            Exception: If BUGSNAG_API_KEY or BUGSNAG_RELEASE_STAGE is not set in the environment.
        """
        self.BUGSNAG_API_KEY = os.getenv("BUGSNAG_API_KEY")
        self.BUGSNAG_RELEASE_STAGE = os.getenv("BUGSNAG_RELEASE_STAGE")
        self.DISABLE_BUGSNAG_LOGGING = os.getenv("DISABLE_BUGSNAG_LOGGING", False)

        if not self.BUGSNAG_API_KEY:
            raise Exception("BUGSNAG_API_KEY is required")
        if not self.BUGSNAG_RELEASE_STAGE:
            raise Exception("BUGSNAG_RELEASE_STAGE is required")

        self.BUGSNAG_ENABLED = bool(
            not self.DISABLE_BUGSNAG_LOGGING and self.BUGSNAG_API_KEY and self.BUGSNAG_RELEASE_STAGE,
        )

        if self.BUGSNAG_ENABLED:
            self._configure_bugsnag()

    def _configure_bugsnag(self):
        """
        Configure Bugsnag with the API key and release stage. Filters sensitive parameters
        like 'Auth-Token' from being logged by Bugsnag.
        """
        bugsnag.configure(
            api_key=self.BUGSNAG_API_KEY,
            project_root=".",
            release_stage=self.BUGSNAG_RELEASE_STAGE,
            params_filters=["Auth-Token"],
        )

    async def _catch_all_exception_handler(self, request: Request, exc: Exception) -> JSONResponse:
        bugsnag.notify(exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "Internal server error"},
        )

    async def _not_found_exception_handler(self, request: Request, exc: Exception) -> JSONResponse:
        bugsnag.notify(exc)
        logger.info(f"404 error: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": "Not found"},
        )

    async def _validation_exception_handler(self, request: Request, exc: RequestValidationError) -> JSONResponse:
        bugsnag.notify(exc)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors()},
        )

    def _send_error_level_logs_to_bugsnag(self):
        """
        Configure logging to send ERROR level logs to Bugsnag using BugsnagHandler logging handler.
        """
        bugsnag_handler = bugsnag.handlers.BugsnagHandler()
        bugsnag_handler.setLevel(logging.ERROR)

        # get root logger and add the bugsnag handler
        root_logger = logging.getLogger()
        root_logger.addHandler(bugsnag_handler)

    def setup_bugsnag(self, app: FastAPI):
        """
        Set up Bugsnag middleware and exception handlers in the FastAPI application
        and configure root logger for sending ERROR level logs to Bugsnag.

        This adds middleware to capture requests and logs to Bugsnag, and attaches the exception handlers
        to the FastAPI app for handling exceptions.

        Args:
            app (FastAPI): The FastAPI app instance to which the middleware and handlers will be added.
        """
        app.add_middleware(BugsnagMiddleware)
        app.add_exception_handler(Exception, self._catch_all_exception_handler)
        app.add_exception_handler(404, self._not_found_exception_handler)
        app.add_exception_handler(RequestValidationError, self._validation_exception_handler)

        self._send_error_level_logs_to_bugsnag()
