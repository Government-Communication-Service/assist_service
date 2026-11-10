from fastapi import Request
from fastapi.responses import JSONResponse, Response

from app.exceptions.app_error import AppError
from app.logs.logs_handler import logger


async def app_error_handler(request: Request, exc: AppError) -> Response:
    """Builds the response for any `AppError` from its attributes, logging at a level set by `exc.expected`."""
    error_name = type(exc).__name__
    if exc.expected:
        logger.info("%s in endpoint %s %s", error_name, request.method, request.url.path)
    else:
        logger.error("%s in endpoint %s %s", error_name, request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=exc.status_code, content=exc.content())


def register_exception_handlers(app):
    """Register all exception handlers with the FastAPI app."""
    app.exception_handler(AppError)(app_error_handler)
