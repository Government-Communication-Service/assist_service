from fastapi import HTTPException
from fastapi.responses import JSONResponse

from app.error_messages import ErrorMessages
from app.logs.logs_handler import LogsHandler


class ApiResponses:
    def error(e, task):
        LogsHandler.error(e, task)
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "status_message": ErrorMessages.default(task, e),
            },
        )

    def http_error(e: HTTPException, task: str):
        LogsHandler().error(e, task)
        return JSONResponse(
            status_code=e.status_code,
            content={"status": "failed", "status_message": e.detail},
        )
