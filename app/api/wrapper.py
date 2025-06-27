from fastapi import HTTPException

from app.api.responses import ApiResponses
from app.config import env_variable
from app.database.database_exception import DatabaseError
from app.logs.logs_handler import logger

TEST_API_FAILURES = env_variable("TEST_API_FAILURES")


def api_wrapper(task):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                if TEST_API_FAILURES:
                    raise Exception("api_wrapper failure invoke")

                res = func(*args, **kwargs)
                return res
            except DatabaseError as database_exception:
                raise database_exception
            except HTTPException as e:
                response = ApiResponses.http_error(e, task)
                logger.error(response)
                return response
            except Exception as e:
                return ApiResponses.error(e, task)

        return wrapper

    return decorator
