import uuid

from fastapi import HTTPException, status

from app.logs.logs_handler import LogsHandler, logger


def verify_uuid(label: str, uuid_: str) -> uuid.UUID | HTTPException:
    logger.debug(f"verify_uuid {label} value: {uuid_}.")

    if not uuid_:
        logger.debug(f"{label} is not provided.")
        e = HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{label} is not provided.")
        LogsHandler.error(e)

        raise e
    try:
        uuid_string = str(uuid_)
        uuid_obj = uuid.UUID(uuid_string)
        logger.debug(f"{label} '{uuid_obj}' is a valid UUID.")

        return uuid_obj
    except Exception as ex:
        message = f"{label} '{uuid_}' is not a valid UUID"

        e = HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=message)
        LogsHandler.error(e)
        raise e from ex
