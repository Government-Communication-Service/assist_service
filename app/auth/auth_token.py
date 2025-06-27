import os

from fastapi import Depends, HTTPException

import app.config
from app.api.config import ApiConfig
from app.error_messages import ErrorMessages
from app.logs.logs_handler import LogsHandler

SECRET_KEY = os.getenv("AUTH_SECRET_KEY")
SECRET_KEY2 = os.getenv("AUTH_SECRET_KEY2")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class AuthToken:
    def validate(self, token: str):
        if app.config.BYPASS_AUTH_VALIDATOR:
            return True

        if not token:
            error = HTTPException(
                status_code=403,
                detail=ErrorMessages.not_provided(ApiConfig.AUTH_TOKEN_ALIAS, "header"),
            )
            LogsHandler.error(error, task="validating auth_token")

            raise error

        if token not in [SECRET_KEY, SECRET_KEY2]:
            error = HTTPException(
                status_code=403,
                detail=ErrorMessages.invalid_or_expired(ApiConfig.AUTH_TOKEN_ALIAS, "header"),
            )
            LogsHandler.error(error, task="validating auth_token")
            raise error

        return True


def auth_token_validator(token=ApiConfig.AUTH_TOKEN):
    AuthToken().validate(token)


AUTH_TOKEN_VALIDATOR: str = Depends(auth_token_validator)


# This validator does not rquire a user_uuid to function
def auth_token_validator_no_user(token=ApiConfig.AUTH_TOKEN):
    AuthToken().validate(token)
