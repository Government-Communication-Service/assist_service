import logging
import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.api import ENDPOINTS, ApiConfig
from app.auth.auth_token import AuthToken
from app.error_messages import ErrorMessages

api = ENDPOINTS()
logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.auth,
    pytest.mark.unit,
]


@pytest.fixture
def auth_token_validator():
    return AuthToken()


class TestAuthSession:
    def test_auth_token_validator_rejects_empty_auth_token_with_http_403_response(self, auth_token_validator):
        try:
            with patch("app.config.BYPASS_AUTH_VALIDATOR", False):
                result = auth_token_validator.validate("")
                logger.error(f"Expected HTTPException, but got result: {result}")
                pytest.fail("Did not raise HTTPException")
        except HTTPException as exc:
            assert exc.status_code == 403
            assert ErrorMessages.not_provided(ApiConfig.AUTH_TOKEN_ALIAS, "header") in str(exc.detail)

    def test_auth_token_validator_rejects_invalid_auth_token_when_jwt_not_enabled(self, auth_token_validator):
        try:
            with patch("app.config.BYPASS_AUTH_VALIDATOR", False):
                result = auth_token_validator.validate("incorrect_key")
                logger.error(f"Expected HTTPException, but got result: {result}")
                pytest.fail("Did not raise HTTPException")
        except HTTPException as exc:
            assert exc.status_code == 403
            assert ErrorMessages.invalid_or_expired(ApiConfig.AUTH_TOKEN_ALIAS, "header") in str(exc.detail)

    def test_auth_token_validator_validates_correct_auth_token(self, auth_token_validator):
        assert auth_token_validator.validate(os.getenv("AUTH_SECRET_KEY")) is True

    def test_auth_token_validator_bypasses_auth_token_validation_when_bypass_enabled(self, auth_token_validator):
        with patch("app.config.BYPASS_AUTH_VALIDATOR", True):
            assert auth_token_validator.validate("any_token") is True

    def test_auth_token_using_secret_key(self):
        """
        - Verify validation of raw key (`secret-key1`).
        - Disables `BYPASS_AUTH_VALIDATOR` and sets the secret key for test.
        """
        with (
            patch("app.config.BYPASS_AUTH_VALIDATOR", False),
            patch("app.auth.auth_token.SECRET_KEY", "secret-key1"),
        ):
            validated = AuthToken().validate("secret-key1")
        assert validated is True

    def test_auth_token_using_secret_key2(self):
        """
        - Verify validation of raw key (`secret-key2`).
        - Disables `BYPASS_AUTH_VALIDATOR` and sets the secret key for test.
        """
        with (
            patch("app.config.BYPASS_AUTH_VALIDATOR", False),
            patch("app.auth.auth_token.SECRET_KEY2", "secret-key2"),
        ):
            validated = AuthToken().validate("secret-key2")
        assert validated is True
