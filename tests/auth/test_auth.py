import logging

import pytest

from app.api.endpoints import ENDPOINTS
from app.auth.config import AUTH_TOKEN, AUTH_TOKEN_2
from app.auth.verify_service import verify_auth_token

api = ENDPOINTS()
logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.auth,
    pytest.mark.unit,
]


class TestAuthSession:
    def test_verify_auth_token_rejects_empty_auth_token(self):
        try:
            verify_auth_token("")
            pytest.fail("verify_auth_token did not raise an exception when provided with an empty Auth-Token")
        except Exception:
            logger.info("verify_auth_token correctly raised an error when provided with an empty Auth-Token")
            return

    def test_verify_auth_token_rejects_invalid_auth_token(self):
        try:
            verify_auth_token("incorrect_key")
            pytest.fail("verify_auth_token did not raise an exception when provided with an incorrect Auth-Token")
        except Exception:
            logger.info("verify_auth_token correctly raised an error when provided with an empty Auth-Token")

    def test_auth_token_using_secret_key(self):
        validated = verify_auth_token(AUTH_TOKEN)
        assert validated is True

    def test_auth_token_using_secret_key2(self):
        validated = verify_auth_token(AUTH_TOKEN_2)
        assert validated is True
