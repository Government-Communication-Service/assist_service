from dataclasses import dataclass
from typing import Any, Optional

import requests

PRINT_SPACING = "         "


@dataclass(frozen=True)
class ApiTestConfig:
    api_base: str = "http://localhost:5312/v1"
    auth_token: str = ""
    user_uuid: str = ""
    session_auth: str = ""


class ApiTestCommon:
    def __init__(self, config: Optional[ApiTestConfig] = None):
        self.config = config or ApiTestConfig()

    @property
    def api_base(self) -> str:
        return self.config.api_base

    @property
    def auth_token(self) -> str:
        return self.config.auth_token

    @property
    def user_uuid(self) -> str:
        return self.config.user_uuid

    @property
    def session_auth(self) -> str:
        return self.config.session_auth

    def base_headers(self, user_uuid: Optional[str] = None) -> dict:
        return {
            "User-Key-UUID": user_uuid or self.user_uuid,
            "Auth-Token": self.auth_token,
        }

    def ok(self, label: str, status: int) -> None:
        print(f"  [PASS] {label} — HTTP {status}")

    def fail(self, label: str, status: int, body: str) -> None:
        print(f"  [FAIL] {label} — HTTP {status}: {body[:300]}")

    def assert_ok(self, label: str, response: requests.Response) -> Any:
        """Print pass/fail and return parsed JSON on success."""
        text = response.text
        if response.status_code in (200, 201):
            self.ok(label, response.status_code)
            if not text:
                return {}
            try:
                return response.json()
            except ValueError:
                return {}
        self.fail(label, response.status_code, text)
        return None

    def assert_no_content(self, label: str, response: requests.Response) -> bool:
        """Handle 204 No Content responses."""
        if response.status_code == 204:
            self.ok(label, response.status_code)
            return True
        text = response.text
        self.fail(label, response.status_code, text)
        return False

    def create_session(self, session: requests.Session, headers: dict) -> str:
        """POST /auth-sessions — return the Session-Auth token."""
        url = f"{self.api_base}/auth-sessions"
        resp = session.post(url, headers=headers)
        data = self.assert_ok("POST /auth-sessions (create session)", resp)
        if data is None:
            raise RuntimeError("Could not create auth session — check AUTH_TOKEN and USER_UUID.")
        token = (
            resp.headers.get("Session-Auth")
            or data.get("Session-Auth")
            or data.get("session_auth")
            or data.get("uuid")
        )
        if not token:
            raise RuntimeError(f"Session-Auth token not found in response: {data}")
        print(f"{PRINT_SPACING}Session-Auth: {token}")
        return token

    def validate_required_config(self) -> None:
        if not self.auth_token:
            raise SystemExit("ERROR: AUTH_TOKEN is not set. Edit the script and add your token.")
        if not self.user_uuid:
            raise SystemExit("ERROR: USER_UUID is not set. Edit the script and add your user UUID.")
