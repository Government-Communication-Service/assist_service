"""
Test the GCS Assist API - User Data endpoints (positive only).

Config:
- API_BASE:    Base URL for the API (default http://localhost:5312/v1)
- AUTH_TOKEN:  Auth-Token
- USER_UUID:   User identifier
- SESSION_AUTH: Session-Auth token

Endpoints exercised:
  POST  /v1/users                          create a new user
  PUT   /v1/user/{user_uuid}               update user profile
  GET   /v1/users/{user_uuid}/documents    list user documents

Usage:
    python3 scripts/api_tests/user_data/test_user_data.py
"""

import sys
import uuid
from pathlib import Path
from typing import Optional

import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common import ApiTestCommon

common = ApiTestCommon()
API_BASE = common.api_base
AUTH_TOKEN = common.auth_token
USER_UUID = common.user_uuid
SESSION_AUTH = common.session_auth
PRINT_SPACING = "         "


# ---------------------------------------------------------------------------
# User Data tests
# ---------------------------------------------------------------------------


def test_create_user(session: requests.Session, headers: dict) -> Optional[str]:
    """POST /users — create a brand-new user, returns the new user UUID."""
    url = f"{API_BASE}/users"
    new_uuid = str(uuid.uuid4())
    payload = {
        "uuid": new_uuid,
        "job_title": "Test Engineer",
        "region": "London",
        "sector": "Central Government",
        "organisation": "GCS Test Org",
        "grade": "SEO",
        "communicator_role": True,
    }
    # create user only needs Auth-Token
    auth_only_headers = {"Auth-Token": AUTH_TOKEN}
    resp = session.post(url, json=payload, headers=auth_only_headers)
    data = common.assert_ok("POST /users (create user)", resp)
    if data:
        print(f"{PRINT_SPACING}Created user UUID: {new_uuid}")

        return new_uuid
    return None


def test_update_user(session: requests.Session, headers: dict, target_uuid: str) -> None:
    """PUT /user/{user_uuid} — update user profile fields."""
    url = f"{API_BASE}/user/{target_uuid}"
    payload = {
        "job_title": "Senior AI Engineer",
        "region": "LONDON",
        "sector": "Local Government",
        "organisation": "GCS Updated Org",
        "grade": "G7",
        "communicator_role": False,
    }
    # update user only needs Auth-Token
    auth_only_headers = {"Auth-Token": AUTH_TOKEN}
    resp = session.put(url, json=payload, headers=auth_only_headers)
    data = common.assert_ok("PUT  /user/{user_uuid} (update user profile)", resp)
    if data:
        print(f"{PRINT_SPACING}Response message: {data.get('message')}")


def test_list_user_documents(session: requests.Session, headers: dict) -> None:
    """GET /users/{user_uuid}/documents — list documents for the configured user."""
    url = f"{API_BASE}/users/{USER_UUID}/documents"
    resp = session.get(url, headers=headers)
    data = common.assert_ok("GET  /users/{user_uuid}/documents (list documents)", resp)
    if data:
        user_docs = data.get("user_documents", [])
        central_docs = data.get("central_documents", [])
        print(f"{PRINT_SPACING}User documents   : {len(user_docs)}")
        print(f"{PRINT_SPACING}Central documents: {len(central_docs)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    common.validate_required_config()

    headers = common.base_headers()
    with requests.Session() as session:
        # ── Auth session ────────────────────────────────────────────────────
        print("\n=== Auth Session ===")
        session_auth = SESSION_AUTH or common.create_session(session, headers)
        headers["Session-Auth"] = session_auth

        # ── User Data Tests ─────────────────────────────────────────────────
        print("\n=== User Data Tests ===")

        # 1. Create a new user
        new_user_uuid = test_create_user(session, headers)

        # 2. Update profile of the newly created user (fall back to USER_UUID if creation failed)
        target_uuid = new_user_uuid or USER_UUID
        test_update_user(session, headers, target_uuid)

        # 3. List documents for the configured user
        test_list_user_documents(session, headers)

        print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
