"""
Test the GCS Assist API - User Prompts endpoints (positive only).

Config:
- API_BASE:    Base URL for the API (default http://localhost:5312/v1)
- AUTH_TOKEN:  Auth-Token
- USER_UUID:   User identifier
- SESSION_AUTH: Session-Auth token

Endpoints exercised:
  GET    /v1/users/{user_uuid}/prompts                            list user prompts
  POST   /v1/users/{user_uuid}/prompts                            create user prompt
  GET    /v1/users/{user_uuid}/prompts/{user_prompt_uuid}         get single prompt
  PATCH  /v1/users/{user_uuid}/prompts/{user_prompt_uuid}         update prompt
  DELETE /v1/users/{user_uuid}/prompts/{user_prompt_uuid}         delete prompt

Usage:
    python3 scripts/api_tests/user_prompts/test_user_prompts.py
"""

import sys
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
# User Prompts tests
# ---------------------------------------------------------------------------

def test_list_user_prompts(session: requests.Session, headers: dict) -> None:
    """GET /users/{user_uuid}/prompts"""
    url = f"{API_BASE}/users/{USER_UUID}/prompts"
    resp = session.get(url, headers=headers)
    data = common.assert_ok("GET  /users/{user_uuid}/prompts (list prompts)", resp)
    if data:
        prompts = data.get("user_prompts", [])
        print(f"{PRINT_SPACING}Prompts found: {len(prompts)}")


def test_create_user_prompt(session: requests.Session, headers: dict) -> Optional[str]:
    """POST /users/{user_uuid}/prompts — returns the new prompt UUID."""
    url = f"{API_BASE}/users/{USER_UUID}/prompts"
    payload = {
        "title": "Test Prompt Title",
        "content": "This is the content of a test prompt created by the automated test script.",
    }
    resp = session.post(url, json=payload, headers=headers)
    data = common.assert_ok("POST /users/{user_uuid}/prompts (create prompt)", resp)
    if data:
        prompt_uuid = str(data.get("uuid", ""))
        print(f"{PRINT_SPACING}Created prompt UUID: {prompt_uuid}")
        print(f"{PRINT_SPACING}Title             : {data.get('title')}")
        return prompt_uuid
    return None


def test_get_user_prompt(
    session: requests.Session, headers: dict, prompt_uuid: str
) -> None:
    """GET /users/{user_uuid}/prompts/{user_prompt_uuid}"""
    url = f"{API_BASE}/users/{USER_UUID}/prompts/{prompt_uuid}"
    resp = session.get(url, headers=headers)
    data = common.assert_ok("GET  /users/{user_uuid}/prompts/{uuid} (get prompt)", resp)
    if data:
        print(f"{PRINT_SPACING}Prompt title: {data.get('title')}")


def test_patch_user_prompt(
    session: requests.Session, headers: dict, prompt_uuid: str
) -> None:
    """PATCH /users/{user_uuid}/prompts/{user_prompt_uuid}"""
    url = f"{API_BASE}/users/{USER_UUID}/prompts/{prompt_uuid}"
    payload = {
        "title": "Updated Prompt Title",
        "content": "This is the updated content of the test prompt.",
    }
    resp = session.patch(url, json=payload, headers=headers)
    data = common.assert_ok("PATCH /users/{user_uuid}/prompts/{uuid} (update prompt)", resp)
    if data:
        print(f"{PRINT_SPACING}Updated title: {data.get('title')}")


def test_delete_user_prompt(
    session: requests.Session, headers: dict, prompt_uuid: str
) -> None:
    """DELETE /users/{user_uuid}/prompts/{user_prompt_uuid}"""
    url = f"{API_BASE}/users/{USER_UUID}/prompts/{prompt_uuid}"
    resp = session.delete(url, headers=headers)
    common.assert_no_content(
        "DELETE /users/{user_uuid}/prompts/{uuid} (delete prompt)", resp
    )


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

        # ── User Prompts Tests ──────────────────────────────────────────────
        print("\n=== User Prompts Tests ===")

        # 1. List existing prompts
        test_list_user_prompts(session, headers)

        # 2. Create a new prompt
        prompt_uuid = test_create_user_prompt(session, headers)
        if not prompt_uuid:
            raise SystemExit("ERROR: Could not create a prompt — aborting remaining tests.")

        # 3. Get the newly created prompt
        test_get_user_prompt(session, headers, prompt_uuid)

        # 4. Update the prompt
        test_patch_user_prompt(session, headers, prompt_uuid)

        # 5. Delete the prompt
        test_delete_user_prompt(session, headers, prompt_uuid)

        print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
