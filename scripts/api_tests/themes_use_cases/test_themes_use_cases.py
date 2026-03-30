"""
Test the GCS Assist API - Themes / Use Cases endpoints (positive only).

Config:
- API_BASE:    Base URL for the API (default http://localhost:5312/v1)
- AUTH_TOKEN:  Auth-Token
- USER_UUID:   User identifier
- SESSION_AUTH: Session-Auth token

Endpoints exercised:
  GET    /v1/prompts/themes                                            list all themes
  POST   /v1/prompts/themes                                            create theme
  GET    /v1/prompts/themes/{theme_uuid}                               get theme
  PUT    /v1/prompts/themes/{theme_uuid}                               update theme
  POST   /v1/prompts/themes/{theme_uuid}/use-cases                     create use case
  GET    /v1/prompts/themes/{theme_uuid}/use-cases                     list use cases
  GET    /v1/prompts/themes/{theme_uuid}/use-cases/{use_case_uuid}     get use case
  GET    /v1/prompts/use-cases/{use_case_uuid}                         get use case (no theme required)
  PUT    /v1/prompts/themes/{theme_uuid}/use-cases/{use_case_uuid}     update use case
  DELETE /v1/prompts/themes/{theme_uuid}/use-cases/{use_case_uuid}     delete use case
  DELETE /v1/prompts/themes/{theme_uuid}                               delete theme
  GET    /v1/prompts/bulk                                              bulk get prompts

Usage:
    python3 scripts/api_tests/themes_use_cases/test_themes_use_cases.py
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
# Theme tests
# ---------------------------------------------------------------------------

def test_list_themes(session: requests.Session, headers: dict) -> None:
    """GET /prompts/themes"""
    url = f"{API_BASE}/prompts/themes"
    resp = session.get(url, headers=headers)
    data = common.assert_ok("GET  /prompts/themes (list themes)", resp)
    if data:
        print(f"{PRINT_SPACING}Themes found: {len(data.get('themes', []))}")


def test_create_theme(session: requests.Session, headers: dict) -> Optional[str]:
    """POST /prompts/themes — returns new theme_uuid."""
    url = f"{API_BASE}/prompts/themes"
    payload = {
        "title": "Test Theme Title",
        "subtitle": "Test theme subtitle for automated testing",
        "position": 999,
    }
    resp = session.post(url, json=payload, headers=headers)
    data = common.assert_ok("POST /prompts/themes (create theme)", resp)
    if data:
        theme_uuid = str(data.get("uuid", ""))
        print(f"{PRINT_SPACING}Created theme UUID : {theme_uuid}")
        print(f"{PRINT_SPACING}Title              : {data.get('title')}")
        return theme_uuid
    return None


def test_get_theme(session: requests.Session, headers: dict, theme_uuid: str) -> None:
    """GET /prompts/themes/{theme_uuid}"""
    url = f"{API_BASE}/prompts/themes/{theme_uuid}"
    resp = session.get(url, headers=headers)
    data = common.assert_ok("GET  /prompts/themes/{theme_uuid} (get theme)", resp)
    if data:
        print(f"{PRINT_SPACING}Theme title: {data.get('title')}")


def test_update_theme(session: requests.Session, headers: dict, theme_uuid: str) -> None:
    """PUT /prompts/themes/{theme_uuid}"""
    url = f"{API_BASE}/prompts/themes/{theme_uuid}"
    payload = {
        "title": "Updated Theme Title",
        "subtitle": "Updated subtitle for automated testing",
        "position": 999,
    }
    resp = session.put(url, json=payload, headers=headers)
    data = common.assert_ok("PUT  /prompts/themes/{theme_uuid} (update theme)", resp)
    if data:
        print(f"{PRINT_SPACING}Updated title: {data.get('title')}")


# ---------------------------------------------------------------------------
# Use Case tests
# ---------------------------------------------------------------------------

def test_create_use_case(
    session: requests.Session, headers: dict, theme_uuid: str
) -> Optional[str]:
    """POST /prompts/themes/{theme_uuid}/use-cases — returns new use_case_uuid."""
    url = f"{API_BASE}/prompts/themes/{theme_uuid}/use-cases"
    payload = {
        "title": "Test Use Case Title",
        "instruction": "This is a test instruction for the automated test use case.",
        "user_input_form": "Enter your test input here.",
        "position": 999,
    }
    resp = session.post(url, json=payload, headers=headers)
    data = common.assert_ok("POST /prompts/themes/{uuid}/use-cases (create use case)", resp)
    if data:
        use_case_uuid = str(data.get("uuid", ""))
        print(f"{PRINT_SPACING}Created use case UUID: {use_case_uuid}")
        print(f"{PRINT_SPACING}Title               : {data.get('title')}")
        return use_case_uuid
    return None


def test_list_use_cases(
    session: requests.Session, headers: dict, theme_uuid: str
) -> None:
    """GET /prompts/themes/{theme_uuid}/use-cases"""
    url = f"{API_BASE}/prompts/themes/{theme_uuid}/use-cases"
    resp = session.get(url, headers=headers)
    data = common.assert_ok("GET  /prompts/themes/{uuid}/use-cases (list use cases)", resp)
    if data:
        print(f"{PRINT_SPACING}Use cases found: {len(data.get('use_cases', []))}")


def test_get_use_case(
    session: requests.Session, headers: dict, theme_uuid: str, use_case_uuid: str
) -> None:
    """GET /prompts/themes/{theme_uuid}/use-cases/{use_case_uuid}"""
    url = f"{API_BASE}/prompts/themes/{theme_uuid}/use-cases/{use_case_uuid}"
    resp = session.get(url, headers=headers)
    data = common.assert_ok("GET  /prompts/themes/{uuid}/use-cases/{uuid} (get use case)", resp)
    if data:
        print(f"{PRINT_SPACING}Use case title: {data.get('title')}")


def test_get_use_case_without_theme(
    session: requests.Session, headers: dict, use_case_uuid: str
) -> None:
    """GET /prompts/use-cases/{use_case_uuid}"""
    url = f"{API_BASE}/prompts/use-cases/{use_case_uuid}"
    resp = session.get(url, headers=headers)
    common.assert_ok("GET  /prompts/use-cases/{uuid} (get use case — no theme required)", resp)


def test_update_use_case(
    session: requests.Session, headers: dict, theme_uuid: str, use_case_uuid: str
) -> None:
    """PUT /prompts/themes/{theme_uuid}/use-cases/{use_case_uuid}"""
    url = f"{API_BASE}/prompts/themes/{theme_uuid}/use-cases/{use_case_uuid}"
    payload = {
        "title": "Updated Use Case Title",
        "instruction": "Updated instruction for the automated test use case.",
        "user_input_form": "Updated user input form text.",
        "position": 999,
        "theme_uuid": theme_uuid,
    }
    resp = session.put(url, json=payload, headers=headers)
    data = common.assert_ok("PUT  /prompts/themes/{uuid}/use-cases/{uuid} (update use case)", resp)
    if data:
        print(f"{PRINT_SPACING}Updated title: {data.get('title')}")


def test_delete_use_case(
    session: requests.Session, headers: dict, theme_uuid: str, use_case_uuid: str
) -> None:
    """DELETE /prompts/themes/{theme_uuid}/use-cases/{use_case_uuid}"""
    url = f"{API_BASE}/prompts/themes/{theme_uuid}/use-cases/{use_case_uuid}"
    resp = session.delete(url, headers=headers)
    common.assert_ok("DELETE /prompts/themes/{uuid}/use-cases/{uuid} (delete use case)", resp)


def test_delete_theme(
    session: requests.Session, headers: dict, theme_uuid: str
) -> None:
    """DELETE /prompts/themes/{theme_uuid}"""
    url = f"{API_BASE}/prompts/themes/{theme_uuid}"
    resp = session.delete(url, headers=headers)
    common.assert_ok("DELETE /prompts/themes/{theme_uuid} (delete theme)", resp)


def test_bulk_get_prompts(session: requests.Session, headers: dict) -> None:
    """GET /prompts/bulk"""
    url = f"{API_BASE}/prompts/bulk"
    resp = session.get(url, headers=headers)
    data = common.assert_ok("GET  /prompts/bulk (bulk get prompts)", resp)
    if data:
        print(f"{PRINT_SPACING}Prompts in bulk response: {len(data.get('prompts', []))}")


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

        # ── Themes / Use Cases Tests ────────────────────────────────────────
        print("\n=== Themes / Use Cases Tests ===")

        # 1. List existing themes
        test_list_themes(session, headers)

        # 2. Create a theme
        theme_uuid = test_create_theme(session, headers)
        if not theme_uuid:
            raise SystemExit("ERROR: Could not create a theme — aborting remaining tests.")

        # 3. Get the theme
        test_get_theme(session, headers, theme_uuid)

        # 4. Update the theme
        test_update_theme(session, headers, theme_uuid)

        # 5. Create a use case under the theme
        use_case_uuid = test_create_use_case(session, headers, theme_uuid)
        if not use_case_uuid:
            raise SystemExit("ERROR: Could not create a use case — aborting remaining tests.")

        # 6. List use cases under the theme
        test_list_use_cases(session, headers, theme_uuid)

        # 7. Get the use case (with theme)
        test_get_use_case(session, headers, theme_uuid, use_case_uuid)

        # 8. Get the use case (without theme)
        test_get_use_case_without_theme(session, headers, use_case_uuid)

        # 9. Update the use case
        test_update_use_case(session, headers, theme_uuid, use_case_uuid)

        # 10. Bulk get all prompts
        test_bulk_get_prompts(session, headers)

        # 11. Delete use case
        test_delete_use_case(session, headers, theme_uuid, use_case_uuid)

        # 12. Delete theme
        test_delete_theme(session, headers, theme_uuid)

        print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
