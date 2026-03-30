"""
Test the GCS Assist API - Chat Sessions endpoints (positive only).

Config:
- API_BASE:    Base URL for the API (default http://localhost:5312/v1)
- AUTH_TOKEN:  Auth-Token
- USER_UUID:   User identifier
- SESSION_AUTH: Session-Auth token

Endpoints exercised:
  POST   /v1/auth-sessions                                          create session
  POST   /v1/chats/users/{user_uuid}                                create chat
  GET    /v1/chats/users/{user_uuid}/chats                          list user chats
  GET    /v1/chats/users/{user_uuid}/chats/{chat_uuid}              get chat item
  GET    /v1/chats/users/{user_uuid}/chats/{chat_uuid}/messages     get messages
  GET    /v1/chats/users/{user_uuid}/chats/{chat_uuid}/title        get title
  PATCH  /v1/chats/users/{user_uuid}/chats/{chat_uuid}/title        update title
  PATCH  /v1/chats/users/{user_uuid}/chats/{chat_uuid}/favourite    toggle favourite
  PATCH  /v1/chats/users/{user_uuid}/chats/{chat_uuid}/archive      archive chat

Usage:
    python3 scripts/chat_sessions/test_chat_sessions.py
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
# Step 2 – Chat CRUD
# ---------------------------------------------------------------------------

def test_create_chat(session: requests.Session, headers: dict) -> Optional[str]:
    """POST /chats/users/{user_uuid} — returns chat_uuid."""
    url = f"{API_BASE}/chats/users/{USER_UUID}"
    payload = {
        "query": "Hello, this is a test message from the automated test script.",
        "use_rag": False,
        "use_gov_uk_search_api": False,
    }
    resp = session.post(url, json=payload, headers=headers)
    data = common.assert_ok("POST /chats/users/{user_uuid} (create chat)", resp)
    if data:
        chat_uuid = data.get("uuid") or (data.get("chat") or {}).get("uuid")
        print(f"{PRINT_SPACING}chat_uuid: {chat_uuid}")
        return str(chat_uuid)
    return None


def test_list_user_chats(session: requests.Session, headers: dict) -> None:
    """GET /chats/users/{user_uuid}/chats"""
    url = f"{API_BASE}/chats/users/{USER_UUID}/chats"
    resp = session.get(url, headers=headers)
    data = common.assert_ok("GET  /chats/users/{user_uuid}/chats (list chats)", resp)
    if data:
        chats = data.get("chats", [])
        print(f"{PRINT_SPACING}Total chats returned: {len(chats)}")


def test_get_chat_item(session: requests.Session, headers: dict, chat_uuid: str) -> None:
    """GET /chats/users/{user_uuid}/chats/{chat_uuid}"""
    url = f"{API_BASE}/chats/users/{USER_UUID}/chats/{chat_uuid}"
    resp = session.get(url, headers=headers)
    common.assert_ok("GET  /chats/…/{chat_uuid} (get chat item)", resp)


def test_get_chat_messages(session: requests.Session, headers: dict, chat_uuid: str) -> None:
    """GET /chats/users/{user_uuid}/chats/{chat_uuid}/messages"""
    url = f"{API_BASE}/chats/users/{USER_UUID}/chats/{chat_uuid}/messages"
    resp = session.get(url, headers=headers)
    data = common.assert_ok("GET  /chats/…/{chat_uuid}/messages (get messages)", resp)
    if data:
        messages = data.get("messages", [])
        print(f"{PRINT_SPACING}Messages in chat: {len(messages)}")


def test_get_chat_title(session: requests.Session, headers: dict, chat_uuid: str) -> None:
    """GET /chats/users/{user_uuid}/chats/{chat_uuid}/title"""
    url = f"{API_BASE}/chats/users/{USER_UUID}/chats/{chat_uuid}/title"
    resp = session.get(url, headers=headers)
    data = common.assert_ok("GET  /chats/…/{chat_uuid}/title (get title)", resp)
    if data:
        print(f"{PRINT_SPACING}Current title: {data.get('title')}")


def test_patch_chat_title(session: requests.Session, headers: dict, chat_uuid: str) -> None:
    """PATCH /chats/users/{user_uuid}/chats/{chat_uuid}/title"""
    url = f"{API_BASE}/chats/users/{USER_UUID}/chats/{chat_uuid}/title"
    payload = {"title": "Updated test chat title"}
    resp = session.patch(url, json=payload, headers=headers)
    data = common.assert_ok("PATCH /chats/…/{chat_uuid}/title (update title)", resp)
    if data:
        print(f"{PRINT_SPACING}New title: {data.get('title')}")


def test_patch_chat_favourite(session: requests.Session, headers: dict, chat_uuid: str) -> None:
    """PATCH /chats/users/{user_uuid}/chats/{chat_uuid}/favourite"""
    url = f"{API_BASE}/chats/users/{USER_UUID}/chats/{chat_uuid}/favourite"
    payload = {"favourite": True}
    resp = session.patch(url, json=payload, headers=headers)
    common.assert_ok("PATCH /chats/…/{chat_uuid}/favourite (set favourite=True)", resp)


def test_patch_chat_archive(session: requests.Session, headers: dict, chat_uuid: str) -> None:
    """PATCH /chats/users/{user_uuid}/chats/{chat_uuid}/archive"""
    url = f"{API_BASE}/chats/users/{USER_UUID}/chats/{chat_uuid}/archive"
    resp = session.patch(url, headers=headers)
    common.assert_ok("PATCH /chats/…/{chat_uuid}/archive (archive chat)", resp)


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

        # ── Chat sessions ───────────────────────────────────────────────────
        print("\n=== Chat Session Tests ===")

        # 1. Create a new chat
        chat_uuid = test_create_chat(session, headers)
        if not chat_uuid:
            raise SystemExit("ERROR: Could not create a chat — aborting remaining tests.")

        # 2. List all user chats
        test_list_user_chats(session, headers)

        # 3. Get the specific chat item
        test_get_chat_item(session, headers, chat_uuid)

        # 4. Get messages in the chat
        test_get_chat_messages(session, headers, chat_uuid)

        # 5. Get chat title
        test_get_chat_title(session, headers, chat_uuid)

        # 6. Update chat title
        test_patch_chat_title(session, headers, chat_uuid)

        # 7. Toggle favourite
        test_patch_chat_favourite(session, headers, chat_uuid)

        # 8. Archive the chat
        test_patch_chat_archive(session, headers, chat_uuid)

        print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
