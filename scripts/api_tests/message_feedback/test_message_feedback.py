"""
Test the GCS Assist API - Message Feedback endpoints (positive only).

Config:
- API_BASE:    Base URL for the API (default http://localhost:5312/v1)
- AUTH_TOKEN:  Auth-Token
- USER_UUID:   User identifier
- SESSION_AUTH: Session-Auth token

Endpoints exercised:
  GET  /v1/chat/messages/feedback/labels                                list feedback labels
  PUT  /v1/chats/users/{user_uuid}/chats/{message_uuid}/feedback        submit positive feedback
  PUT  /v1/chats/users/{user_uuid}/chats/{message_uuid}/feedback        submit negative feedback (with label)
  PUT  /v1/chats/users/{user_uuid}/chats/{message_uuid}/feedback        remove feedback (score=0)

Note:
  A chat is created automatically to obtain a real message UUID to test against.

Usage:
    python3 scripts/api_tests/message_feedback/test_message_feedback.py
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


def create_chat_and_get_message_uuid(session: requests.Session, headers: dict) -> Optional[str]:
    """
    POST /chats/users/{user_uuid} to create a chat, then pull the first
    assistant message UUID from the response.
    """
    url = f"{API_BASE}/chats/users/{USER_UUID}"
    payload = {
        "query": "Hello, this is a test message for feedback testing.",
        "use_rag": False,
        "use_gov_uk_search_api": False,
    }
    resp = session.post(url, json=payload, headers=headers)
    data = common.assert_ok("POST /chats/users/{user_uuid} (setup — create chat)", resp)
    if not data:
        return None

        # The response may contain messages directly or nested under a key
    messages = data.get("messages") or []
    if not messages:
        chat = data.get("chat") or {}
        messages = chat.get("messages") or []

    if messages:
        message_uuid = str(messages[0].get("uuid", ""))
        print(f"{PRINT_SPACING}message_uuid (for feedback): {message_uuid}")
        return message_uuid

    print("  [WARN] No messages found in create-chat response; trying GET /messages")
    return None


def get_first_message_uuid(session: requests.Session, headers: dict, chat_uuid: str) -> Optional[str]:
    """GET /chats/users/{user_uuid}/chats/{chat_uuid}/messages — return first message UUID."""
    url = f"{API_BASE}/chats/users/{USER_UUID}/chats/{chat_uuid}/messages"
    resp = session.get(url, headers=headers)
    data = common.assert_ok("GET /chats/…/{chat_uuid}/messages (setup — fetch messages)", resp)
    if not data:
        return None
    messages = data.get("messages") or []
    if messages:
        message_uuid = str(messages[0].get("uuid", ""))
        print(f"{PRINT_SPACING}message_uuid (for feedback): {message_uuid}")
        return message_uuid
    return None


# ---------------------------------------------------------------------------
# Feedback tests
# ---------------------------------------------------------------------------

def test_get_feedback_labels(session: requests.Session, headers: dict) -> Optional[str]:
    """GET /chat/messages/feedback/labels — returns list of label UUIDs."""
    url = f"{API_BASE}/chat/messages/feedback/labels"
    resp = session.get(url, headers=headers)
    data = common.assert_ok("GET  /chat/messages/feedback/labels (list labels)", resp)
    if data:
        labels = data if isinstance(data, list) else data.get("labels", [])
        print(f"{PRINT_SPACING}Labels available: {len(labels)}")
        if labels:
            label_uuid = str(labels[0].get("uuid", ""))
            print(f"{PRINT_SPACING}First label uuid: {label_uuid}")
            return label_uuid
    return None


def test_submit_positive_feedback(
    session: requests.Session, headers: dict, message_uuid: str
) -> None:
    """PUT /chats/users/{user_uuid}/chats/{message_uuid}/feedback — score=1 (positive)."""
    url = f"{API_BASE}/chats/users/{USER_UUID}/chats/{message_uuid}/feedback"
    payload = {"score": 1}
    resp = session.put(url, json=payload, headers=headers)
    data = common.assert_ok("PUT  /…/{message_uuid}/feedback (positive, score=1)", resp)
    if data:
        print(f"{PRINT_SPACING}Feedback score: {data.get('score') or data.get('feedback_score')}")


def test_submit_negative_feedback(
    session: requests.Session, headers: dict, message_uuid: str, label_uuid: Optional[str]
) -> None:
    """PUT /chats/users/{user_uuid}/chats/{message_uuid}/feedback — score=-1 (negative)."""
    url = f"{API_BASE}/chats/users/{USER_UUID}/chats/{message_uuid}/feedback"
    payload: dict = {"score": -1, "freetext": "The response was not helpful."}
    if label_uuid:
        payload["label"] = label_uuid
    resp = session.put(url, json=payload, headers=headers)
    common.assert_ok("PUT  /…/{message_uuid}/feedback (negative, score=-1)", resp)


def test_remove_feedback(
    session: requests.Session, headers: dict, message_uuid: str
) -> None:
    """PUT /chats/users/{user_uuid}/chats/{message_uuid}/feedback — score=0 (remove)."""
    url = f"{API_BASE}/chats/users/{USER_UUID}/chats/{message_uuid}/feedback"
    payload = {"score": 0}
    resp = session.put(url, json=payload, headers=headers)
    common.assert_ok("PUT  /…/{message_uuid}/feedback (remove, score=0)", resp)


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

        # ── Setup: create a chat to get a real message UUID ─────────────────
        print("\n=== Setup: Create Chat to Obtain Message UUID ===")
        message_uuid = create_chat_and_get_message_uuid(session, headers)

        if not message_uuid:
            # Fall back: list chats, pick the first chat, then fetch its messages
            list_url = f"{API_BASE}/chats/users/{USER_UUID}/chats"
            list_resp = session.get(list_url, headers=headers)
            try:
                list_data = list_resp.json()
            except ValueError:
                list_data = {}
            chats = list_data.get("chats", [])
            if not chats:
                raise SystemExit("ERROR: No chats found — cannot obtain a message UUID.")
            chat_uuid = str(chats[0].get("uuid", ""))
            message_uuid = get_first_message_uuid(session, headers, chat_uuid)

        if not message_uuid:
            raise SystemExit("ERROR: Could not obtain a message UUID — aborting feedback tests.")

        # ── Message Feedback Tests ──────────────────────────────────────────
        print("\n=== Message Feedback Tests ===")

        # 1. List feedback labels (grab one for use in negative feedback)
        label_uuid = test_get_feedback_labels(session, headers)

        # 2. Submit positive feedback
        test_submit_positive_feedback(session, headers, message_uuid)

        # 3. Submit negative feedback (with optional label)
        test_submit_negative_feedback(session, headers, message_uuid, label_uuid)

        # 4. Remove feedback
        test_remove_feedback(session, headers, message_uuid)

        print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
