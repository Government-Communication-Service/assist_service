"""
Test the GCS Assist API - Document Management endpoints (positive only).

Config:
- API_BASE:    Base URL for the API (default http://localhost:5312/v1)
- AUTH_TOKEN:  Auth-Token
- USER_UUID:   User identifier
- SESSION_AUTH: Session-Auth token

Endpoints exercised:
  GET    /v1/users/{user_uuid}/documents                     list user documents
  POST   /v1/users/{user_uuid}/documents                     upload a document (txt file)
  DELETE /v1/users/{user_uuid}/documents/{document_uuid}     delete a document

Usage:
    python3 scripts/api_tests/document_management/test_document_management.py
"""

import io
import sys
from pathlib import Path
from typing import Optional

import requests

sys.path.append(str(Path(__file__).resolve().parents[1])) # Append the parent directory to sys.path

from common import ApiTestCommon

common = ApiTestCommon()
API_BASE = common.api_base
AUTH_TOKEN = common.auth_token
USER_UUID = common.user_uuid
SESSION_AUTH = common.session_auth
PRINT_SPACING = "         "


# ---------------------------------------------------------------------------
# Document Management tests
# ---------------------------------------------------------------------------

def test_list_documents(session: requests.Session, headers: dict) -> None:
    """GET /users/{user_uuid}/documents"""
    url = f"{API_BASE}/users/{USER_UUID}/documents"
    resp = session.get(url, headers=headers)
    data = common.assert_ok("GET  /users/{user_uuid}/documents (list documents)", resp)
    if data:
        user_docs = data.get("user_documents", [])
        central_docs = data.get("central_documents", [])
        print(f"{PRINT_SPACING}User documents   : {len(user_docs)}")
        print(f"{PRINT_SPACING}Central documents: {len(central_docs)}")


def test_upload_document(session: requests.Session, headers: dict) -> Optional[str]:
    """POST /users/{user_uuid}/documents — upload a small text file, returns document_uuid."""
    url = f"{API_BASE}/users/{USER_UUID}/documents"

    content = (
        "This is a test document uploaded by the automated document management test script.\n"
        "It contains enough text for the document parser to process successfully.\n"
        "Government Communication Service automated testing.\n"
    )
    file_data = io.BytesIO(content.encode("utf-8"))

    files = {
        "file": ("test_document.txt", file_data, "text/plain"),
    }
    form_data = {"description": "Automated test document"}

    resp = session.post(url, files=files, data=form_data, headers=headers)
    data = common.assert_ok("POST /users/{user_uuid}/documents (upload document)", resp)
    if data:
        document_uuid = str(data.get("document_uuid", ""))
        print(f"{PRINT_SPACING}Uploaded document UUID: {document_uuid}")
        print(f"{PRINT_SPACING}Message              : {data.get('message')}")
        return document_uuid
    return None


def test_delete_document(
    session: requests.Session, headers: dict, document_uuid: str
) -> None:
    """DELETE /users/{user_uuid}/documents/{document_uuid}"""
    url = f"{API_BASE}/users/{USER_UUID}/documents/{document_uuid}"
    resp = session.delete(url, headers=headers)
    data = common.assert_ok("DELETE /users/{user_uuid}/documents/{uuid} (delete document)", resp)
    if data:
        print(f"{PRINT_SPACING}Message: {data.get('message')}")


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

        # ── Document Management Tests ───────────────────────────────────────
        print("\n=== Document Management Tests ===")

        # 1. List documents before upload
        test_list_documents(session, headers)

        # 2. Upload a test document
        document_uuid = test_upload_document(session, headers)
        if not document_uuid:
            raise SystemExit("ERROR: Could not upload a document — aborting remaining tests.")

        # 3. List documents after upload (should include the new one)
        test_list_documents(session, headers)

        # 4. Delete the uploaded document
        test_delete_document(session, headers, document_uuid)

        # 5. List documents after deletion (should no longer include the deleted one)
        test_list_documents(session, headers)

        print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
