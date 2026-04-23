import random
from pathlib import Path

from tasks import UserProtocol

_TEST_DATA = Path(__file__).parent.parent.parent.parent / "data" / "ignored" / "load_testing"
_FILES = [
    ("small.txt", "text/plain"),
    ("medium.txt", "text/plain"),
    ("large.txt", "text/plain"),
    ("sample.csv", "text/csv"),
]
# Weight uploads toward txt files (cheaper) but include variety
_WEIGHTS = [2, 3, 3, 1]

_missing = [f for f, _ in _FILES if not (_TEST_DATA / f).exists()]
if _missing:
    raise RuntimeError(
        f"Missing test data files in {_TEST_DATA}: {', '.join(_missing)}\nSee README.md for setup instructions."
    )


def upload_document(user: UserProtocol) -> None:
    filename, mime = random.choices(_FILES, weights=_WEIGHTS, k=1)[0]
    filepath = _TEST_DATA / filename
    file_bytes = filepath.read_bytes()

    with user.client.post(
        f"/v1/users/{user.user_uuid}/documents",
        headers=user.auth_headers,
        files={"file": (filename, file_bytes, mime)},
        catch_response=True,
        name="POST /v1/users/{uuid}/documents",
    ) as resp:
        if resp.status_code == 200:
            doc_uuid = resp.json().get("document_uuid")
            if doc_uuid:
                user.uploaded_document_uuids.append(doc_uuid)
            resp.success()
        # elif resp.status_code in (400, 422):
        #     # Expected for unsupported format or no text content
        #     resp.success()
        else:
            resp.failure(f"Unexpected {resp.status_code}: {resp.text[:200]}")


def list_documents(user: UserProtocol) -> None:
    user.client.get(
        f"/v1/users/{user.user_uuid}/documents",
        headers=user.auth_headers,
        name="GET /v1/users/{uuid}/documents",
    )


def delete_document(user: UserProtocol) -> None:
    if not user.uploaded_document_uuids:
        return
    doc_uuid = user.uploaded_document_uuids.pop(0)
    with user.client.delete(
        f"/v1/users/{user.user_uuid}/documents/{doc_uuid}",
        headers=user.auth_headers,
        catch_response=True,
        name="DELETE /v1/users/{uuid}/documents/{doc_uuid}",
    ) as resp:
        if resp.status_code in (200, 404):
            resp.success()
        else:
            resp.failure(f"Unexpected {resp.status_code}: {resp.text[:200]}")
