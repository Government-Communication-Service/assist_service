"""
Upload many tiny test documents to the API for testing.

Config:
- API_BASE: Base URL for the API (default http://localhost:5312/v1)
- AUTH_TOKEN: Auth token for the API
- USER_UUID: User identifier
- SESSION_AUTH: Session auth token (set manually if required)
- DOCUMENT_COUNT: Number of documents to upload
- CONCURRENCY: Parallel uploads (increase for speed, decrease if server throttles)

Usage:
    /usr/bin/python3 /test.py

LOAD TEST: DELETE EXPIRED DOCUMENTS

Description:
        This workflow performs a load test on the system's document expiration and
        cleanup services. It simulates a high-volume user scenario to ensure that
    expired documents are correctly identified and deleted from the database.

Workflow Steps:
    1. Generate Load (Python Script: scripts/upload_documents/upload_test_documents.py):
             - Uses "aiohttp" to asynchronously upload a large number of documents
        to the API endpoint "/users/{USER_UUID}/documents".
       - Configuration variables such as "API_BASE", "AUTH_TOKEN", and "DOCUMENT_COUNT"
         "DOCUMENT_COUNT" control the scale of the test.

    2. Simulate Expiration (SQL):
             - Executes a SQL update on the "document_user_mapping" table within the
         PostgreSQL container.
             - Sets the "expired_at" timestamp to the current time ("NOW()"), making
         all documents immediately eligible for deletion.
         Command: "UPDATE document_user_mapping SET expired_at = NOW();
                 Docker command: docker compose exec postgres psql -U postgres -d copilot -c
                 "UPDATE document_user_mappingSET expired_at = NOW();"

    3. Verify Deletion:
       - Triggers the expired documents service ("/v1/documents/expired").
             - Queries the user's document list ("/v1/users/{user_uuid}/documents")
         to confirm it is empty, indicating successful cleanup.


"""


import asyncio
import io

import aiohttp

API_BASE = "http://localhost:5312/v1"  # Replace with your actual API base URL.
AUTH_TOKEN = ""  # Replace with your actual auth token.
USER_UUID = ""  # Replace with your actual user UUID.
SESSION_AUTH = ""  # Optional: set if your API requires a session auth token.
DOCUMENT_COUNT = 50000  # Adjust as needed.
CONCURRENCY = 5  # Adjust based on server capacity.



async def create_session(session: aiohttp.ClientSession, headers) -> str:
    """Create an auth session and return the Session-Auth UUID."""
    for path in ("sessions", "session"):
        url = f"{API_BASE}/{path}"
        async with session.post(url, headers=headers) as response:
            if response.status in (200, 201):
                data = await response.json()
                session_auth = (
                    response.headers.get("Session-Auth")
                    or data.get("Session-Auth")
                    or data.get("session_auth")
                )
                print(f"Created session: {session_auth}")
                return session_auth
            if response.status == 404:
                continue
            text = await response.text()
            raise Exception(f"Failed to create session: {response.status} - {text}")

    print("Session endpoint not found; no Session-Auth available.")
    return ""


async def upload_document(session: aiohttp.ClientSession, index: int, headers):
    """Upload a single tiny document."""
    url = f"{API_BASE}/users/{USER_UUID}/documents"

    # Create a tiny text file in memory
    content = f"test word {index}"
    file_data = io.BytesIO(content.encode())

    form = aiohttp.FormData()
    form.add_field("file", file_data, filename=f"test_doc_{index}.txt", content_type="text/plain")
    form.add_field("description", f"Test document {index}")

    async with session.post(url, data=form, headers=headers) as response:
        if response.status == 200:
            result = await response.json()
            return result
        text = await response.text()
        print(f"Doc {index}: {response.status} - {text[:200]}")
        return None


async def bounded_upload(semaphore: asyncio.Semaphore, session: aiohttp.ClientSession, index: int, headers):
    async with semaphore:
        return await upload_document(session, index, headers)


async def main():
    print(f"Uploading {DOCUMENT_COUNT} documents to {API_BASE}...")
    print(f"User UUID: {USER_UUID}")

    headers = {
        "User-Key-UUID": USER_UUID,
        "Auth-Token": AUTH_TOKEN,
    }
    if SESSION_AUTH:
        headers["Session-Auth"] = SESSION_AUTH
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Step 1: Use provided session auth or create one
        session_auth = SESSION_AUTH or await create_session(session, headers)
        if session_auth:
            headers["Session-Auth"] = session_auth
        else:
            raise Exception(
                "Session-Auth is required. Set SESSION_AUTH or fix session creation endpoint."
            )

        # Step 2: Upload documents concurrently
        semaphore = asyncio.Semaphore(CONCURRENCY)
        tasks = [bounded_upload(semaphore, session, i, headers) for i in range(DOCUMENT_COUNT)]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        success = sum(1 for r in results if r is not None)
        print(f"Created {success}/{DOCUMENT_COUNT} documents")


if __name__ == "__main__":
    asyncio.run(main())
