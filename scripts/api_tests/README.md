# API Tests

This folder contains standalone API test scripts for major endpoints.

## Prerequisites

- API service is running and reachable (default: `http://localhost:5312/v1`)
- Python virtual environment is available
- Required package installed: `requests`

Install dependency (if needed):

```bash
pip install requests
```

## Configuration

Shared test configuration is centralized in:

- `scripts/api_tests/common.py`

Update these values in `ApiTestConfig` as needed:

- `api_base`
- `auth_token`
- `user_uuid`
- `session_auth`

## How to run

```bash
python scripts/api_tests/chat_sessions/test_chat_sessions.py
python scripts/api_tests/central_rag/test_central_rag.py
python scripts/api_tests/document_management/test_document_management.py
python scripts/api_tests/message_feedback/test_message_feedback.py
python scripts/api_tests/themes_use_cases/test_themes_use_cases.py
python scripts/api_tests/user_data/test_user_data.py
python scripts/api_tests/user_prompts/test_user_prompts.py
```

Or run via Docker:

```bash
docker exec -it api sh -c "python scripts/api_tests/chat_sessions/test_chat_sessions.py"
docker exec -it api sh -c "python scripts/api_tests/central_rag/test_central_rag.py"
docker exec -it api sh -c "python scripts/api_tests/document_management/test_document_management.py"
docker exec -it api sh -c "python scripts/api_tests/message_feedback/test_message_feedback.py"
docker exec -it api sh -c "python scripts/api_tests/themes_use_cases/test_themes_use_cases.py"
docker exec -it api sh -c "python scripts/api_tests/user_data/test_user_data.py"
docker exec -it api sh -c "python scripts/api_tests/user_prompts/test_user_prompts.py"
```


## Notes

- Scripts print `[PASS]` / `[FAIL]` per API call.


