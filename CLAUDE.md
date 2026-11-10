# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Copilot API (now known as "Assist"), a GenAI-powered FastAPI service for the GCS (Government Communication Service) chat project. It provides a REST API with chat functionality, document management, RAG (Retrieval Augmented Generation), and integrates with AWS Bedrock for LLM services.

## Development Commands

### Environment Setup
- **Docker build and start**: `make start` (combines build and up)
- **Start dependencies only**: `make deps` (allows running API locally)
- **Build containers**: `make build`
- **Start containers**: `make up`
- **Stop containers**: `make down`
- **Debug mode**: `make up-debug` (enables debugpy on port 5678)

### Dependencies (uv)
Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. Test, lint and debug tools are in the `dev` dependency group, which isn't installed in production. torch and torchvision come from the PyTorch CPU index via `[tool.uv.sources]`.
- **Add or change a dependency**: edit `pyproject.toml` (or `uv add <pkg>`), then `uv lock`
- **Elastic Beanstalk**: installs with pip from `requirements.txt`, which isn't committed. `.platform/hooks/prebuild/01_generate_requirements.sh` generates it from `uv.lock` on each deploy.
- **Local venv**: `uv sync` (the Docker images install into `/opt/venv`)

### Database Management
- **Reset database**: `make db` (WARNING: deletes all data)
- **Database migrations**: `make migrate` (with interactive message prompt)
- **Apply migrations**: `make db-head`
- **Test database setup**: `make test-db`

### Testing
- **Run all tests**: `make test` (sets up test DB and runs full suite)
- **Specific test suites**:
  - `make test-chat` - Chat functionality tests
  - `make test-bedrock` - AWS Bedrock handler tests (mocked)
  - `make test-central-guidance` - RAG/central guidance tests
  - `make test-document-upload` - Document upload tests
  - `make test-gov-uk-search` - Gov.UK search integration tests
  - `make test-opensearch` - OpenSearch functionality tests

### Code Quality
- **Lint and format**: `make lint` (uses ruff with automatic fixing)
- **Pre-commit hooks**: `pre-commit install` (must run after initial setup)

### Dev Tooling — Observing LLM Requests
Set `LOG_FULL_INVOCATION_REQUEST_TO_FILE_PATH=/tmp/api/invocation.json` in `.env` to make the app write the full Bedrock request (model, system prompt blocks, messages, tools, and token usage) to a file on every LLM call. Each call gets its own numbered file (`invocation_1.json`, `invocation_2.json`, ...) so concurrent calls - e.g. a background compaction call - don't clobber each other. The `/tmp/api` directory is bind-mounted from the host in `docker-compose.yml` so the files are directly accessible. To inspect one as collapsible HTML:

```bash
python scripts/system_prompt/pretty_print_invocation.py /tmp/api/invocation.json
# opens /tmp/api/invocation.html
```

The HTML shows token usage at the top, each system block and message as a collapsible section (cached blocks flagged), and any extra kwargs (tools, tool_choice) in a separate section. Unset or leave empty to disable logging.

## Architecture

### Project Structure
The codebase follows a modular FastAPI structure with domain-specific packages:

```
app/
├── main.py                 # FastAPI app initialization
├── config.py              # Global configuration and environment variables
├── database/               # Database models, sessions, operations
├── api/                    # API layer (endpoints, responses, paths)
├── auth/                   # Authentication and session management
├── bedrock/                # AWS Bedrock LLM integration
├── chat/                   # Core chat functionality
├── central_guidance/       # RAG system for central documentation
├── document_upload/        # Personal document management and RAG
├── gov_uk_search/          # Gov.UK search API integration
├── opensearch/             # OpenSearch service for document indexing
├── themes_use_cases/       # Predefined themes and use cases management
├── user/                   # User management
└── personal_prompts/       # User's personal prompt library
```

Each domain module contains:
- `routes.py` - FastAPI endpoints
- `service.py` - Business logic
- `schemas.py` - Pydantic models
- `queries.py` - Database queries (models themselves are central in `app/database/models.py`)
- `constants.py` - Module-specific constants
- `config.py` - Module-specific configuration

### Key Design Patterns

#### Service Layer Returns Simple Data
Service functions should return simple data structures (dicts, lists) rather than Pydantic models. The API layer is responsible for converting to proper response schemas.

#### Auth system using dependency injection
When creating a new endpoint, you will need to perform some validation of the user input. Compose these validations as dependencies. Configure your dependencies in a thoughtful way such resources needed for business logic are available after validation.

```python
# app/new_module/routes.py

from app.auth.verify_service import (
  verify_auth_token,
  verify_and_get_auth_session_from_header,
  verify_and_get_user_from_path_and_header,
)
from app.database.models import User, AuthSession

@router.post(
  path=ENDPOINTS.NEW_PATH_NAME_1,
  dependencies=[
    Depends(verify_auth_token) # The Auth-Token is not returned, so this dependency is in the decorator.
  ]
)
async def new_endpoint_1(
  db_session: DbSession, # We want the database session in our business logic so we return it here
  user: User = Depends(verify_and_get_user_from_path_and_header), # We want the user object in our business logic so we return it here
  auth_session: AuthSession = Depends(verify_and_get_auth_session_from_header), # We want the auth session object in our business logic so we return it here
) -> ...:
  ...

@router.put(
  path=ENDPOINTS.NEW_PATH_NAME_2,
  # Sometimes you don't need some / any of the resources in the business logic but we still want to perform validation. Therefore, we put the dependencies in the decorator instead.
  dependencies=[
    Depends(verify_auth_token),
    Depends(verify_and_get_auth_session_from_header),
    Depends(verify_and_get_user_from_path_and_header),
  ]
)
async def new_endpoint_2():
  ...
```

Sometimes you will want to create additional validation logic. This validation should always appear as a new dependency injection. The extra validation logic should be written in the utils.py file of the current module which you are developing for.

```python
# app/new_module/utils.py
# e.g. validating a message uuid
from fastapi import Path
from typing import Annotated

from app.auth.utils import verify_and_parse_uuid
from app.database.db_session import DbSession
from app.database.models import Message

async def verify_and_get_message_from_path(message_uuid: Annotated[str, Path(...)], db_session: DbSession) -> Message:
  # Logic to verify and get message
  message = ...
  return message
```

```python
# app/new_module/routes.py
from app.new_module.utils import verify_and_get_message_from_path

@router.get(...)
async def new_endpoint_3(
  message: Message = Depends(verify_and_get_message_from_path)
):
  ...
```

#### Database sessions and queries
Keep sessions short. As a default, avoid holding a session open across an LLM call, a stream or another slow external call, because it ties up a pooled connection and any row locks for the whole call. The usual shape is to commit what you need first, make the call, then open a new session for any writes that follow.

- **Routes** take the request's session with the `DbSession` alias (`app/database/db_session.py`). It commits when the endpoint function returns, before the response is sent or a stream starts. Always use the alias, never `Depends(get_db_session)`: mixing the two in one request gives two separate sessions.
  ```python
  from app.database.db_session import DbSession

  @router.get(...)
  async def new_endpoint(db_session: DbSession) -> ...:
    ...
  ```
- **Queries live in the module that owns the use case**, in `queries.py` (see `app/projects/queries.py`). A cross-domain query belongs to the module whose use case needs it, not a shared file. `DbOperations` (`app/database/db_operations.py`) and `app/database/table.py` are legacy: don't add to them, and move a method into its module's `queries.py` when you change it.
- **There are two kinds of database function.** Keep them visibly separate, by name or by file:
  - *Component functions* take `db_session`, and never commit or open a session. Use `flush()` when you need generated IDs. They're building blocks that combine within one transaction.
  - *Session functions* open their own `async with async_db_session()`, take no session parameter, and are a complete unit of work. `async_db_session` commits on exit and rolls back on an exception. They return plain data or IDs, not ORM objects (e.g. `acquire_compaction_lock` in `app/compaction/service.py`). Calling one never joins the caller's transaction.
- **Group work by transaction, not by query.** Writes that must succeed or fail together share one session, and related reads share one session rather than opening one each.
- **Don't pass ORM objects between sessions.** `expire_on_commit=False` keeps loaded columns readable after a session closes, but lazy-loading a relationship then raises under async. Pass IDs or plain data, or eager-load what you need.
- **Models stay central** in `app/database/models.py`, which Alembic autogenerate and string-named relationships need.
- New queries don't need an `Action` entry or `LogsHandler.with_logging`. Use `logger.exception` where extra context helps.

#### Cross-Module Imports
Be explicit and specific when importing from other packages:
```python
from app.auth.utils import verify_and_parse_uuid
```

#### Free-text input and error messages
- Clean free-text request fields with the validators in `app/schemas/text.py`: `CleanSingleLine` for names and titles, `CleanMultiline` for descriptions, prompts and messages. Add the validator after `StringConstraints` so length limits apply to the cleaned value: `Annotated[str, StringConstraints(max_length=255), CleanSingleLine]`. For text that can't be rejected, such as uploaded filenames, use `strip_to_single_line`.
- JSON bodies containing a NUL character are rejected with 422 globally by `RejectNulBytesMiddleware` (`app/api/nul_bytes_middleware.py`).
- Exception and log messages carry IDs, never user-supplied text such as names, descriptions or queries.

#### Exceptions that become HTTP responses
Subclass `AppError` (`app/exceptions/app_error.py`) in the module's `exceptions.py` and set the mapping as class attributes. One handler turns any subclass into a response, so don't add handlers to `app/exceptions/handlers.py`.

```python
class ProjectNotFoundError(AppError):
    status_code = 404
    detail = "Project not found"  # response body is {"detail": ...}
    expected = True  # info log, no Bugsnag. False (default): error log, reported to Bugsnag
```

- Set `expected = True` for normal API outcomes: not found, conflicts, missing or malformed headers.
- Use properties when the mapping depends on the instance (see `DatabaseError`), and override `content()` only to keep an existing body shape.
- Don't change the status code of an existing error without checking Connect: it retries 401 and 403 responses and then logs the user out.
- Exceptions that are always caught internally stay plain `Exception`.

#### LLM Prompts — `prompts.py` convention

Every module that sends LLM calls keeps its prompt text in a file called `prompts.py` within that module. This makes all prompts easy to locate (`find app -name prompts.py`) and collectively review.

- Prompt text lives as named module-level constants (`SYSTEM_PROMPT_FOO = "..."`) or factory functions that return a string or content-block list.
- Assembly logic (DB queries, feature-flag checks, segment building) lives in the same `prompts.py` — keep prompt text and the code that shapes it together.

The main chat system prompt lives in `app/chat/prompts.py` and is the reference implementation. It returns a `list[dict]` (Bedrock content blocks) with the static portion first and a `cache_control` marker on that block, so the large static text is eligible for prompt caching. Dynamic content (today's date, DB-sourced lists, feature segments) is appended as a second block and is never cached.

#### Style Guide Checks (GOV.UK)
The style guide feature lives in `app/style_guide/` and integrates into chat flows via
`app/style_guide/service.py`. Rules are defined in `app/style_guide/rule_mapping.json`, with
deterministic checks in `style_guide_checker.py` and LLM-assisted summaries in the service layer.
When adjusting rules or behavior, update the relevant tests in `tests/style_guide/`.

### LLM Configuration
The system uses multiple Claude models via AWS Bedrock for different purposes:
- **Chat Response**: `claude-sonnet-5-5` (highest quality)
- **Chat Titles**: `claude-haiku-5-5`
- **Query Generation**: `claude-sonnet-4-6`
- **Index Routing**: `claude-haiku-5-5` (lightweight decisions)
- **Document Review**: `claude-haiku-5-5`

Models are configured in `app/config.py` and can be overridden via environment variables.

### RAG System Architecture
The RAG system operates through multiple components:
1. **Central Guidance**: Uses OpenSearch for GCS documentation retrieval
2. **Personal Documents**: User-uploaded document processing and indexing
3. **Gov.UK Search**: Integration with Gov.UK search API for government content
4. **Query Routing**: LLM-based decision making for which knowledge sources to use

### Database
- **Main DB**: PostgreSQL (`copilot` database)
- **Test DB**: Separate `testcopilot` database, recreated for each test run
- **Migrations**: Alembic-based, stored in `app/alembic/versions/`
- **Models**: SQLAlchemy ORM models, all in `app/database/models.py`

## Environment Configuration

Key environment variables (defined in `.env`):
- `USE_RAG`: Enable/disable RAG functionality
- `DEBUG_MODE`: Enable debugpy debugging
- `LLM_DEFAULT_MODEL`: Override default LLM model
- Various model-specific configurations for different LLM use cases

## Testing Strategy

- **Unit tests**: Test individual functions and methods
- **Integration tests**: Test component interactions
- **E2E tests**: Test full API workflows
- **Test isolation**: Each test run recreates the test database from scratch
- **No live external services**: Tests never call AWS or OpenSearch; only PostgreSQL is real. `tests/conftest.py` sets this up for every test:
  - The chat model is replaced by a scripted fake (`mock_chat_model`; request it to script replies or tool calls). Mock other model calls (`BedrockHandler.invoke_async`) yourself; a test that sends a request to Bedrock fails.
  - OpenSearch is an in-memory fake (`tests/fake_opensearch.py`) with simple word-count scoring, so it can't test search relevance.
  - Other AWS calls made through boto3 (e.g. S3) are served by moto.

## Local Development

1. Copy `.env.example` to `.env` and configure secrets
2. Ensure Docker has at least 4GB memory allocated
3. Run `make start` to build and start all services
4. API available at `http://localhost:5312` (docs at `/docs`)

## Pull Requests

Title pull requests `Type(scope): summary`, e.g. `Fix(chat): stop duplicate citations` or `Refactor(tests): mock Bedrock and OpenSearch`.

## Deployment

The project uses Git-based deployment to AWS:
- **Dev**: `git push aws dev-production`
- **Test**: `git push aws test-production`
- **Production**: `git push aws production`
