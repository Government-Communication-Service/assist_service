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

### Database Management
- **Reset database**: `make db` (WARNING: deletes all data)
- **Database migrations**: `make migrate` (with interactive message prompt)
- **Apply migrations**: `make db-head`
- **Test database setup**: `make test-db`

### Testing
- **Run all tests**: `make test` (sets up test DB and runs full suite)
- **Specific test suites**:
  - `make test-chat` - Chat functionality tests
  - `make test-bedrock` - AWS Bedrock integration tests
  - `make test-central-guidance` - RAG/central guidance tests
  - `make test-document-upload` - Document upload tests
  - `make test-gov-uk-search` - Gov.UK search integration tests
  - `make test-opensearch` - OpenSearch functionality tests

### Code Quality
- **Lint and format**: `make lint` (uses ruff with automatic fixing)
- **Pre-commit hooks**: `pre-commit install` (must run after initial setup)

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
- `models.py` - Database models (SQLAlchemy)
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
  db_session: AsyncSession = Depends(get_db_session), # We want the database session in our business logic so we return it here
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
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated

from app.auth.utils import verify_and_parse_uuid
from app.database.db_session import get_db_session
from app.database.models import Message

async def verify_and_get_message_from_path(message_uuid: Annotated[str, Path(...)], db_session: AsyncSession = Depends(get_db_session)) -> Message:
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

#### Database Session Injection
Use dependency injection for database sessions at the API level, then pass the same session through the service layer functions to maintain consistency.
```python
from app.database.db_session import get_db_session
from fastapi import Depends

@app.get(...)
async def new_endpoint(
  db_session=Depends(get_db_session)
) -> ...:
  ...
```

#### Cross-Module Imports
Be explicit and specific when importing from other packages:
```python
from app.auth.utils import verify_and_parse_uuid
```

### LLM Configuration
The system uses multiple Claude models via AWS Bedrock for different purposes:
- **Chat Response**: `claude-sonnet-4-20250514-v1:0` (highest quality)
- **Chat Titles**: `claude-3-7-sonnet-20250219-v1:0`
- **Query Generation**: `claude-3-7-sonnet-20250219-v1:0`
- **Index Routing**: `claude-3-5-haiku-20241022-v1:0` (lightweight decisions)
- **Document Review**: `claude-3-5-haiku-20241022-v1:0`

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
- **Models**: SQLAlchemy ORM models in domain-specific `models.py` files

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

## Local Development

1. Copy `.env.example` to `.env` and configure secrets
2. Ensure Docker has at least 4GB memory allocated
3. Run `make start` to build and start all services
4. API available at `http://localhost:5312` (docs at `/docs`)
5. Database admin at `http://localhost:4040`

## Deployment

The project uses Git-based deployment to AWS:
- **Dev**: `git push aws dev-production`
- **Test**: `git push aws test-production`
- **Production**: `git push aws production`
