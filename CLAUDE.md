# CLAUDE.md

Copilot API ("Assist"): a GenAI FastAPI service for the GCS (Government Communication Service) chat project, providing chat, document management and RAG on AWS Bedrock. The API serves the Assist app, part of the Connect website.

## Commands

- **Setup**: copy `.env.example` to `.env` and fill in the secrets. `make up` runs, or `make start` builds and runs (`make down` stops), `make deps` starts the dependencies only, and `make up-debug` runs with debugpy on port 5678. The API is at `http://localhost:5312`, docs at `/docs`.
- **Dependencies (uv)**: edit `pyproject.toml` (or `uv add <pkg>`), then `uv lock`. Use `uv sync` for a local venv (Docker installs into `/opt/venv`). Dev tools are in the `dev` group, which isn't installed in production. torch comes from the PyTorch CPU index (`[tool.uv.sources]`). Elastic Beanstalk installs with pip from a `requirements.txt` that `.platform/hooks/prebuild/01_generate_requirements.sh` generates from `uv.lock`.
- **Database**: `make db` resets the dev DB (deletes all data), `make db-head` applies migrations. Migrations are in `app/alembic/versions/`. Create them with the `alembic-migration` skill.
- **Lint**: Ruff runs during `pre-commit`. Run `pre-commit install` once.
- **Make targets** work without a TTY.

### Testing
- `make test` runs the full suite and `make test-<module>` runs one module (e.g. `make test-chat`). Failures print one line each. For full tracebacks, add `PYTEST_ARGS=` (e.g. `make test-chat PYTEST_ARGS=-v`).
- Tests use `testcopilot`, never the dev `copilot` DB. `make test-db` copies `copilot` into it and points the API container at it. Re-run it after a schema change. When calling pytest directly:
  ```bash
  docker exec -e POSTGRES_DB=testcopilot -e TEST_POSTGRES_DB=testcopilot api pytest tests/chat
  ```
- Only PostgreSQL is real. `tests/conftest.py` fakes the rest for every test:
  - Chat model: request `mock_chat_model` to script replies or tool calls. Mock other model calls (`BedrockHandler.invoke_async`) yourself. A request that reaches Bedrock fails the test.
  - OpenSearch: an in-memory fake (`tests/fake_opensearch.py`). Its word-count scoring can't test search relevance.
  - Other boto3 calls: moto.

### Observing LLM requests
Set `LOG_FULL_INVOCATION_REQUEST_TO_FILE_PATH=/tmp/api/invocation.json` in `.env` to write each full Bedrock request (system blocks, messages, tools, token usage) to its own numbered file (`invocation_1.json`, ...). `/tmp/api` is bind-mounted from the host. To view one as HTML:
```bash
python scripts/system_prompt/pretty_print_invocation.py /tmp/api/invocation.json
```

## Architecture

There's one package per domain under `app/`. A module has whichever of these it needs: `routes.py`, `service.py`, `schemas.py`, `queries.py`, `exceptions.py` (`AppError` subclasses), `prompts.py`, `utils.py` (validation dependencies) and `constants.py` (fixed values). `app/projects` is the reference module.

- **Settings** go in `app/config.py` or the DB, never in a module. Module `config.py` files are legacy: move a setting out when you change it.
- **Services** return dicts and lists, not Pydantic models. Routes convert them to response schemas.
- **Imports** are explicit, from the defining module: `from app.auth.utils import verify_and_parse_uuid`.
- **Models**: there's one Bedrock model per use case, each an `llm_*` setting in `app/config.py` that an environment variable can override.

### Validation as dependencies
If the endpoint uses a dependency's result, take it as a parameter. If it only validates, put it in the decorator's `dependencies`. Write new validation in the module's `utils.py`.

```python
@router.post(
  path=ENDPOINTS.NEW_PATH,
  dependencies=[Depends(verify_auth_token), Depends(verify_and_get_auth_session_from_header)],  # validate only
)
async def new_endpoint(
  db_session: DbSession,
  user: User = Depends(verify_and_get_user_from_path_and_header),  # app/auth/verify_service.py
  message: Message = Depends(verify_and_get_message_from_path),  # app/new_module/utils.py
) -> ...:
```

### Database sessions and queries
Keep sessions short. Don't hold one open across an LLM call, a stream or any other slow call, because it ties up a pooled connection and any row locks. Commit, make the call, then open a new session for the writes that follow.
- **Routes** use the `DbSession` alias (`app/database/db_session.py`), never `Depends(get_db_session)`, which gives a second session. It commits when the endpoint returns, before the response or stream starts.
- **Queries** go in `queries.py` of the module that owns the use case, cross-domain ones included. `DbOperations` and `app/database/table.py` are legacy: move a method out when you change it.
- **Component functions** take `db_session`. They never commit or open a session, and use `flush()` for IDs.
- **Session functions** open their own `async with async_db_session()`, which commits or rolls back. They take no session, are a complete unit of work and return plain data or IDs (e.g. `acquire_compaction_lock`). Keep the two kinds visibly separate.
- **Group work by transaction.** Writes that succeed or fail together share a session, and related reads share one too.
- **Don't pass ORM objects between sessions.** Lazy-loading a relationship after the session closes raises under async. Pass IDs or plain data, or eager-load.
- **Models** stay in `app/database/models.py`, which Alembic and string-named relationships need.
- New queries don't need an `Action` entry or `LogsHandler.with_logging`.

### Free-text input and errors
- Clean free-text fields with the validators in `app/schemas/text.py`: `CleanSingleLine` for names and titles, `CleanMultiline` for prose. Put the validator after the limits: `Annotated[str, StringConstraints(max_length=255), CleanSingleLine]`. For text that can't be rejected (filenames), use `strip_to_single_line`.
- JSON bodies containing a NUL character get a 422 from `RejectNulBytesMiddleware`.
- Exception and log messages carry IDs, never user text.

### Exceptions that become HTTP responses
Subclass `AppError` (`app/exceptions/app_error.py`) in the module's `exceptions.py`. One handler covers every subclass, so don't add handlers.

```python
class ProjectNotFoundError(AppError):
    status_code = 404
    detail = "Project not found"  # body: {"detail": ...}
    expected = True  # info log, no Bugsnag. False (default): error log + Bugsnag
```

- Set `expected = True` for normal outcomes: not found, conflicts, bad headers.
- If the mapping depends on the instance, use properties (see `DatabaseError`). Override `content()` only to keep an existing body shape.
- Check Connect before changing an existing error's status code: it retries 401 and 403, then logs the user out.
- Exceptions that are always caught internally stay plain `Exception`.

### Prompts
Each module that calls an LLM keeps its prompt text in its own `prompts.py`, together with the code that assembles it. The reference is `app/chat/prompts.py`. It returns content blocks: static text first with `cache_control`, then an uncached dynamic block.

### Chat agent loop and tools
A chat response runs as an agent loop (`app/chat/agent_loop.py`): the model calls tools over several iterations, then answers. The model chooses the sources and writes its own queries; there's no routing step. The loop caps tool calls, times out slow tools and truncates long results.

The tools are central guidance (OpenSearch), uploaded documents, GOV.UK search and pages, Smart Targets benchmarks, audience segments and the style guide. Each one's `ToolDef` is in its module's `tools.py`. Executors and `TOOL_REGISTRY` are in `app/chat/tools.py`.
- Every enabled tool is sent on every request, sorted by name, so the tool list is byte-stable for caching.
- A deploy switch (`enabled`, e.g. `USE_RAG`) removes a tool. A chat option the user has turned off doesn't: `is_available` returns false and the model gets `unavailable_message` back.
- Tool results carry citations, which become the response's sources.

Long chats are compacted in the background (`app/compaction`): history is replaced with a summary, under a DB lock.

Style guide (`app/style_guide/`): rules are in `rule_mapping.json`, deterministic checks in `style_guide_checker.py`, and tests in `tests/style_guide/`.

## Skills

The project skills are in `.claude/skills/`: `alembic-migration` (use it for every migration), `api-test`, `local-postgres-db`, `bedrock-usage`, `write-intent-docs` and `run-evaluation` (user-invoked only).

## Style

Always type hints. In new code, use 3.12 types: str | None not Optional[str]; list[str] not List[str]. Google style (public) or one-line (private) docstrings. Avoid narrative comments - comment only to warn about non-obvious risks. Keep non-runtime code (scripts and evals) out of app/ and in scripts/.

## Pull Requests

Title: `Type(scope): summary`, e.g. `Fix(chat): stop duplicate citations`.

You cannot push or open PRs but you can edit the description when asked. Put everything you write in the description inside a collapsed block labelled with your model name. Leave the space above it for the human-written summary:
```
<details><summary><your model name></summary>

...

</details>
```

If the PR adds a migration, put 🗃️ after the type in the title (`Feat(chat): 🗃️ Remember chat options`). The description becomes the squash commit body, so end it with `Migration: <revision ID>`, one line per revision, after `</details>`, so git reads it as a trailer.

## Deployment

Dev, test and production run on Elastic Beanstalk. Deploy with `git push aws dev-production`, `test-production` or `production`. A preview environment runs on ECS, defined with CDK in `infra/` (see `infra/README.md`).
