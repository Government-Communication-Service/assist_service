DOCKER_COMPOSE_FILE = docker-compose.yml
DOCKER_COMPOSE_LOCAL_FILE = docker-compose.local.yml
POSTGRES_DB = copilot
TEST_POSTGRES_DB = testcopilot

container-name = api
docker-cmd = docker exec -it ${container-name} sh -c
venv-cmd = source .venv/bin/activate
test-cmd = pytest

logs:
	docker logs -f ${container-name}

deps:
	docker compose --file docker-compose.deps.yml up --detach

up:
	docker compose -f $(DOCKER_COMPOSE_FILE) up -d --remove-orphans

test-up:
	TEST_POSTGRES_DB=$(TEST_POSTGRES_DB) POSTGRES_DB=$(TEST_POSTGRES_DB) docker compose -f $(DOCKER_COMPOSE_FILE) up -d --remove-orphans

up-debug:
	DEBUG_MODE=True docker compose -f $(DOCKER_COMPOSE_FILE) up -d --remove-orphans

down:
	docker compose -f $(DOCKER_COMPOSE_FILE) down

build:
	docker compose -f $(DOCKER_COMPOSE_FILE) build --no-cache

start:
	$(MAKE) build
	$(MAKE) up

restart:
	$(MAKE) down
	$(MAKE) build
	$(MAKE) up

rebuild-db:
	$(MAKE) down
	$(MAKE) build
	$(MAKE) up
	$(MAKE) db-reset

stream:
	python test_stream.py

sync-central-rag:
	$(docker-cmd) "python scripts/sync_opensearch/sync_central_rag.py"

empty-central-rag:
	$(docker-cmd) "curl -X DELETE 'http://opensearch-node1:9200/central_guidance'"

test:
	$(MAKE) test-db
	$(MAKE) sync-central-rag
	$(docker-cmd) "${test-cmd} tests"

os-indexes:
	$(docker-cmd) "curl -X GET "http://host.docker.internal:9200/_cat/indices?v""

test-evals:
	$(docker-cmd) "${test-cmd} tests/evals"

test-stream:
	$(docker-cmd) "${test-cmd} tests/streaming/test_stream_v1.py::test_stream_response_valid_json"

test-system-prompt:
	$(docker-cmd) "${test-cmd} tests/unit/test_system_prompt.py"

test-table:
	$(MAKE) db
	$(docker-cmd) "${test-cmd} tests/unit/test_table.py"

test-prompts:
	$(docker-cmd) "${test-cmd} tests/prompts"

test-prompts-bulk-up:
	$(docker-cmd) "${test-cmd} tests/prompts/test_prompts_v1_e2e.py::TestThemes::test_bulk_upload_happy_path"

test-prompts-bulk-get:
	$(docker-cmd) "${test-cmd} tests/prompts/test_prompts_v1_e2e.py::TestThemes::test_get_prompts_bulk"

test-feedback:
	$(docker-cmd) "${test-cmd} tests/feedback"

test-opensearch:
	$(MAKE) sync-central-rag
	$(docker-cmd) "${test-cmd} tests/opensearch"

test-central-guidance:
	$(MAKE) sync-central-rag
	$(docker-cmd) "${test-cmd} tests/central_guidance"

test-central-guidance-rag:
	$(MAKE) sync-central-rag
	$(docker-cmd) "${test-cmd} tests/central_guidance/test_integration_central_guidance.py::test_search_central_guidance_with_mcom_query"

test-document-upload:
	$(docker-cmd) "${test-cmd} tests/document_upload tests/opensearch"

test-document-rag-integration:
	$(docker-cmd) "${test-cmd} tests/document_upload/rag/test_document_upload_rag_integration.py -v"

test-document-rag-unit:
	$(docker-cmd) "${test-cmd} tests/document_upload/rag/test_document_upload_rag_unit.py -v"

test-gov-uk-search:
	$(docker-cmd) "${test-cmd} tests/gov_uk_search"

test-gov-uk-search-regression:
	$(docker-cmd) "${test-cmd} tests/gov_uk_search/test_gov_uk_search_regression.py"

test-personal-prompts:
	$(docker-cmd) "${test-cmd} tests/personal_prompts"


test-bedrock:
	$(docker-cmd) "${test-cmd} tests/bedrock"

test-bedrock-cost-calculation:
	$(docker-cmd) "${test-cmd} tests/bedrock/test_bedrock.py::test_completion_cost_calculation"

test-chat:
	$(docker-cmd) "${test-cmd} tests/chat"

test-chat-cleanup:
	$(docker-cmd) "${test-cmd} tests/chat/test_message_cleanup.py"

test-user:
	$(docker-cmd) "${test-cmd} tests/user"

test-chat-create-message:
	$(docker-cmd) "${test-cmd} tests/chat/test_chat_create_message.py"

api-stop:
	docker stop ${container-name}

# Pauses the API service
api-pause:
	docker pause ${container-name}

# Unpauses the API service
api-unpause:
	docker unpause ${container-name}

# Restarts the API service
api-restart:
	docker restart ${container-name}

db-head:
	$(docker-cmd) "cd app/alembic && alembic upgrade head"

test-db-head:
	TEST_POSTGRES_DB=$(TEST_POSTGRES_DB) POSTGRES_DB=$(TEST_POSTGRES_DB) $(docker-cmd) "cd app/alembic && alembic upgrade head"

db-down:
	$(docker-cmd) "cd app/alembic && alembic downgrade -1"


db-history:
	$(docker-cmd) "cd app/alembic && alembic history"

db-reset:
	docker exec -it postgres psql -U postgres -c "DROP DATABASE IF EXISTS copilot;" && \
	docker exec -it postgres psql -U postgres -c "CREATE DATABASE copilot;"

test-db-reset:
	docker exec -it postgres psql -U postgres -c "DROP DATABASE IF EXISTS ${TEST_POSTGRES_DB};" && \
	docker exec -it postgres psql -U postgres -c "CREATE DATABASE ${TEST_POSTGRES_DB} TEMPLATE ${POSTGRES_DB};"

test-db-schema-reset:
	docker exec -i postgres pg_dump -U postgres --schema-only --format c -d $(POSTGRES_DB) > psql -U postgres -d $(TEST_POSTGRES_DB)

db: api-stop db-reset up

test-db: api-stop test-db-reset test-db-schema-reset test-up

migrate:
	@read -p "Enter migration message - what have you changed in the database? " user_message; \
	$(docker-cmd) "cd app/alembic && alembic revision --autogenerate -m '$$user_message'"

install:
	pip install -r requirements.txt

freeze:
	pip freeze > requirements.txt

lint:
	$(docker-cmd) "ruff check --fix --extend-ignore F403 --extend-ignore E402 --extend-ignore E712 --extend-ignore E711 && ruff format"

env:
	${venv-cmd}

local:
	cd app/alembic && alembic upgrade head && \
	cd ../ && \
	uvicorn app.main:app --host 0.0.0.0 --port 3520 --reload
