#!/bin/sh
# Entrypoint baked into Dockerfile.ecs (the ECS preview image) - not used by
# the shared Dockerfile (local dev / EB), which is unmodified.
#
# 1. Verify the preview database exists. It's created once by a manual
#    bootstrap step (see infra/README.md), not by this container - the task's
#    Postgres role has no CREATEDB grant. Only creates it itself if
#    CREATE_DB_IF_MISSING=true is set, which the task definition doesn't set.
# 2. Apply Alembic migrations (idempotent - safe to run on every task start;
#    Alembic takes a lock so this is safe even if a redeploy briefly runs two
#    tasks).
# 3. Sync the central_guidance OpenSearch index from the DocumentChunk rows
#    Alembic just migrated - a fresh/reset OpenSearch domain has no indices,
#    and unlike local dev (docker-compose runs scripts/sync_opensearch/
#    sync_central_rag.py, which isn't in this image) nothing else creates them.
# 4. Start gunicorn with uvicorn workers, same flags as the real Procfile.
set -e

echo "Checking database ${POSTGRES_DB}..."
python - <<'EOF'
import os
import sys

import psycopg2

from app.config import settings

db = settings.postgres_db
conn = psycopg2.connect(
    host=settings.postgres_host,
    port=settings.postgres_port,
    user=settings.postgres_user,
    password=settings.postgres_password.get_secret_value(),
    dbname="postgres",
)
conn.autocommit = True
cur = conn.cursor()
cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db,))
exists = cur.fetchone()
if exists:
    print(f"Database '{db}' exists.")
elif os.environ.get("CREATE_DB_IF_MISSING", "").lower() == "true":
    cur.execute(f'CREATE DATABASE "{db}"')
    print(f"Database '{db}' created.")
else:
    print(f"ERROR: database '{db}' does not exist and CREATE_DB_IF_MISSING is not set.", file=sys.stderr)
    sys.exit(1)
EOF

echo "Running database migrations..."
cd /app/app/alembic && alembic upgrade head
cd /app

echo "Syncing central_guidance OpenSearch index..."
python - <<'EOF' || echo "WARNING: central_guidance sync failed - continuing startup without it" >&2
import asyncio

from app.central_guidance.service_index import sync_central_index
from app.database.db_session import async_db_session


async def sync():
    async with async_db_session() as db_session:
        await sync_central_index(db_session)


asyncio.run(sync())
EOF

echo "Starting gunicorn..."
exec gunicorn \
  -w "${GUNICORN_WORKERS:-3}" \
  -k uvicorn.workers.UvicornWorker \
  --max-requests 1000 \
  --max-requests-jitter 200 \
  --log-level info \
  --timeout 60 \
  --bind "0.0.0.0:${PORT:-8000}" \
  --log-config logging.ini \
  app.main:app
