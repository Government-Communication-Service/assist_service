#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

AUTH_TOKEN=$(grep '^AUTH_SECRET_KEY=' .env | head -1 | cut -d= -f2 | cut -d'#' -f1 | xargs)
USER_UUID=$(docker exec postgres psql -U postgres -d copilot --csv -t -c \
  "SELECT uuid FROM \"user\" WHERE deleted_at IS NULL LIMIT 1;" | xargs)
SESSION_UUID=$(docker exec postgres psql -U postgres -d copilot --csv -t -c \
  "SELECT uuid FROM auth_session WHERE user_id = (SELECT id FROM \"user\" WHERE uuid = '$USER_UUID') AND deleted_at IS NULL ORDER BY created_at DESC LIMIT 1;" | xargs)

curl -s -X POST "http://localhost:5312/v1/chats/users/$USER_UUID" \
  -H "Auth-Token: $AUTH_TOKEN" \
  -H "Session-Auth: $SESSION_UUID" \
  -H "User-Key-UUID: $USER_UUID" \
  -H "Content-Type: application/json" \
  -d '{"query": "Write me a haiku about sausages", "use_rag": false}' \
  | jq -r '.message.content'
