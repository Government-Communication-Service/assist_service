#!/usr/bin/env bash

# Opens an interactive psql session on a copilot-api database via a bastion
# host over SSH (same pattern as run-query.sh).
#
# Usage: ./db-connect.sh <bastion-host> <env> [postgres-user]

set -euo pipefail

declare -A DB_NAMES=(
        [dev]="dev-gcs-llm-copilot"
        [test]="test_gcs_llm_copilot"
        [prod]="prod_gcs_llm_copilot"
        [preview]="preview_gcs_llm_copilot"
)

usage() {
        echo "Usage: $0 <bastion-host> <env> [postgres-user]"
        echo "Valid env options: ${!DB_NAMES[*]}"
        exit 1
}

BASTION_HOST="${1:-}"
ENV="${2:-}"
POSTGRES_USER="${3:-gcs_copilot_api_user}"

if [[ -z "$BASTION_HOST" || -z "$ENV" ]]; then
        usage
fi

if [[ -z "${DB_NAMES[$ENV]+_}" ]]; then
        echo "Invalid environment option '${ENV}'"
        echo "Valid options: ${!DB_NAMES[*]}"
        exit 1
fi

DB="${DB_NAMES[$ENV]}"
RDS_HOST="gcs-llm-db.c3osqu0y68vr.eu-west-2.rds.amazonaws.com"

ssh -t "$BASTION_HOST" "psql -h $RDS_HOST -p 5432 -U $POSTGRES_USER -d $DB"
