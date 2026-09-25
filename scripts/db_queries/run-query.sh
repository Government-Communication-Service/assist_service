#!/usr/bin/env bash

# Runs a .sql file against the analytics read replica via a bastion host,
# streaming the results back to stdout as CSV.
#
# Usage: ./run-query.sh <bastion-host> <env> <sql-file> > output.csv

set -euo pipefail

declare -A DB_NAMES=(
        [dev]="dev-gcs-llm-copilot"
        [test]="test_gcs_llm_copilot"
        [prod]="prod_gcs_llm_copilot"
)

usage() {
        echo "Usage: $0 <bastion-host> <env> <sql-file> [> output.csv]"
        echo "Valid env options: ${!DB_NAMES[*]}"
        exit 1
}

BASTION_HOST="${1:-}"
ENV="${2:-}"
SQL_FILE="${3:-}"

if [[ -z "$BASTION_HOST" || -z "$ENV" || -z "$SQL_FILE" ]]; then
        usage
fi

if [[ -z "${DB_NAMES[$ENV]+_}" ]]; then
        echo "Invalid environment option '${ENV}'"
        echo "Valid options: ${!DB_NAMES[*]}"
        exit 1
fi

if [[ ! -f "$SQL_FILE" ]]; then
        echo "SQL file not found: $SQL_FILE"
        exit 1
fi

DB="${DB_NAMES[$ENV]}"

START_TIME=$(date +%s)

grep -v '^--' "$SQL_FILE" \
        | grep -v '^[[:space:]]*$' \
        | sed 's/;$//' \
        | tr '\n' ' ' \
        | sed 's/^/\\copy (/; s/$/) TO STDOUT WITH CSV HEADER;/' \
        | ssh "$BASTION_HOST" "psql -h gcs-llm-db-analytics-read-replica.c3osqu0y68vr.eu-west-2.rds.amazonaws.com -p 5432 -U postgres -d $DB -f -"

END_TIME=$(date +%s)
echo "Query took $((END_TIME - START_TIME))s" >&2
