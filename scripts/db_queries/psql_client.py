"""Shared helper for running read-only queries against a copilot-api Postgres database over an
SSH bastion (same connection pattern as scripts/db_queries/db-connect.sh), used by the various
export_*.py scripts in this directory.
"""

import html
import json
import shlex
import subprocess
import sys

DB_NAMES = {
    "dev": "dev-gcs-llm-copilot",
    "test": "test_gcs_llm_copilot",
    "prod": "prod_gcs_llm_copilot",
    "preview": "preview_gcs_llm_copilot",
}
RDS_HOST = "gcs-llm-db-analytics-read-replica.c3osqu0y68vr.eu-west-2.rds.amazonaws.com"


def run_psql_json(
    bastion_host: str, db: str, user: str, sql: str, variables: dict[str, str] | None = None
) -> list[dict]:
    """Run a SELECT over SSH+psql and return its rows as a list of dicts, via json_agg.

    Values passed in `variables` are injected as psql variables and must be referenced in `sql`
    as `:'name'` so psql applies SQL-literal quoting (safe against injection). Note: psql only
    performs `:'name'` substitution on script input (stdin/-f), not on `-c` command strings, so
    the query is piped to psql's stdin via `-f -` rather than passed as `-c`.
    """
    wrapped_sql = f"SELECT COALESCE(json_agg(t), '[]'::json)::text FROM ({sql.strip().rstrip(';')}) t;"
    psql_args = [
        "psql",
        "-h",
        RDS_HOST,
        "-p",
        "5432",
        "-U",
        user,
        "-d",
        db,
        "-X",
        "-q",
        "-t",
        "-A",
        "-v",
        "ON_ERROR_STOP=1",
    ]
    for name, value in (variables or {}).items():
        psql_args += ["-v", f"{name}={value}"]
    psql_args += ["-f", "-"]

    # ssh joins its trailing argv with spaces and hands the result to the remote shell, so we
    # pre-quote each token ourselves rather than relying on subprocess's (local-only) argv handling.
    remote_command = " ".join(shlex.quote(arg) for arg in psql_args)

    result = subprocess.run(["ssh", bastion_host, remote_command], input=wrapped_sql, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"psql query failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    output = result.stdout.strip()
    return json.loads(output) if output else []


def escape(text) -> str:
    if text is None:
        return "(null)"
    return html.escape(str(text))


def format_number(n) -> str:
    """Render an integer with thousands separators, e.g. 12345 -> '12,345'."""
    if n is None:
        return "(null)"
    return f"{n:,}"


def iso_ts(pg_timestamp_text: str | None) -> str:
    """Convert a Postgres `timestamp::text` value ('YYYY-MM-DD HH:MI:SS[.ffffff]') to the
    'T'-separated ISO form used by <input type="datetime-local"> so string comparisons in JS
    filters line up correctly.
    """
    if not pg_timestamp_text:
        return ""
    return pg_timestamp_text.replace(" ", "T", 1)
