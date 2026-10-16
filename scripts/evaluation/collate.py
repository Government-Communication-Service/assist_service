"""Collate one evaluation run: join runner output with token and cost data from each environment's database.

For each `<env>.jsonl` in the run directory this reads:
  - per-turn tokens and cost from `message`, joined on the assistant message uuid;
  - total LLM usage over the eval window (from `<env>.meta.json`) from `llm_internal_response`.

Writes `<run-dir>/collated/turns.json` and `<run-dir>/collated/windows.json`, which report.py reads.

Databases are reached over an SSH bastion with psql, like scripts/db_queries (see psql_client.py there).
Map each environment in the run to a database with `--db-env <env>=<dev|test|prod|preview>`:

    uv run collate.py --run-id 2026-10-06-all_on --bastion my-bastion \
        --db-env agent_loop=preview --db-env dev_main=test

Queries go to the analytics read replica, which can lag the primary slightly.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "db_queries"))

from psql_client import DB_NAMES, run_psql_json  # noqa: E402

DEFAULT_RESULTS_DIR = HERE.parent.parent / "data" / "ignored" / "evaluation" / "results"

# Per turn: the assistant row holds output tokens and the turn's answer cost; its parent user row holds
# input tokens (uncached) and cache read/write tokens. Same layout in both the old pipeline and the agent loop.
TURN_USAGE_SQL = """
select a.uuid::text as assistant_message_uuid,
       p.tokens as input_tokens,
       a.tokens as output_tokens,
       p.cache_read_tokens,
       p.cache_write_tokens,
       a.completion_cost::float8 as answer_cost
from message a
join message p on p.id = a.parent_message_id
where a.uuid::text = any(string_to_array(:'message_uuids', ','))
"""

# The window bounds are UTC; created_at is a naive timestamp in the database's own time zone.
WINDOW_SQL = """
select {purpose} as purpose,
       count(*) as calls,
       coalesce(sum(tokens_in), 0) as input_tokens,
       coalesce(sum(tokens_out), 0) as output_tokens,
       coalesce(sum(cache_read_tokens), 0) as cache_read_tokens,
       coalesce(sum(cache_write_tokens), 0) as cache_write_tokens,
       coalesce(sum(completion_cost), 0)::float8 as cost
from llm_internal_response
where created_at >= (:'started_at'::timestamptz at time zone current_setting('TimeZone'))
  and created_at <= (:'finished_at'::timestamptz at time zone current_setting('TimeZone'))
group by 1
order by 1
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--bastion", required=True, help="SSH host/alias for the bastion (as passed to `ssh`)")
    parser.add_argument(
        "--db-env",
        action="append",
        default=[],
        metavar="ENV=DB",
        help=f"Database for one environment of the run; DB is one of {', '.join(sorted(DB_NAMES))}. Repeatable.",
    )
    parser.add_argument("--postgres-user", default="postgres", help="Postgres user (default: postgres)")
    return parser.parse_args()


def database_name(env: str, mapping: dict[str, str]) -> str:
    key = mapping.get(env)
    if key is None:
        raise SystemExit(f"No database for '{env}': pass --db-env {env}=<{'|'.join(sorted(DB_NAMES))}>")
    if key not in DB_NAMES:
        raise SystemExit(f"Unknown database '{key}' for '{env}'; choose from {', '.join(sorted(DB_NAMES))}")
    return DB_NAMES[key]


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fetch_turn_usage(bastion: str, db: str, user: str, message_uuids: list[str]) -> dict[str, dict]:
    rows = run_psql_json(bastion, db, user, TURN_USAGE_SQL, {"message_uuids": ",".join(message_uuids)})
    return {row["assistant_message_uuid"]: row for row in rows}


def fetch_window_usage(bastion: str, db: str, user: str, started_at: str, finished_at: str) -> dict:
    has_purpose = bool(
        run_psql_json(
            bastion,
            db,
            user,
            "select 1 as present from information_schema.columns "
            "where table_name = 'llm_internal_response' and column_name = 'purpose'",
        )
    )
    # Environments without the purpose column predate the agent loop, so label by what is known.
    purpose_expr = "coalesce(purpose, 'unlabelled')" if has_purpose else "'unlabelled'"
    by_purpose = run_psql_json(
        bastion,
        db,
        user,
        WINDOW_SQL.format(purpose=purpose_expr),
        {"started_at": started_at, "finished_at": finished_at},
    )
    return {
        "has_purpose_column": has_purpose,
        "by_purpose": by_purpose,
        "started_at": started_at,
        "finished_at": finished_at,
    }


def collate_env(env: str, run_dir: Path, bastion: str, db: str, user: str) -> tuple[list[dict], dict]:
    records = read_records(run_dir / f"{env}.jsonl")
    meta = json.loads((run_dir / f"{env}.meta.json").read_text())
    if not meta.get("finished_at"):
        raise SystemExit(f"{env}: run has no finished_at yet; let it finish (or resume it) before collating")

    message_uuids = sorted({r["assistant_message_uuid"] for r in records if r["assistant_message_uuid"]})
    usage = fetch_turn_usage(bastion, db, user, message_uuids)
    window = fetch_window_usage(bastion, db, user, meta["started_at"], meta["finished_at"])

    turns = []
    for record in records:
        row = usage.get(record["assistant_message_uuid"]) or {}
        turns.append(
            {
                **{k: v for k, v in record.items() if k not in ("request", "statuses")},
                "query": record["request"]["query"],
                "flags": record["request"]["flags"],
                "n_tool_calls": sum(1 for s in record["statuses"] if s["state"] == "running"),
                "tools": [s["tool"] for s in record["statuses"] if s["state"] == "running"],
                "input_tokens": row.get("input_tokens"),
                "output_tokens": row.get("output_tokens"),
                "cache_read_tokens": row.get("cache_read_tokens"),
                "cache_write_tokens": row.get("cache_write_tokens"),
                "answer_cost": row.get("answer_cost"),
                "usage_found": bool(row),
            }
        )
    missing = sum(1 for t in turns if t["assistant_message_uuid"] and not t["usage_found"])
    if missing:
        print(f"WARNING {env}: {missing} turns have an assistant message uuid but no message row in the database")
    return turns, {**window, "flag_mode": meta["flag_mode"], "n_cases": meta["n_cases"], "repeats": meta["repeats"]}


def main() -> None:
    args = parse_args()
    mapping = dict(item.split("=", 1) for item in args.db_env)
    run_dir = args.results_dir / args.run_id
    envs = sorted(
        path.name.removesuffix(".jsonl") for path in run_dir.glob("*.jsonl") if path.name != "judgments.jsonl"
    )
    if not envs:
        raise SystemExit(f"No <env>.jsonl files in {run_dir}")

    all_turns: list[dict] = []
    windows: dict[str, dict] = {}
    for env in envs:
        turns, window = collate_env(env, run_dir, args.bastion, database_name(env, mapping), args.postgres_user)
        all_turns.extend(turns)
        windows[env] = window
        print(f"{env}: {len(turns)} turns, window {window['started_at']} to {window['finished_at']}")

    out_dir = run_dir / "collated"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "turns.json").write_text(json.dumps(all_turns))
    (out_dir / "windows.json").write_text(json.dumps(windows, indent=2))
    print(f"Wrote {out_dir} at {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
