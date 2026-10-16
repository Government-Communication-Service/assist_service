"""Run the evaluation cases against one environment and write one JSON line per turn.

Chats run concurrently (--concurrency) but turns within a chat run in order. The environment's
total LLM cost is read off a single time window, so nothing else should use it during a run.
Run once per environment (same cases, same users, same concurrency), then compare. See README.md.

    uv run run_eval.py --env-name agent_loop --env-file .env.agent_loop \
        --cases cases/*.json --flag-mode all_on --run-id 2026-10-06-all_on
"""

import argparse
import json
import logging
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cases import FLAGS, Case, load_cases
from client import EvalClient, EvalUser, TurnResult
from dotenv import dotenv_values

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
EVAL_DATA_DIR = REPO_ROOT / "data" / "ignored" / "evaluation"
DEFAULT_DOCUMENTS_DIR = EVAL_DATA_DIR / "documents"
DEFAULT_RESULTS_DIR = EVAL_DATA_DIR / "results"

logger = logging.getLogger("eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--env-name", required=True, help="Label written to every record, e.g. prod_clone or agent_loop."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=REPO_ROOT / ".env",
        help="Holds AUTH_SECRET_KEY and, optionally, BASE_URL and FRONTEND_URL.",
    )
    parser.add_argument("--base-url", help="Overrides BASE_URL from the env file.")
    parser.add_argument("--cases", nargs="+", type=Path, required=True, help="Case JSON files, run in the order given.")
    parser.add_argument("--documents-dir", type=Path, default=DEFAULT_DOCUMENTS_DIR)
    parser.add_argument("--users-file", type=Path, default=HERE / "users.json")
    parser.add_argument("--flag-mode", choices=["all_on", "needed_only", "always_rag"], required=True)
    parser.add_argument(
        "--run-id", required=True, help="Output goes to <out-dir>/<run-id>/. Re-using an id resumes that run."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--repeat-gap-s",
        type=float,
        default=0,
        help="Minimum time from the end of a case to the start of its next repeat (cache TTL is 5 minutes).",
    )
    parser.add_argument("--turn-delay-s", type=float, default=2, help="Pause before each follow-up turn.")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of chats in flight at once.")
    parser.add_argument(
        "--drain-s",
        type=float,
        default=30,
        help="Wait after the last turn so background LLM calls land inside the eval window.",
    )
    parser.add_argument("--timeout-s", type=float, default=300)
    parser.add_argument("--scenario", action="append", help="Only run these scenarios (repeatable).")
    parser.add_argument("--limit", type=int, help="Only run the first N cases (after the scenario filter).")
    return parser.parse_args()


def chat_flags(case: Case, flag_mode: str) -> dict[str, bool]:
    if flag_mode == "all_on":
        return dict.fromkeys(FLAGS, True)
    if flag_mode == "always_rag":
        return {flag: flag in case.needed_flags or flag == "use_rag" for flag in FLAGS}
    return {flag: flag in case.needed_flags for flag in FLAGS}


def load_users(path: Path) -> list[EvalUser]:
    return [
        EvalUser(uuid=entry["uuid"], details={k: v for k, v in entry.items() if k != "uuid"})
        for entry in json.loads(path.read_text())
    ]


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def completed_runs(records: list[dict], cases: list[Case]) -> set[tuple[str, int]]:
    """(case id, repeat) pairs with every turn recorded and no errors."""
    turns_needed = {case.id: len(case.turns) for case in cases}
    good: dict[tuple[str, int], set[int]] = {}
    for record in records:
        if record["error"] is None:
            good.setdefault((record["case_id"], record["repeat"]), set()).add(record["turn_index"])
    return {key for key, turn_indexes in good.items() if len(turn_indexes) == turns_needed.get(key[0], -1)}


def last_case_ends(records: list[dict]) -> dict[str, datetime]:
    """When each case last finished a request, so repeat pacing survives a resume."""
    ends: dict[str, datetime] = {}
    for record in records:
        sent_at = datetime.fromisoformat(record["sent_at"])
        end = sent_at + timedelta(seconds=record["total_time_s"] or 0)
        ends[record["case_id"]] = max(end, ends.get(record["case_id"], end))
    return ends


class Runner:
    def __init__(self, args: argparse.Namespace, client: EvalClient, users: list[EvalUser], out_file: Path):
        self.args = args
        self.client = client
        self.users = users
        self.out_file = out_file
        self.document_uuids: dict[tuple[str, str], str] = {}  # (user uuid, file name) -> uploaded document uuid
        self._upload_lock = threading.Lock()
        self._write_lock = threading.Lock()

    def _document_uuid(self, user: EvalUser, name: str) -> str:
        key = (user.uuid, name)
        with self._upload_lock:
            if key not in self.document_uuids:
                logger.info("Uploading %s for user %s", name, user.uuid)
                self.document_uuids[key] = self.client.upload_document(user, self.args.documents_dir / name)
            return self.document_uuids[key]

    def run_case(self, case: Case, user: EvalUser, repeat: int) -> None:
        flags = chat_flags(case, self.args.flag_mode)
        nonce = secrets.token_hex(3)
        chat_uuid: str | None = None
        for turn_index, turn in enumerate(case.turns):
            if turn_index > 0:
                time.sleep(self.args.turn_delay_s)
            # The nonce makes the conversation prefix unique to this run, so repeats can't read each
            # other's cache entries. The system prompt and tools still cache normally.
            query_sent = f"[ref {nonce}] {turn.query}" if turn_index == 0 else turn.query
            new_document_uuids = [self._document_uuid(user, name) for name in turn.add_documents]
            # The API reads flags from each request body, so resend the same flags every turn to keep them
            # constant for the chat.
            body = {"query": query_sent, **flags, "document_uuids": new_document_uuids or None}

            sent_at = datetime.now(timezone.utc)
            result = self.client.stream_chat(user, body, chat_uuid)
            chat_uuid = result.chat_uuid
            self._write(
                case, repeat, turn_index, turn, user, nonce, flags, query_sent, new_document_uuids, sent_at, result
            )
            logger.info(
                "%s repeat=%d turn=%d/%d ttft=%s total=%ss %s",
                case.id,
                repeat,
                turn_index + 1,
                len(case.turns),
                result.time_to_first_token_s,
                result.total_time_s,
                "ERROR" if result.error else "ok",
            )
            if result.error or chat_uuid is None:
                logger.warning("Abandoning %s repeat %d after a failed turn: %s", case.id, repeat, result.error)
                return

    def _write(
        self,
        case,
        repeat,
        turn_index,
        turn,
        user,
        nonce,
        flags,
        query_sent,
        new_document_uuids,
        sent_at,
        result: TurnResult,
    ) -> None:
        record = {
            "run_id": self.args.run_id,
            "env": self.args.env_name,
            "flag_mode": self.args.flag_mode,
            "scenario": case.scenario,
            "case_id": case.id,
            "repeat": repeat,
            "turn_index": turn_index,
            "n_turns": len(case.turns),
            "nonce": nonce,
            "user_uuid": user.uuid,
            "chat_uuid": result.chat_uuid,
            "assistant_message_uuid": result.assistant_message_uuid,
            "request": {
                "query": turn.query,
                "query_sent": query_sent,
                "flags": flags,
                "add_documents": list(turn.add_documents),
                "new_document_uuids": new_document_uuids,
            },
            "expected_answer": turn.expected_answer,
            "sent_at": sent_at.isoformat(),
            "time_to_first_token_s": result.time_to_first_token_s,
            "total_time_s": result.total_time_s,
            "http_status": result.http_status,
            "response_text": result.response_text,
            "citations": result.citations,
            "sources": result.sources,
            "statuses": result.statuses,
            "error": result.error,
        }
        with self._write_lock, self.out_file.open("a") as f:
            f.write(json.dumps(record) + "\n")


def write_meta(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    env = dotenv_values(args.env_file)
    base_url = args.base_url or env.get("BASE_URL") or "http://localhost:5312"
    auth_secret_key = env.get("AUTH_SECRET_KEY")
    if not auth_secret_key:
        raise SystemExit(f"AUTH_SECRET_KEY is not set in {args.env_file}")
    frontend_url = env.get("FRONTEND_URL") or "https://connect.communications.gov.uk"

    cases = load_cases(args.cases, args.documents_dir)
    # Index in the full list, so a case gets the same user in every environment whatever filters are used.
    user_index = {case.id: index for index, case in enumerate(cases)}
    if args.scenario:
        cases = [case for case in cases if case.scenario in args.scenario]
    if args.limit:
        cases = cases[: args.limit]
    users = load_users(args.users_file)
    unfinished = [case.id for case in cases if any("TODO" in turn.expected_answer for turn in case.turns)]
    if unfinished:
        logger.warning(
            "Expected answers still marked TODO (judge results will be unreliable): %s", ", ".join(unfinished)
        )

    run_dir = args.out_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_file = run_dir / f"{args.env_name}.jsonl"
    meta_file = run_dir / f"{args.env_name}.meta.json"

    records = read_records(out_file)
    done = completed_runs(records, cases)
    ends = last_case_ends(records)
    previous_meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    started_at = previous_meta.get("started_at") or datetime.now(timezone.utc).isoformat()
    meta = {
        "run_id": args.run_id,
        "env": args.env_name,
        "base_url": base_url,
        "flag_mode": args.flag_mode,
        "repeats": args.repeats,
        "n_cases": len(cases),
        "case_ids": [case.id for case in cases],
        "started_at": started_at,
        "finished_at": None,
    }
    write_meta(meta_file, meta)
    if done:
        logger.info("Resuming: %d (case, repeat) runs already complete", len(done))

    client = EvalClient(base_url, auth_secret_key, frontend_url, args.timeout_s)
    for user in users:
        client.start_session(user)
    runner = Runner(args, client, users, out_file)

    ends_lock = threading.Lock()

    def run_task(repeat: int, case: Case) -> None:
        with ends_lock:
            last_end = ends.get(case.id)
        if last_end is not None:
            wait_s = (last_end + timedelta(seconds=args.repeat_gap_s) - datetime.now(timezone.utc)).total_seconds()
            if wait_s > 0:
                logger.info("Waiting %.0fs before repeating %s", wait_s, case.id)
                time.sleep(wait_s)
        runner.run_case(case, users[user_index[case.id] % len(users)], repeat)
        with ends_lock:
            ends[case.id] = datetime.now(timezone.utc)

    tasks = [(repeat, case) for repeat in range(1, args.repeats + 1) for case in cases if (case.id, repeat) not in done]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_task, repeat, case) for repeat, case in tasks]
        for future, (repeat, case) in zip(futures, tasks, strict=False):
            try:
                future.result()
            except Exception:
                logger.exception("Case %s repeat %d crashed", case.id, repeat)

    logger.info("All cases done; waiting %.0fs for background LLM calls to finish", args.drain_s)
    time.sleep(args.drain_s)
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_meta(meta_file, meta)
    logger.info("Eval window: %s to %s", meta["started_at"], meta["finished_at"])


if __name__ == "__main__":
    main()
