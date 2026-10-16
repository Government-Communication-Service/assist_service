"""Blind-judging helper: export anonymised packets for an LLM judge, then collect its verdicts.

Standard library only, so it can run from any directory.

    uv run judge.py export --run-id 2026-10-06-all_on --workspace ../../../copilot-evaluation
    # ...in the workspace, run the judge-eval-packets workflow (or /eval-judge by hand) and save its result...
    uv run judge.py collect --run-id 2026-10-06-all_on --workspace ../../../copilot-evaluation \
        --verdicts-json ../../../copilot-evaluation/verdicts.json

`export` writes one packet per (case, repeat, turn) with the two environments' responses as A and B in a
reproducible random order, plus a key (kept in the run directory, not the workspace) that maps A/B back to
environments. A fraction of items are exported a second time with A and B swapped to measure position bias.
`export` also writes `batches.json` (the workflow's args) and copies in the skill and the workflow.
`collect` reads verdict files and/or a workflow result and writes `judgments.jsonl` for report.py.
"""

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_RESULTS_DIR = HERE.parent.parent / "data" / "ignored" / "evaluation" / "results"
SKILL_SOURCE = HERE / "judge" / "SKILL.md"
WORKFLOW_SOURCE = HERE / "judge" / "judge-eval-packets.js"
WORKFLOW_DIR = Path(".claude") / "workflows"
KEY_FILE = "judge_key.json"
NO_RESPONSE = "[No response was produced]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["export", "collect"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--workspace", type=Path, required=True, help="Directory the judge sessions run in.")
    parser.add_argument(
        "--envs", nargs=2, metavar="ENV", help="The two environments to compare (default: the two in the run)."
    )
    parser.add_argument(
        "--swap-fraction", type=float, default=0.15, help="Share of items also exported with A/B swapped."
    )
    parser.add_argument("--batch-size", type=int, default=30, help="Packets per judge agent in batches.json.")
    parser.add_argument(
        "--verdicts-json",
        type=Path,
        nargs="+",
        help="collect: also read verdicts from judge workflow results (JSON files; later files override earlier ones).",
    )
    parser.add_argument("--force", action="store_true", help="Export even if the workspace already holds verdicts.")
    return parser.parse_args()


def stable_int(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()[:8], "big")


def load_env_records(path: Path) -> dict[tuple[str, int, int], dict]:
    """Records keyed by (case_id, repeat, turn_index), keeping only the latest chat of each (case, repeat)."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    latest_chat: dict[tuple[str, int], tuple[str, str | None]] = {}
    for record in sorted(records, key=lambda r: r["sent_at"]):
        latest_chat[(record["case_id"], record["repeat"])] = (record["sent_at"], record["chat_uuid"])
    return {
        (r["case_id"], r["repeat"], r["turn_index"]): r
        for r in records
        if r["chat_uuid"] == latest_chat[(r["case_id"], r["repeat"])][1]
    }


def response_of(record: dict | None) -> str:
    """The response text with the cache-busting nonce removed, or a placeholder if the turn produced none."""
    if not record or record["error"] or not record["response_text"].strip():
        return NO_RESPONSE
    return record["response_text"].replace(f"[ref {record['nonce']}]", "").replace(record["nonce"], "").strip()


def render_packet(
    item_id: str, scenario: str, history: list[dict], current: dict, response_a: str, response_b: str
) -> str:
    lines = [f"# Item {item_id}", "", f"Scenario: {scenario}", "", "<conversation_so_far>"]
    if not history:
        lines.append("(This is the first message of the conversation.)")
    for record in history:
        lines += [
            f"User, message {record['turn_index'] + 1}: {record['request']['query']}",
            f"Reference answer for that message: {record['expected_answer']}",
            "",
        ]
    lines += [
        "</conversation_so_far>",
        "",
        f'<user_message number="{current["turn_index"] + 1}">',
        current["request"]["query"],
        "</user_message>",
        "",
        "<reference_answer>",
        current["expected_answer"],
        "</reference_answer>",
        "",
        "<response_a>",
        response_a,
        "</response_a>",
        "",
        "<response_b>",
        response_b,
        "</response_b>",
        "",
        f"Write your verdict to verdicts/{item_id}.json",
        "",
    ]
    return "\n".join(lines)


def export(args: argparse.Namespace) -> None:
    run_dir = args.results_dir / args.run_id
    envs = args.envs or sorted(
        p.name.removesuffix(".jsonl") for p in run_dir.glob("*.jsonl") if p.name != "judgments.jsonl"
    )
    if len(envs) != 2:
        raise SystemExit(f"Need exactly two environments, found {envs}; pass --envs")
    records = {env: load_env_records(run_dir / f"{env}.jsonl") for env in envs}

    verdicts_dir = args.workspace / "verdicts"
    if verdicts_dir.exists() and any(verdicts_dir.iterdir()) and not args.force:
        raise SystemExit(f"{verdicts_dir} already holds verdicts; re-exporting would orphan them (use --force)")

    # (case, repeat) pairs an environment never attempted, as opposed to attempted and failed.
    attempted = {env: {key[:2] for key in records[env]} for env in envs}

    skipped, auto, items = [], [], []
    for key in sorted(set(records[envs[0]]) | set(records[envs[1]])):
        case_id, repeat, turn_index = key
        by_env = {env: records[env].get(key) for env in envs}
        reference = next(r for r in by_env.values() if r)
        base = {"case_id": case_id, "repeat": repeat, "turn_index": turn_index}
        not_run = [env for env in envs if key[:2] not in attempted[env]]
        if not_run:
            skipped.append({**base, "reason": f"never run on {', '.join(not_run)}"})
        elif "TODO" in reference["expected_answer"]:
            skipped.append({**base, "reason": "expected answer still marked TODO"})
        elif all(response_of(r) == NO_RESPONSE for r in by_env.values()):
            auto.append(base)
        else:
            swap = stable_int(args.run_id, "swap", *key) % 256 < args.swap_fraction * 256
            a_first = stable_int(args.run_id, "order", *key) % 2 == 0
            first, second = (envs[0], envs[1]) if a_first else (envs[1], envs[0])
            items.append({**base, "a_env": first, "b_env": second, "is_swap": False})
            if swap:
                items.append({**base, "a_env": second, "b_env": first, "is_swap": True})

    # Shuffle so the judge sees scenarios and repeat copies in no predictable order.
    random.Random(args.run_id).shuffle(items)
    width = max(3, len(str(len(items))))
    (args.workspace / "packets").mkdir(parents=True, exist_ok=True)
    verdicts_dir.mkdir(exist_ok=True)
    skill_dir = args.workspace / ".claude" / "skills" / "eval-judge"
    skill_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(SKILL_SOURCE, skill_dir / "SKILL.md")
    workflow_dir = args.workspace / WORKFLOW_DIR
    workflow_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(WORKFLOW_SOURCE, workflow_dir / WORKFLOW_SOURCE.name)

    key_items = {}
    for number, item in enumerate(items, start=1):
        item_id = str(number).zfill(width)
        key = (item["case_id"], item["repeat"], item["turn_index"])
        current = records[item["a_env"]].get(key) or records[item["b_env"]][key]
        history = [
            records[item["a_env"]].get((key[0], key[1], t)) or records[item["b_env"]][(key[0], key[1], t)]
            for t in range(item["turn_index"])
        ]
        packet = render_packet(
            item_id,
            current["scenario"],
            history,
            current,
            response_of(records[item["a_env"]].get(key)),
            response_of(records[item["b_env"]].get(key)),
        )
        (args.workspace / "packets" / f"{item_id}.md").write_text(packet)
        key_items[item_id] = item

    item_ids = list(key_items)
    batches = [{"ids": item_ids[i : i + args.batch_size]} for i in range(0, len(item_ids), args.batch_size)]
    (args.workspace / "batches.json").write_text(json.dumps({"batches": batches}))
    (run_dir / KEY_FILE).write_text(
        json.dumps(
            {"run_id": args.run_id, "envs": envs, "items": key_items, "auto": auto, "skipped": skipped}, indent=1
        )
    )
    n_swapped = sum(i["is_swap"] for i in items)
    print(f"Exported {len(items)} packets ({n_swapped} are swapped repeats) to {args.workspace}/packets")
    print(f"{len(auto)} turns had no response from either environment (judged automatically)")
    for reason, count in Counter(s["reason"] for s in skipped).items():
        print(f"{count} turns skipped: {reason}")
    print(f"Key written to {run_dir / KEY_FILE} (keep it out of the workspace)")
    print(f"Wrote {len(batches)} batches of up to {args.batch_size} to {args.workspace}/batches.json")
    print(
        f"Next: cd {args.workspace} && claude, then ask it to run the judge-eval-packets workflow "
        "with the contents of batches.json as args and to write every returned verdict to "
        "verdicts/<id>.json (the eval-judge skill explains how)"
    )


def validate_verdict(verdict: object) -> dict | str:
    """The verdict if it is well formed, or a string describing what is wrong with it."""
    if not isinstance(verdict, dict):
        return "not a JSON object"
    for field in ("a_acceptable", "b_acceptable"):
        if not isinstance(verdict.get(field), bool):
            return f"{field} must be true or false"
    if verdict.get("winner") not in ("A", "B", "tie"):
        return "winner must be A, B or tie"
    if not isinstance(verdict.get("reason"), str) or not verdict["reason"].strip():
        return "reason is missing"
    only_acceptable = {(True, False): "A", (False, True): "B"}.get((verdict["a_acceptable"], verdict["b_acceptable"]))
    if only_acceptable and verdict["winner"] != only_acceptable:
        return f"only {only_acceptable} is acceptable, so {only_acceptable} must win (check the acceptable flags)"
    return verdict


def read_verdict(path: Path) -> dict | str:
    """The validated verdict in a file, or a string describing what is wrong with it."""
    try:
        return validate_verdict(json.loads(path.read_text()))
    except json.JSONDecodeError as e:
        return f"invalid JSON ({e})"


def read_verdicts_json(path: Path) -> dict[str, object]:
    """Verdicts by item id from a workflow result: a list of verdicts, or {"verdicts": [...]}."""
    data = json.loads(path.read_text())
    entries = data["verdicts"] if isinstance(data, dict) else data
    return {str(entry.get("id")): entry for entry in entries if isinstance(entry, dict)}


def collect(args: argparse.Namespace) -> None:
    run_dir = args.results_dir / args.run_id
    key = json.loads((run_dir / KEY_FILE).read_text())
    rows, problems, missing = [], [], []
    verdicts: dict[str, dict] = {}
    from_json: dict[str, object] = {}
    for json_path in args.verdicts_json or []:
        from_json.update(read_verdicts_json(json_path))
    for item_id in key["items"]:
        path = args.workspace / "verdicts" / f"{item_id}.json"
        if path.exists():
            verdict = read_verdict(path)
        elif item_id in from_json:
            verdict = validate_verdict(from_json[item_id])
        else:
            missing.append(item_id)
            continue
        if isinstance(verdict, str):
            problems.append(f"{item_id}: {verdict}")
        else:
            verdicts[item_id] = verdict

    def winner_env(item: dict, verdict: dict) -> str:
        return {"A": item["a_env"], "B": item["b_env"], "tie": "tie"}[verdict["winner"]]

    for item_id, verdict in verdicts.items():
        item = key["items"][item_id]
        if item["is_swap"]:
            continue  # Swapped repeats only feed the consistency check.
        ids = {"case_id": item["case_id"], "repeat": item["repeat"], "turn_index": item["turn_index"], "item": item_id}
        for side, env in (("a", item["a_env"]), ("b", item["b_env"])):
            rows.append(
                {
                    "kind": "acceptable",
                    "env": env,
                    **ids,
                    "acceptable": verdict[f"{side}_acceptable"],
                    "reason": verdict["reason"],
                    "judge": verdict.get("judge"),
                }
            )
        rows.append(
            {
                "kind": "pairwise",
                **ids,
                "winner": winner_env(item, verdict),
                "reason": verdict["reason"],
                "judge": verdict.get("judge"),
            }
        )

    for ids in key["auto"]:
        for env in key["envs"]:
            rows.append(
                {"kind": "acceptable", "env": env, **ids, "acceptable": False, "reason": "no response", "judge": "auto"}
            )
        rows.append(
            {
                "kind": "pairwise",
                **ids,
                "winner": "tie",
                "reason": "no response from either environment",
                "judge": "auto",
            }
        )

    (run_dir / "judgments.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    # Summary
    print(f"{len(verdicts)} verdicts read; {len(missing)} packets without a verdict; {len(problems)} malformed")
    for problem in problems[:20]:
        print("  MALFORMED", problem)
    if missing:
        print(f"  missing: {', '.join(missing[:20])}{' ...' if len(missing) > 20 else ''}")
    retry_file = args.workspace / "batches-retry.json"
    retry_ids = sorted(set(missing) | {problem.split(":")[0] for problem in problems})
    if retry_ids:
        size = args.batch_size
        retry_batches = [{"ids": retry_ids[i : i + size]} for i in range(0, len(retry_ids), size)]
        retry_file.write_text(json.dumps({"batches": retry_batches}))
        print(f"  wrote {retry_file} for {len(retry_ids)} missing or malformed items; judge those, collect again")
    else:
        retry_file.unlink(missing_ok=True)
    main_rows = [r for r in rows if r["judge"] != "auto"]
    for env in key["envs"]:
        env_rows = [r for r in main_rows if r["kind"] == "acceptable" and r["env"] == env]
        if env_rows:
            print(f"{env}: acceptable {sum(r['acceptable'] for r in env_rows)}/{len(env_rows)}")
    wins = Counter(r["winner"] for r in main_rows if r["kind"] == "pairwise")
    print("pairwise winners:", dict(wins))
    positions = Counter(v["winner"] for v in verdicts.values())
    print(f"position check across all packets: A won {positions['A']}, B won {positions['B']}, tie {positions['tie']}")

    by_pair: dict[tuple, dict[bool, tuple[dict, dict]]] = {}
    for item_id, verdict in verdicts.items():
        item = key["items"][item_id]
        by_pair.setdefault((item["case_id"], item["repeat"], item["turn_index"]), {})[item["is_swap"]] = (item, verdict)
    pairs = [p for p in by_pair.values() if len(p) == 2]
    if pairs:
        same_winner = sum(winner_env(*p[False]) == winner_env(*p[True]) for p in pairs)
        print(f"swap consistency: same winner in {same_winner}/{len(pairs)} swapped pairs")


def main() -> None:
    args = parse_args()
    export(args) if args.command == "export" else collect(args)


if __name__ == "__main__":
    main()
