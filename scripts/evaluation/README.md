# Evaluation runner

Runs scripted chats against one environment and records every turn as JSON. Run it once per
environment with the same cases and users, then compare. The tools are not tied to particular environments: call the
two sides whatever you like (`--env-name`). Below, **baseline** is the environment you compare against (for example
`prod`) and **candidate** is the one being evaluated (a new feature or branch). The examples use `agent_loop`
(candidate) and `dev_main` (baseline) from the agent-loop comparison; substitute your own names, env files and
databases.
Design: `thoughts/agent_loop/evaluation-spec.md` and `thoughts/agent_loop/evaluation-case-format.md`.

## Rerunning a comparison: the whole procedure

1. **Prepare the environments** (see "Before a run"): same deployed prompts and settings on both.
2. **Run** each flag mode on both environments at the same time, with a new run id per mode (see "Running").
3. **Judge**: archive the previous judge workspace, `export`, run the judge session, `collect` (see "Judging").
4. **Collate** the costs from the databases, then **render** the report (see "Collating and reporting").
5. Optionally build the **deck** and the **cases document** (see the last section).

Run ids are `<date>-<flag mode>`, for example `2026-10-08-always_rag`. Reusing an id resumes that run, so take a new
id for every fresh run. To add cases later without rerunning everything, run them under their own id
(for example `2026-10-08-always_rag-new`) rather than resuming: a resume keeps the original `started_at` and moves
`finished_at`, so the cost window would cover the gap and any other traffic in it.

## Setup

```bash
cd scripts/evaluation
uv sync
```

One env file per environment (git-ignored via `.env.*`), e.g. `.env.candidate` and `.env.baseline`:

```
BASE_URL=https://<api host>
AUTH_SECRET_KEY=<AUTH_SECRET_KEY of that environment>
FRONTEND_URL=https://<frontend host>   # optional, sent as Origin/Referer to pass the WAF
```

Documents come from `data/ignored/evaluation/documents/`. Users come from `users.json` (fixed UUIDs,
created on first use and reused on every environment).

## Before a run

- Both environments must differ only in the feature under test: deploy the candidate build, and make sure
  everything else (model configuration, `compaction_token_threshold`, cache settings, prompt changes you want on both
  sides) matches. The runner cannot check this.
- Nothing else should use the two environments during a run: total cost is read off a single time window.
- Cases are the JSON files in `cases/`. A case with a non-empty `"skip"` reason is left out (the runner prints it).
  Cases with a `TODO` or empty expected answer still run but are skipped at judging. Fill in or fix expected answers
  before the run, not after.
- Read the cases and expected answers again for the new comparison. They were written during the agent-loop work, so
  a few encode its behaviour or the data of the day (the Smart Targets figures come from the dev GCS Data API and can
  change; some answers say what a response must not do, such as invent a quote). Cases that both sides always pass can
  be skipped with `"skip"` to save time.
- The test files the cases upload must be in `data/ignored/evaluation/documents/`.

## Running

Run from `scripts/evaluation`. Start both environments at the same time, one terminal each:

```bash
uv run run_eval.py --env-name agent_loop --env-file .env.agent_loop \
    --cases cases/*.json --flag-mode always_rag --run-id 2026-10-08-always_rag --repeats 6
uv run run_eval.py --env-name dev_main --env-file .env.dev-main \
    --cases cases/*.json --flag-mode always_rag --run-id 2026-10-08-always_rag --repeats 6
```

(`cases/smoke/smoke.json` is a placeholder smoke test, deliberately outside the `cases/*.json` glob.) The usual
set is three flag modes, each its own run id and each run on both environments: `all_on`, `needed_only` and
`always_rag`. Check afterwards that both environments have the same number of records
(`wc -l data/ignored/evaluation/results/<run-id>/*.jsonl`) before judging; a partial run gives partial packets.

Chats within an environment run concurrently (`--concurrency`, default 4); use the same value for both.
Latency includes any contention from concurrent chats, so for clean latency numbers also do a
`--concurrency 1` run.

Flags:

- `--flag-mode`: `all_on` sends all chat flags on. `needed_only` sends each case's `needed_flags`. `always_rag` sends
  the `needed_flags` plus `use_rag`, simulating the app default of RAG always ticked. If one side retrieves on every
  turn and the other decides for itself (the old pipeline against the agent loop), `all_on` flatters the one that
  decides, so report `needed_only` too as the conservative end. Between two builds that behave alike, one mode is
  enough, but pick the one that matches how users actually use the feature.
- `--repeats N` runs every case N times. `--repeat-gap-s` (default 0) is the minimum time from the end of a case to
  the start of its next repeat; raise it (the cache TTL is 5 minutes) to measure cold-cache cost.
- `--turn-delay-s` (default 2) pauses before each follow-up. `--drain-s` (default 30) waits at the end for background
  LLM calls to land inside the window.
- `--scenario`, `--limit` select a subset. Re-running with the same `--run-id` resumes, skipping completed (case,
  repeat) runs.

To stop a run, Ctrl+C (twice if needed) or `pkill -f run_eval.py`. Delete its results directory to start clean.

## Output

`data/ignored/evaluation/results/<run-id>/<env-name>.jsonl` has one record per turn (fields listed in the case-format doc).
`data/ignored/evaluation/results/<run-id>/<env-name>.meta.json` has the eval window (`started_at`, `finished_at`); sum
`llm_internal_response` over that window for total cost, which assumes nothing else is using the env.

- A failed turn abandons the rest of its chat. Resuming re-runs that case in a new chat, so a failed attempt's records stay in the file: when analysing, take the latest chat per (`case_id`, `repeat`). The abandoned chat's LLM cost stays in the window total.
- Per-turn tokens and cost come from `message` via `assistant_message_uuid` and `chat_uuid`.
- The `[ref xxxxxx]` nonce on the first user message of each chat is recorded as `nonce`.

## Collating and reporting

After both environments have finished, join the results with token and cost data from each environment's
database, then open the report. `collate.py` reaches the databases over an SSH bastion with `psql`, the same
way as `scripts/db_queries` (it imports `psql_client.py` from there), so you need `ssh` access to the bastion and
`psql` on it. Map each environment in the run to a database (`dev`, `test`, `prod` or `preview`):

```bash
uv run collate.py --run-id 2026-10-08-always_rag --bastion my-bastion \
    --db-env agent_loop=preview --db-env dev_main=test      # <env name>=<database>, e.g. prod=prod
uv run marimo edit report.py     # or: uv run marimo run report.py
```

Queries go to the analytics read replica, so if you collate immediately after a run, allow for replication lag
(a few minutes to be safe). `--postgres-user` defaults to `postgres`. Both environments need `finished_at` in their
`.meta.json`, which a completed run writes.

To render a report as a file (do this after `judge.py collect`, so the quality section is filled in), name the run
and the baseline explicitly; otherwise the report opens on the alphabetically last run:

```bash
EVAL_RUN=2026-10-08-always_rag EVAL_BASELINE=<baseline env name> \
    uv run marimo export html report.py -o ../../data/ignored/evaluation/report-2026-10-08-always_rag.html
EVAL_RUN=2026-10-08-always_rag EVAL_BASELINE=<baseline env name> uv run marimo run report.py --port 2718   # interactive
```

The exported HTML is static (its filters are frozen); use `marimo run` or `marimo edit` for the interactive version.

Sanity checks after collating: `usage_found` is true for every turn in `collated/turns.json`; for an environment with a
`purpose` column the `chat_loop` window total equals the summed answer cost; failed turns are zero or understood. An
environment without a `purpose` column (the old pipeline) shows all its background calls as "unlabelled", and some of
them (GOV.UK relevance, Smart Targets filter selection) are not metered, so its total cost is a floor. Two builds that
both have the column are costed on equal terms.

`collate.py` writes `collated/turns.json` and `collated/windows.json` into the run directory. The report shows
whole-run cost (answer cost from `message` plus non-`chat_loop` rows in `llm_internal_response` over the eval
window), then per-turn cost, tokens, cache share, latency and tool calls by scenario, cost scaling for
`long_chat`, a baseline-versus-candidate paired comparison, quality (if `judgments.jsonl` exists, format in the
report) and a browser for individual responses against the expected answers.

## Judging

An LLM judge (Claude) decides, for each turn, whether each environment's response is acceptable against the
expected answer, and which response is better. The judge never sees environment names: `judge.py` exports
anonymised packets and, afterwards, maps the verdicts back.

### Before you start

- Both environments' `<env>.jsonl` files must be complete and in the run directory (`export` needs exactly two; pass
  `--envs A B` if the directory holds more). `collate.py` is not required.
- Cases whose expected answer still contains `TODO` or is empty are **skipped**: they get no packet and no verdict.
- Turns where neither environment produced a response are judged automatically (both unacceptable, tie).
- Use a workspace outside the repo (`../../../copilot-evaluation`, a sibling of the repo), so the judges inherit none
  of the repo's context or results.
- **The workspace holds one run at a time, and `export` does not clear old packets.** Before exporting a new run, move
  the previous `packets/`, `batches.json` and `verdicts/` into `archive/<previous run id>/` inside the workspace
  (never delete judging output). `export` refuses to run if `verdicts/` still has files, unless you pass `--force`.

### Steps

Run `judge.py` from `scripts/evaluation`:

```bash
uv run judge.py export --run-id 2026-10-08-always_rag --workspace ../../../copilot-evaluation --batch-size 15
(cd ../../../copilot-evaluation && claude)
```

About 500 packets in batches of 15 is around 34 agents; the workflow runs at most 16 at a time. Packets include a
share repeated with A and B swapped (`--swap-fraction`, default 0.15) to measure position bias.

In that Claude Code session, paste this prompt:

> Read `batches.json` and run the `judge-eval-packets` workflow with its full contents as `args`. Every verdict must
> end up as `verdicts/<id>.json`: the workflow returns verdicts rather than writing them, so write one file per
> returned verdict yourself, with exactly the fields in `.claude/skills/eval-judge/SKILL.md`. Don't read anything
> outside `packets/` and the skill. When it finishes, list any ids that have no file in `verdicts/`. If any are
> missing, rerun only those ids in batches of 15.

Approve the first launch of the saved workflow when asked. The judge sees only the packets and the skill, never the
repo. When the session says it is done, check that `verdicts/` holds one file per packet (`ls verdicts | wc -l` against
the packet count `export` printed). Then, back in `scripts/evaluation`:

```bash
uv run judge.py collect --run-id 2026-10-08-always_rag --workspace ../../../copilot-evaluation
```

`collect` reads every `verdicts/<id>.json`, validates them, writes `judgments.jsonl` into the run directory and prints
acceptable rates, pairwise winners, and a position-bias check (how often A won, and how often the same environment won
when an item was judged twice with A and B swapped). Look for "0 packets without a verdict; 0 malformed".

Validation rejects a malformed verdict, and one that contradicts itself (for example only B acceptable but A the
winner). If any are missing or rejected, `collect` also writes `batches-retry.json` in the workspace. Ask the session
to run the workflow with that file as args and write the verdict files, then collect again. `--verdicts-json` is only
needed if a workflow result was saved as a single JSON file instead of per-item files; later files override earlier ones.

### What is in the workspace

`packets/` (one markdown file per item), `batches.json`, `verdicts/`, `archive/` (earlier runs), the `eval-judge` skill
(`.claude/skills/eval-judge/SKILL.md`) and the workflow (`.claude/workflows/judge-eval-packets.js`); the sources are in
`judge/`. The A/B key (`judge_key.json`) stays in the run directory, outside the workspace, so the judges can't see it.

The judging rubric (what counts as acceptable, length, commentary, non-answers, how to pick a winner) is in
`judge/SKILL.md`. Edit it there; `export` copies it into the workspace each time. Expected answers are part of the
rubric in practice: the judge treats them as the reference, so write them to say what a good answer must and must not
contain.

### Options and limits

- `--batch-size` (default 30) is the number of packets per judge agent. Smaller batches (15) let more agents run in
  parallel and keep each agent's context short.
- `--swap-fraction` (default 0.15) is the share of items also exported with A and B swapped.
- Judging by hand is also possible: run `/eval-judge` (or `/eval-judge 031-060`) in the workspace; it writes
  `verdicts/<id>.json` files, which `collect` reads.
- The judgments format the report reads is documented at the top of its Quality section.

## Smoke test

```bash
uv run run_eval.py --env-name local --cases cases/smoke/smoke.json --flag-mode needed_only \
    --run-id smoke --out-dir /tmp/eval --drain-s 3 --turn-delay-s 2
```

Uses `.env` in the repo root and `http://localhost:5312`. The expected answers in `cases/smoke/smoke.json` are placeholders.

## Summary deck and cases document

Both scripts declare their own dependencies, so `uv run` needs no changes to `pyproject.toml`.

```bash
# Slide deck for a judged and collated run (writes evaluation-summary-<run-id>.pptx next to the HTML reports)
uv run build_deck.py --run-id 2026-10-08-always_rag --candidate new_feature --baseline prod [--examples "govuk-003:4:2;long_chat-001:2:11"]

# Word document listing every case, question and expected answer (replaces data/ignored/evaluation/evaluation-cases.docx)
uv run build_cases_doc.py
```

`build_deck.py` compares `--candidate` (default `agent_loop`) with `--baseline` (default `dev_main`); give your own
environment names and, if you like, `--candidate-label` and `--baseline-label` for the slides. The unlabelled-cost note
appears only for an environment without a `purpose` column. It reads the verdict files from
`../../../copilot-evaluation/verdicts` (`--workspace` changes that). `--examples` picks the individual turns shown in the appendix as `case_id:repeat:turn_index`. `build_cases_doc.py` keeps the previous document's styles and footer and saves it first as `evaluation-cases-<date>.docx`.
