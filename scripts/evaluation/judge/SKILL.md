---
name: eval-judge
description: Judge anonymised pairs of AI assistant responses against a reference answer. Use when asked to judge evaluation packets in this workspace (packets/*.md), optionally with a range such as "031-060".
---

# Judging evaluation packets

You are judging answers from an AI assistant used by UK government communications professionals. Each file in `packets/` is one item: a user message, a reference answer, and two anonymous responses, A and B. You decide whether each response is acceptable and which is better, and record a verdict for it. Verdicts end up as `verdicts/<item id>.json`; who writes the file depends on how you were started (see below).

## How you were started

- **Orchestrating the `judge-eval-packets` workflow** (the usual case): read `batches.json` and run the workflow with its full contents as `args`. Its agents return verdicts and write no files, so when it finishes, write each returned verdict to `verdicts/<id>.json` yourself, using exactly the fields in "Verdict file" below. Then list every id in `batches.json` that has no file in `verdicts/`, and rerun just those through the workflow in batches of 15. The workflow runs at most 16 agents at once, so a large `batches.json` runs in waves.
- **Judging a batch of packets** (as a workflow agent, told which ids to judge): return one verdict per id in your structured output and write no files.
- **Run by hand** (`/eval-judge`, optionally with a range): follow "What to do" and write the files yourself.

## What to do

1. The request may name a range of item ids (for example `031-060`). If it does not, take the first 30 items in `packets/` that have no file in `verdicts/`. Never redo an item that already has a verdict.
2. For each item, read the packet, judge it, and write the verdict file with the Write tool (run by hand only) before moving to the next item. Work through items one at a time and judge each on its own.
3. When the batch is done, say how many items you judged and how many packets still have no verdict. Stop there; do not continue past your batch.

Use only the packet. Do not read other files, search the workspace for context, or try to work out which system produced a response. Which response is A or B carries no information.

## Verdict file

```json
{"a_acceptable": true, "b_acceptable": false, "winner": "A", "reason": "One or two sentences.", "judge": "<your model id>"}
```

- `a_acceptable` and `b_acceptable`: `true` or `false`, judged independently of each other.
- `winner`: `"A"`, `"B"` or `"tie"`.
- `reason`: one or two sentences naming the deciding factor. For example, which required fact was missing or which instruction was broken.
- `judge`: your own model id, so the run is reproducible.

## Acceptable

A response is acceptable if a government communications professional could use it without being misled:

- It contains the key facts in the reference answer, allowing different wording, rounding, and reasonable formatting of numbers.
- It does not contradict the reference answer.
- It does not state specific facts (numbers, dates, names, quotes) that are fabricated or that conflict with the reference. Extra detail that you cannot verify and that does not conflict with the reference is not a reason to fail it.
- It answers the question asked, using the earlier messages for context. The conversation so far shows only the user's messages and reference answers, not the responses, so do not penalise a response for differing from the reference answer to an earlier message.

If the reference answer describes required behaviour instead of facts (for example "plain text with no formatting", "does not invent a quote", "no tool is needed"), the response must show that behaviour.

A response of "[No response was produced]" is not acceptable. Neither is a response that says it could not find or read the content, or asks the user to re-upload or paste it, when the reference shows the content was available.

### Length, structure and commentary

- You cannot count words reliably, so do not try. Judge length by eye. A stated limit ("no more than 200 words", "under 100 words") is a requirement, but fail a response for length only when it is clearly over, roughly a third or more beyond the limit. Slightly over is not a failure.
- A line or two of preamble or commentary on the process (for example "The document repeats itself, so...") is fine, provided the substantive answer follows, is clearly signposted, and is the right length. It counts against a response only when it is long enough to bury the answer or push the response clearly over a stated limit.
- Format instructions (a table, bullets, plain text) are requirements. Check them.

## Winner

Choose the response that would serve the user better, in this order:

1. Accuracy and fidelity to the reference answer.
2. Completeness for what was asked, with no padding.
3. Following the user's instructions on format and length, and plain, well-organised British English.

Say `tie` whenever the responses are about equally good; do not choose a winner on length, confidence or polish alone. If neither response is acceptable, pick the less bad one, or `tie` if you cannot separate them. If only one is acceptable, it wins: set `winner` to that response. A verdict whose acceptable flags and winner disagree is rejected and sent back, so check them against each other before you write it.
