"""Check text against the style guide rules: patterns over the whole text, then parallel model calls.

Each rule's check (rule_mapping.json) says what the model does for it:

- none: every pattern match is a finding; no model call.
- matches: the model judges each pattern match in its context, in batches of matches.
- chunks: the model reads paragraph-aligned chunks of the text for breaches.
- document: the model judges the title, summary and headings once.

All model calls for a check run at once, under a concurrency limit, a per-call timeout and a deadline
for the whole check. Calls that fail or don't finish are reported as rules not checked, and the
findings from the rest are kept.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import IntEnum

from pydantic import BaseModel
from sqlalchemy import select

from app.bedrock import BedrockHandler, RunMode
from app.bedrock.call_log import LlmCallContext
from app.config import STYLE_GUIDE_LLM_MODEL, ThinkingLevel, settings
from app.database.models import LLM
from app.database.table import async_db_session
from app.style_guide.exceptions import StyleGuideCheckError
from app.style_guide.matching import Match, find_matches, locate_quote, split_into_chunks
from app.style_guide.prompts import (
    MATCH_VERDICTS_SCHEMA,
    SYSTEM_PROMPT,
    chunk_prompt,
    document_prompt,
    findings_schema,
    matches_prompt,
)
from app.style_guide.rules import RULES, LlmKind, Rule

logger = logging.getLogger(__name__)

# Characters of the text's start given to the document rules when the whole text is too long for them.
DOCUMENT_DIGEST_HEAD_CHARS = 4000
DOCUMENT_DIGEST_MAX_LINES = 60
_HEADING_LIKE = re.compile(r"^[ \t]*(\S[^\n]{0,118}?)[ \t]*$", re.MULTILINE)


class Source(IntEnum):
    """How a finding was found, in reporting order."""

    PATTERN = 0
    CONFIRMED = 1
    MODEL = 2


@dataclass(frozen=True)
class Finding:
    """One place the text breaks a rule.

    Attributes:
        rule_id: The rule broken.
        start: Offset of the text in the checked content.
        end: Offset just after it.
        text: The text that breaks the rule.
        source: How it was found.
    """

    rule_id: str
    start: int
    end: int
    text: str
    source: Source


@dataclass
class CheckResult:
    """What a check found, and what it couldn't check.

    Attributes:
        findings: Findings in text order, at most one per rule and start offset.
        not_checked: Titles of rules whose model calls failed or didn't finish.
        not_reviewed: Pattern matches not sent to the model because of the per-rule limit, by rule id.
        calls: Model calls made.
        failed_calls: Model calls that failed or didn't finish.
    """

    findings: list[Finding] = field(default_factory=list)
    not_checked: list[str] = field(default_factory=list)
    not_reviewed: dict[str, int] = field(default_factory=dict)
    calls: int = 0
    failed_calls: int = 0


@dataclass(frozen=True)
class _Job:
    """One model call: its prompt, schema, rules and how to read its answer."""

    kind: LlmKind
    prompt: str
    schema: dict
    rules: list[Rule]
    items: list[Match] = field(default_factory=list)
    scope_offset: int = 0
    scope_text: str = ""


class _MatchVerdict(BaseModel):
    item: int
    reason: str = ""
    breaks_rule: bool = True


class _MatchVerdicts(BaseModel):
    violations: list[_MatchVerdict] = []


class _QuotedFinding(BaseModel):
    rule_id: str
    quote: str
    reason: str = ""
    breaks_rule: bool = True


class _QuotedFindings(BaseModel):
    violations: list[_QuotedFinding] = []


async def get_llm(model: str) -> LLM:
    """Look up the model's LLM row in a short session of its own.

    Raises:
        StyleGuideCheckError: The model isn't in the LLM table.
    """
    async with async_db_session() as db_session:
        llm = await db_session.scalar(select(LLM).where(LLM.model == model))
    if llm is None:
        raise StyleGuideCheckError(f"Style guide model {model} is not in the LLM table")
    return llm


def _select_for_review(matches: list[Match], rule: Rule, limit: int) -> tuple[list[Match], int]:
    """The matches of one rule to send to the model, and how many were left out by the limit."""
    if rule.first_only:
        seen: set[str] = set()
        firsts = []
        for match in matches:
            key = match.text.casefold().rstrip("s")
            if key not in seen:
                seen.add(key)
                firsts.append(match)
        matches = firsts
    return matches[:limit], max(len(matches) - limit, 0)


def _document_digest(text: str, max_chars: int) -> tuple[str, bool]:
    """The text for the document rules: all of it if short, else its start and its heading-like lines."""
    if len(text) <= max_chars:
        return text, True
    head = text[:DOCUMENT_DIGEST_HEAD_CHARS]
    lines = []
    for m in _HEADING_LIKE.finditer(text, len(head)):
        line = m.group(1)
        if line[-1] not in ".,;:" and len(line.split()) <= 15:
            lines.append(line)
        if len(lines) >= DOCUMENT_DIGEST_MAX_LINES:
            break
    return head + "\n[...]\n" + "\n".join(lines), False


def build_jobs(
    text: str, rules: list[Rule], matches: dict[str, list[Match]], checker_context: str | None
) -> tuple[list[_Job], dict[str, int]]:
    """Plan the model calls for a check.

    Returns:
        The jobs, and the number of matches per rule left out by the per-rule limit.
    """
    jobs: list[_Job] = []
    not_reviewed: dict[str, int] = {}

    items: list[Match] = []
    for rule in rules:
        if rule.llm != LlmKind.MATCHES or rule.rule_id not in matches:
            continue
        selected, left_out = _select_for_review(matches[rule.rule_id], rule, settings.style_guide_max_matches_per_rule)
        items.extend(selected)
        if left_out:
            not_reviewed[rule.rule_id] = left_out
    by_id = {rule.rule_id: rule for rule in rules}
    batch_size = settings.style_guide_match_batch_size
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        batch_rules = [by_id[rule_id] for rule_id in dict.fromkeys(m.rule_id for m in batch)]
        jobs.append(
            _Job(
                LlmKind.MATCHES,
                matches_prompt(text, batch_rules, batch, checker_context),
                MATCH_VERDICTS_SCHEMA,
                batch_rules,
                items=batch,
            )
        )

    chunk_rules = [rule for rule in rules if rule.llm == LlmKind.CHUNKS]
    if chunk_rules:
        chunks = split_into_chunks(text, settings.style_guide_chunk_chars)
        schema = findings_schema([rule.rule_id for rule in chunk_rules])
        for part, (offset, chunk) in enumerate(chunks, 1):
            prompt = chunk_prompt(chunk, part, len(chunks), chunk_rules, checker_context)
            jobs.append(_Job(LlmKind.CHUNKS, prompt, schema, chunk_rules, scope_offset=offset, scope_text=chunk))

    document_rules = [rule for rule in rules if rule.llm == LlmKind.DOCUMENT]
    if document_rules:
        digest, is_whole = _document_digest(text, settings.style_guide_chunk_chars)
        jobs.append(
            _Job(
                LlmKind.DOCUMENT,
                document_prompt(digest, is_whole, document_rules, checker_context),
                findings_schema([rule.rule_id for rule in document_rules]),
                document_rules,
                scope_text=text,
            )
        )
    return jobs, not_reviewed


def _request_kwargs(schema: dict, thinking: ThinkingLevel) -> dict:
    output_config: dict = {"format": {"type": "json_schema", "schema": schema}}
    if thinking == ThinkingLevel.disabled:
        return {"thinking": {"type": "disabled"}, "output_config": output_config}
    output_config["effort"] = thinking.value
    return {"thinking": {"type": "adaptive"}, "output_config": output_config}


async def _call(job: _Job, handler: BedrockHandler, context: LlmCallContext, thinking: ThinkingLevel) -> dict:
    """Make one model call and return its parsed JSON answer.

    Raises:
        StyleGuideCheckError: The answer was cut off, refused or not JSON.
    """
    response = await handler.invoke_async(
        messages=[{"role": "user", "content": job.prompt}],
        context=context,
        system=SYSTEM_PROMPT,
        **_request_kwargs(job.schema, thinking),
    )
    if response.stop_reason in ("max_tokens", "refusal"):
        raise StyleGuideCheckError(f"Style guide {job.kind.value} call stopped: {response.stop_reason}")
    text = next((block.text for block in response.content if block.type == "text"), None)
    if text is None:
        raise StyleGuideCheckError(f"Style guide {job.kind.value} call returned no text")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise StyleGuideCheckError(f"Style guide {job.kind.value} call returned invalid JSON") from e


def _read_answer(job: _Job, answer: dict) -> list[Finding]:
    """Turn a call's answer into findings, dropping any that don't point at real text."""
    findings = []
    if job.kind == LlmKind.MATCHES:
        for verdict in _MatchVerdicts.model_validate(answer).violations:
            if not verdict.breaks_rule:
                continue
            if not 1 <= verdict.item <= len(job.items):
                logger.warning(f"Style guide model returned unknown item {verdict.item}")
                continue
            match = job.items[verdict.item - 1]
            findings.append(Finding(match.rule_id, match.start, match.end, match.text, Source.CONFIRMED))
        return findings
    rule_ids = {rule.rule_id for rule in job.rules}
    dropped = 0
    for quoted in _QuotedFindings.model_validate(answer).violations:
        if not quoted.breaks_rule:
            continue
        span = locate_quote(job.scope_text, quoted.quote) if quoted.rule_id in rule_ids else None
        if span is None:
            dropped += 1
            continue
        start, end = span[0] + job.scope_offset, span[1] + job.scope_offset
        findings.append(Finding(quoted.rule_id, start, end, job.scope_text[span[0] : span[1]], Source.MODEL))
    if dropped:
        logger.info(f"Dropped {dropped} style guide finding(s) whose quote isn't in the text")
    return findings


async def _run_jobs(
    jobs: list[_Job], llm: LLM, context: LlmCallContext, thinking: ThinkingLevel
) -> list[list[Finding] | BaseException]:
    """Run the jobs in parallel; each result is the job's findings or why it failed."""
    handler = BedrockHandler(llm=llm, mode=RunMode.ASYNC, max_tokens=settings.style_guide_checker_max_output_tokens)
    semaphore = asyncio.Semaphore(settings.style_guide_max_concurrent_calls)

    async def run(job: _Job) -> list[Finding]:
        async with semaphore:
            answer = await asyncio.wait_for(
                _call(job, handler, context, thinking), timeout=settings.style_guide_call_timeout_seconds
            )
        return _read_answer(job, answer)

    tasks = [asyncio.create_task(run(job)) for job in jobs]
    _, pending = await asyncio.wait(tasks, timeout=settings.style_guide_check_deadline_seconds)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    results: list[list[Finding] | BaseException] = []
    for task in tasks:
        if task in pending:
            results.append(TimeoutError("Style guide check deadline passed"))
        elif task.exception() is not None:
            results.append(task.exception())
        else:
            results.append(task.result())
    return results


def _dedupe(findings: list[Finding]) -> list[Finding]:
    """One finding per rule and start offset, preferring the most certain source; in text order."""
    best: dict[tuple[str, int], Finding] = {}
    for finding in findings:
        key = (finding.rule_id, finding.start)
        if key not in best or finding.source < best[key].source:
            best[key] = finding
    return sorted(best.values(), key=lambda f: (f.start, f.rule_id))


async def run_check(
    text: str,
    checker_context: str | None,
    context: LlmCallContext,
    rules: list[Rule] = RULES,
    model: str = STYLE_GUIDE_LLM_MODEL,
    thinking: ThinkingLevel | None = None,
) -> CheckResult:
    """Check text against the rules.

    Args:
        text: The content to check.
        checker_context: Guidance from the chat model for the model calls, or None.
        context: What the calls are for, recorded on their llm_internal_response rows.
        rules: The rules to apply.
        model: The LLM table's model id for the calls.
        thinking: Thinking level for the calls; defaults to the style_guide_thinking_level setting.

    Returns:
        The findings, and what couldn't be checked.

    Raises:
        StyleGuideCheckError: The model isn't in the LLM table.
    """
    thinking = thinking or settings.style_guide_thinking_level
    started = time.monotonic()
    matches = find_matches(text, rules)
    by_id = {rule.rule_id: rule for rule in rules}
    findings = [
        Finding(m.rule_id, m.start, m.end, m.text, Source.PATTERN)
        for rule_id, rule_matches in matches.items()
        if by_id[rule_id].llm == LlmKind.NONE
        for m in rule_matches
    ]
    jobs, not_reviewed = build_jobs(text, rules, matches, checker_context)
    result = CheckResult(not_reviewed=not_reviewed, calls=len(jobs))
    if jobs:
        llm = await get_llm(model)
        not_checked: dict[str, None] = {}
        for job, outcome in zip(jobs, await _run_jobs(jobs, llm, context, thinking), strict=True):
            if isinstance(outcome, BaseException):
                logger.error(f"Style guide {job.kind.value} call failed: {outcome!r}")
                result.failed_calls += 1
                not_checked.update(dict.fromkeys(rule.title for rule in job.rules))
            else:
                findings.extend(outcome)
        result.not_checked = list(not_checked)
    result.findings = _dedupe(findings)
    logger.info(
        f"Style guide check: {len(result.findings)} findings, {result.calls} calls "
        f"({result.failed_calls} failed) in {time.monotonic() - started:.1f}s"
    )
    return result
