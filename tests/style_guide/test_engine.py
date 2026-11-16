"""The checker engine, with the model calls mocked."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.bedrock.call_log import LlmCallContext
from app.config import ThinkingLevel
from app.style_guide.engine import Source, build_jobs, run_check
from app.style_guide.matching import find_matches
from app.style_guide.rules import LlmKind, parse_rule

pytestmark = [pytest.mark.unit]

CONTEXT = LlmCallContext(purpose="style_guide_validation", chat_id=1, message_id=2)


def make_rule(rule_id: str, priority: int = 100, **check):
    return parse_rule(
        {"rule_id": rule_id, "rule": rule_id.title(), "priority": priority, "details": "D.", "check": check}
    )


PATTERN_RULE = make_rule("rule_pattern", patterns=[r"\betc\b"], llm="none")
MATCH_RULE = make_rule("rule_match", patterns=[r"\bHub\b"], case_sensitive=True, llm="matches")
CHUNK_RULE = make_rule("rule_chunk", llm="chunks")
DOCUMENT_RULE = make_rule("rule_document", llm="document")
TEXT = "Apples etc. Visit the Hub today. The Hub is open."


def response(answer: dict, stop_reason: str = "end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=json.dumps(answer))])


class FakeModel:
    """Answers each call by the kind of prompt it gets, and records the requests."""

    def __init__(self, matches=None, chunks=None, document=None):
        self.answers = {"items": matches, 'content scope="the whole content"': chunks, "title, summary": document}
        self.requests = []

    async def invoke_async(self, messages, context, **kwargs):
        self.requests.append({"messages": messages, "context": context, **kwargs})
        prompt = messages[0]["content"]
        if "<items>" in prompt:
            answer = self.answers["items"]
        elif "title, summary and headings" in prompt:
            answer = self.answers["title, summary"]
        else:
            answer = self.answers['content scope="the whole content"']
        if isinstance(answer, BaseException):
            raise answer
        if callable(answer):
            return await answer()
        return answer if isinstance(answer, SimpleNamespace) else response(answer or {"violations": []})


async def check(fake: FakeModel, rules, text: str = TEXT, **kwargs):
    with (
        patch("app.style_guide.engine.get_llm", AsyncMock(return_value=Mock())),
        patch("app.style_guide.engine.BedrockHandler", return_value=fake),
    ):
        return await run_check(text, kwargs.pop("checker_context", None), CONTEXT, rules=rules, **kwargs)


@pytest.mark.asyncio
async def test_pattern_rules_need_no_model_call():
    fake = FakeModel()

    result = await check(fake, [PATTERN_RULE])

    assert fake.requests == []
    assert [(f.rule_id, f.text, f.source) for f in result.findings] == [("rule_pattern", "etc", Source.PATTERN)]
    assert result.calls == 0


@pytest.mark.asyncio
async def test_model_confirms_some_matches():
    fake = FakeModel(
        matches={
            "violations": [
                {"item": 1, "reason": "Words to avoid.", "breaks_rule": True},
                {"item": 2, "reason": "On reflection fine.", "breaks_rule": False},
                {"item": 9, "reason": "No such item.", "breaks_rule": True},
            ]
        }
    )

    result = await check(fake, [MATCH_RULE])

    [finding] = result.findings
    assert (finding.text, finding.start, finding.source) == ("Hub", TEXT.index("Hub"), Source.CONFIRMED)
    prompt = fake.requests[0]["messages"][0]["content"]
    assert "occurrence=\"2 of 2, first: 'Hub'\"" in prompt


@pytest.mark.asyncio
async def test_quotes_must_be_in_the_text():
    fake = FakeModel(
        chunks={
            "violations": [
                {
                    "rule_id": "rule_chunk",
                    "quote": "visit the hub",
                    "reason": "R",
                    "breaks_rule": True,
                },
                {
                    "rule_id": "rule_chunk",
                    "quote": "not in the text",
                    "reason": "R",
                    "breaks_rule": True,
                },
            ]
        }
    )

    result = await check(fake, [CHUNK_RULE])

    [finding] = result.findings
    assert (finding.text, finding.source) == ("Visit the Hub", Source.MODEL)
    assert finding.start == TEXT.index("Visit")


@pytest.mark.asyncio
async def test_requests_use_structured_output_and_the_call_context():
    fake = FakeModel()

    await check(fake, [MATCH_RULE, CHUNK_RULE, DOCUMENT_RULE], checker_context="For teachers")

    assert len(fake.requests) == 3
    for request in fake.requests:
        assert request["context"] is CONTEXT
        assert request["output_config"]["format"]["type"] == "json_schema"
        item = request["output_config"]["format"]["schema"]["properties"]["violations"]["items"]
        fields = list(item["properties"])
        assert fields.index("breaks_rule") == fields.index("reason") + 1
        assert "suggestion" not in fields
        assert request["thinking"] == {"type": "disabled"}
        assert "<assistant-context>\nFor teachers\n</assistant-context>" in request["messages"][0]["content"]


@pytest.mark.asyncio
async def test_thinking_level_sets_effort():
    fake = FakeModel()

    await check(fake, [CHUNK_RULE], thinking=ThinkingLevel.low)

    request = fake.requests[0]
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"]["effort"] == "low"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [RuntimeError("Bedrock down"), response({"violations": []}, stop_reason="max_tokens"), response({}, "refusal")],
)
async def test_failed_call_keeps_the_other_findings(answer):
    fake = FakeModel(matches=answer, chunks={"violations": []})

    result = await check(fake, [PATTERN_RULE, MATCH_RULE, CHUNK_RULE])

    assert result.failed_calls == 1
    assert result.not_checked == ["Rule_Match"]
    assert [f.rule_id for f in result.findings] == ["rule_pattern"]


@pytest.mark.asyncio
async def test_calls_past_the_deadline_are_cancelled(monkeypatch):
    monkeypatch.setattr("app.style_guide.engine.settings.style_guide_check_deadline_seconds", 0.05)

    async def slow():
        await asyncio.sleep(5)

    fake = FakeModel(matches=slow)

    result = await check(fake, [MATCH_RULE, CHUNK_RULE])

    assert result.failed_calls == 1
    assert result.not_checked == ["Rule_Match"]


@pytest.mark.asyncio
async def test_findings_are_deduplicated_preferring_the_more_certain_source():
    hub_pattern = make_rule("rule_chunk_and_pattern", patterns=[r"\bHub\b"], case_sensitive=True, llm="none")
    fake = FakeModel()

    result = await check(fake, [hub_pattern])
    assert [f.start for f in result.findings] == [TEXT.index("Hub"), TEXT.rindex("Hub")]

    from app.style_guide.engine import Finding, _dedupe

    start = TEXT.index("Hub")
    kept = _dedupe(
        [Finding("r", start, start + 3, "Hub", Source.MODEL), Finding("r", start, start + 3, "Hub", Source.PATTERN)]
    )
    assert [f.source for f in kept] == [Source.PATTERN]


class TestBuildJobs:
    def test_matches_are_batched_with_only_their_rules(self, monkeypatch):
        monkeypatch.setattr("app.style_guide.engine.settings.style_guide_match_batch_size", 2)
        etc_rule = make_rule("rule_etc", patterns=[r"\betc\b"], llm="matches")
        rules = [etc_rule, MATCH_RULE]

        jobs, _ = build_jobs(TEXT, rules, find_matches(TEXT, rules), None)

        assert [(j.kind, [m.text for m in j.items], [r.rule_id for r in j.rules]) for j in jobs] == [
            (LlmKind.MATCHES, ["etc", "Hub"], ["rule_etc", "rule_match"]),
            (LlmKind.MATCHES, ["Hub"], ["rule_match"]),
        ]

    def test_per_rule_limit_reports_the_rest(self, monkeypatch):
        monkeypatch.setattr("app.style_guide.engine.settings.style_guide_max_matches_per_rule", 1)

        jobs, not_reviewed = build_jobs(TEXT, [MATCH_RULE], find_matches(TEXT, [MATCH_RULE]), None)

        assert [len(j.items) for j in jobs] == [1]
        assert not_reviewed == {"rule_match": 1}

    def test_first_only_sends_the_first_of_each_text(self):
        rule = make_rule(
            "rule_abbr", patterns=[r"\b[A-Z]{2,}s?\b"], case_sensitive=True, llm="matches", first_only=True
        )
        text = "PIP and ESA. PIP again, ESAs too, and DLA."

        jobs, not_reviewed = build_jobs(text, [rule], find_matches(text, [rule]), None)

        assert [m.text for m in jobs[0].items] == ["PIP", "ESA", "DLA"]
        assert not_reviewed == {}

    def test_chunk_rules_get_one_call_per_chunk(self, monkeypatch):
        monkeypatch.setattr("app.style_guide.engine.settings.style_guide_chunk_chars", 60)
        text = "\n\n".join(["A paragraph of some forty characters long."] * 4)

        jobs, _ = build_jobs(text, [CHUNK_RULE], {}, None)

        assert len(jobs) == 4
        assert all(text[j.scope_offset :].startswith(j.scope_text) for j in jobs)
        assert 'scope="part 1 of 4 of the content"' in jobs[0].prompt

    def test_document_rules_get_a_digest_of_long_text(self, monkeypatch):
        monkeypatch.setattr("app.style_guide.engine.settings.style_guide_chunk_chars", 100)
        monkeypatch.setattr("app.style_guide.engine.DOCUMENT_DIGEST_HEAD_CHARS", 50)
        text = "Title line\n\n" + "Body sentence here. " * 20 + "\n\nA later heading\n\nMore body text here."

        [job] = build_jobs(text, [DOCUMENT_RULE], {}, None)[0]

        assert "A later heading" in job.prompt
        assert "More body text here." not in job.prompt
        assert job.scope_text == text
