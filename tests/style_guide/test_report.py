"""The tool result built from a check's findings."""

import pytest

from app.config import StyleGuideThoroughness
from app.style_guide.engine import CheckResult, Finding, Source
from app.style_guide.report import format_result

pytestmark = [pytest.mark.unit]

TEXT = "You can't use the Hub, etc. We organize events. You can't stop."


def finding(rule_id: str, text: str, source: Source, nth: int = 0, **kwargs) -> Finding:
    start = -1
    for _ in range(nth + 1):
        start = TEXT.index(text, start + 1)
    return Finding(rule_id, start, start + len(text), text, source, **kwargs)


def test_no_findings():
    content, failed = format_result(CheckResult(), TEXT, "the text passed in")

    assert not failed
    assert "No issues found" in content


def test_issues_are_grouped_by_rule_and_ordered_by_priority_then_source():
    result = CheckResult(
        findings=[
            finding("rule_024", "organize", Source.CONFIRMED),  # priority 0, confirmed
            finding("rule_099", "can't", Source.PATTERN),  # priority 0, pattern
            finding("rule_099", "can't", Source.PATTERN, nth=1),
            finding("rule_157", "etc", Source.CONFIRMED),  # priority 0, confirmed
            finding("rule_547", "Hub", Source.CONFIRMED),  # priority 0
            finding("rule_501", "You can't use", Source.MODEL),  # priority 100
        ]
    )

    content, failed = format_result(result, TEXT, "the text passed in")

    assert not failed
    assert "6 issues found under 5 rules" in content
    order = [content.index(f"<rule>{title}</rule>") for title in ("contractions", "Words to avoid", "Titles")]
    assert order == sorted(order)
    assert content.index("<rule>contractions</rule>") < content.index("<rule>American and UK English</rule>")
    assert "<count>2</count>" in content
    for removed in ("<reason>", "<suggestion>", "<detected-by>", "<also-found>", "<context>"):
        assert removed not in content


def test_each_issue_has_rule_guidance_count_and_one_example_sentence():
    text = "First we organize. Then we organize again."
    findings = [
        Finding("rule_024", text.index("organize"), text.index("organize") + 8, "organize", Source.PATTERN),
        Finding("rule_024", text.rindex("organize"), text.rindex("organize") + 8, "organize", Source.PATTERN),
    ]

    content, _ = format_result(CheckResult(findings=findings), text, "the text")

    issue = content[content.index("<issue>") : content.index("</issue>")]
    assert [line.split(">")[0] + ">" for line in issue.splitlines()[1:]] == [
        "<rule>",
        "<guidance>",
        "<count>",
        "<example>",
    ]
    assert "<count>2</count>" in issue
    assert "<example>First we organize.</example>" in issue


def test_an_example_starting_with_a_line_break_shows_its_own_line():
    text = "Do this:\n- first thing\n- second thing"
    start = text.index("\n- second")
    finding_ = Finding("rule_055", start, start + 3, "\n- ", Source.PATTERN)

    content, _ = format_result(CheckResult(findings=[finding_]), text, "the text")

    assert "<example>- second thing</example>" in content


def test_an_example_longer_than_its_sentence_is_kept_whole():
    text = "One sentence. Another one here."
    finding_ = Finding("rule_371", 0, len(text), text, Source.MODEL)

    content, _ = format_result(CheckResult(findings=[finding_]), text, "the text")

    assert f"<example>{text}</example>" in content


def test_issues_past_the_size_limit_are_named_with_counts(monkeypatch):
    monkeypatch.setattr("app.style_guide.report.settings.style_guide_max_result_chars", 600)
    result = CheckResult(
        findings=[finding("rule_099", "can't", Source.PATTERN), finding("rule_501", "You can't use", Source.MODEL)]
    )

    content, _ = format_result(result, TEXT, "the text")

    assert "<rule>contractions</rule>" in content
    assert "<rule>Titles</rule>" not in content
    assert "<left-out>" in content and "Titles (1)" in content


def test_thorough_checks_use_the_larger_size_limit(monkeypatch):
    monkeypatch.setattr("app.style_guide.report.settings.style_guide_max_result_chars", 600)
    monkeypatch.setattr("app.style_guide.report.settings.style_guide_max_result_chars_thorough", 5000)
    result = CheckResult(
        findings=[finding("rule_099", "can't", Source.PATTERN), finding("rule_501", "You can't use", Source.MODEL)]
    )

    content, _ = format_result(result, TEXT, "the text", StyleGuideThoroughness.thorough)

    assert "<rule>Titles</rule>" in content
    assert "<left-out>" not in content


def test_failed_calls_are_reported():
    result = CheckResult(
        findings=[finding("rule_099", "can't", Source.PATTERN)], not_checked=["Titles"], failed_calls=1
    )

    content, failed = format_result(result, TEXT, "the text")

    assert not failed
    assert "<warning>" in content
    assert "<not-checked>Titles</not-checked>" in content


def test_failure_with_no_findings_is_an_error():
    content, failed = format_result(CheckResult(not_checked=["Titles"], failed_calls=1), TEXT, "the text")

    assert failed
    assert "<error>" in content


def test_unreviewed_matches_are_reported():
    content, _ = format_result(CheckResult(not_reviewed={"rule_063": 12}), TEXT, "the text")

    assert "Capitalisation (12)" in content


def test_rules_left_out_by_thoroughness_are_reported():
    content, _ = format_result(CheckResult(), TEXT, "the text", StyleGuideThoroughness.lite, 340)

    assert "<thoroughness>lite: 340 lower-priority rules (specific terms, spellings and capitalisation)" in content
    assert "thoroughness 'thorough'" in content
