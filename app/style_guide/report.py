"""Format a style guide check's findings as the tool result for the chat model."""

from dataclasses import dataclass

from app.config import StyleGuideThoroughness, settings
from app.style_guide.engine import CheckResult, Finding
from app.style_guide.matching import sentence_bounds
from app.style_guide.rules import RULES_BY_ID, Rule

TAG = "style-guide-check"
MAX_EXAMPLE_CHARS = 400
SKIPPED_AT = {
    StyleGuideThoroughness.lite: "specific terms, spellings and capitalisation",
    StyleGuideThoroughness.balanced: "mostly capitalisation of individual terms",
}


def tagged(body: str) -> str:
    return f"<{TAG}>\n{body}\n</{TAG}>"


@dataclass
class _Group:
    rule: Rule
    findings: list[Finding]

    @property
    def sort_key(self) -> tuple:
        return (self.rule.priority, min(f.source for f in self.findings), self.findings[0].start)


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _example(finding: Finding, text: str) -> str:
    """The sentence the finding is in, or the finding itself if it is longer, such as a whole paragraph."""
    start = finding.start + len(finding.text) - len(finding.text.lstrip())
    end = max(start, finding.end - (len(finding.text) - len(finding.text.rstrip())))
    left, right = sentence_bounds(text, start, end)
    return _shorten(text[min(left, start) : max(right, end)], MAX_EXAMPLE_CHARS)


def _format_group(group: _Group, text: str) -> str:
    rule, findings = group.rule, group.findings
    return "\n".join(
        [
            "<issue>",
            f"<rule>{rule.title}</rule>",
            f"<guidance>{_shorten(rule.details, settings.style_guide_max_guidance_chars)}</guidance>",
            f"<count>{len(findings)}</count>",
            f"<example>{_example(findings[0], text)}</example>",
            "</issue>",
        ]
    )


def _thoroughness(thoroughness: StyleGuideThoroughness, skipped_rules: int) -> str:
    if not skipped_rules:
        return f"<thoroughness>{thoroughness.value}: all rules were run.</thoroughness>"
    return (
        f"<thoroughness>{thoroughness.value}: {skipped_rules} lower-priority rules ({SKIPPED_AT[thoroughness]}) "
        "were not run. Checking again with thoroughness 'thorough' covers them.</thoroughness>"
    )


def format_result(
    result: CheckResult,
    text: str,
    checked: str,
    thoroughness: StyleGuideThoroughness | None = None,
    skipped_rules: int = 0,
) -> tuple[str, bool]:
    """Turn a check's findings into the tool result.

    Issues are grouped by rule and ordered by rule priority, then by how they were found (pattern only,
    then confirmed by the model, then found by the model). Groups past the size limit, which is doubled
    for a thorough check, are listed by rule name and count only.

    Args:
        result: The check's result.
        text: The content that was checked.
        checked: What was checked, in words, e.g. "the text passed in".
        thoroughness: The thoroughness the check ran at, or None to leave it out of the result.
        skipped_rules: How many rules the thoroughness left out.

    Returns:
        The <style-guide-check> segment, and whether the check couldn't be run.
    """
    parts = [f"<checked>{checked}</checked>"]
    if thoroughness is not None:
        parts.append(_thoroughness(thoroughness, skipped_rules))
    if result.failed_calls:
        parts.append(
            "<warning>Part of this check failed to run, so some issues may be missing. Say so when reporting the "
            "results, naming the rules not checked, and don't tell the user the text is free of issues.</warning>"
        )
        parts.append(f"<not-checked>{'; '.join(result.not_checked)}</not-checked>")
    if result.not_reviewed:
        counts = "; ".join(f"{RULES_BY_ID[r].title} ({n})" for r, n in result.not_reviewed.items())
        parts.append(
            "<not-reviewed>The text had too many possible matches for these rules to review them all; the "
            f"number not reviewed is in brackets. Checking a shorter section will cover them: {counts}</not-reviewed>"
        )
    if not result.findings:
        if result.failed_calls:
            return tagged("\n".join(parts + ["<error>The check could not be completed. Try again.</error>"])), True
        parts.append("<result>No issues found against the GOV.UK style guide rules this check covers.</result>")
        return tagged("\n".join(parts)), False

    grouped: dict[str, list[Finding]] = {}
    for finding in result.findings:
        grouped.setdefault(finding.rule_id, []).append(finding)
    groups = sorted((_Group(RULES_BY_ID[r], f) for r, f in grouped.items()), key=lambda g: g.sort_key)
    issue_word = "issue" if len(result.findings) == 1 else "issues"
    rule_word = "rule" if len(groups) == 1 else "rules"
    parts.append(
        f"<result>{len(result.findings)} {issue_word} found under {len(groups)} {rule_word}, listed most "
        "important first.</result>"
    )

    max_chars = (
        settings.style_guide_max_result_chars_thorough
        if thoroughness == StyleGuideThoroughness.thorough
        else settings.style_guide_max_result_chars
    )
    budget = max_chars - sum(len(p) for p in parts)
    listed, left_out = [], []
    for group in groups:
        formatted = _format_group(group, text)
        if not left_out and (len(formatted) <= budget or not listed):
            listed.append(formatted)
            budget -= len(formatted)
        else:
            left_out.append(group)
    parts.extend(listed)
    if left_out:
        names = "; ".join(f"{g.rule.title} ({len(g.findings)})" for g in left_out)
        parts.append(
            "<left-out>Issues under these rules were found but not listed, to keep this result short; the "
            f"number found is in brackets. Checking a shorter section will list them: {names}</left-out>"
        )
    return tagged("\n".join(parts)), False
