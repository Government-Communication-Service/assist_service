"""The style guide rules from rule_mapping.json, with their checks compiled."""

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.config import StyleGuideThoroughness

RULES_FILE = Path(__file__).parent / "rule_mapping.json"
PRIORITIES = (0, 100, 1000, 2000)
MATCHERS = ("american_spelling", "long_sentences")
# The lowest priority (highest number) each thoroughness runs.
MAX_PRIORITY = {
    StyleGuideThoroughness.lite: 100,
    StyleGuideThoroughness.balanced: 1000,
    StyleGuideThoroughness.thorough: 2000,
}


class LlmKind(StrEnum):
    """What the model does for a rule.

    none: every pattern match is a finding. matches: the model judges each pattern match in its context.
    chunks: the model reads the text for breaches. document: the model judges the title, summary and
    headings once.
    """

    NONE = "none"
    MATCHES = "matches"
    CHUNKS = "chunks"
    DOCUMENT = "document"


@dataclass(frozen=True)
class CorrectForm:
    """A case rule's correct spelling, and the pattern that finds it in any case."""

    form: str
    pattern: re.Pattern


@dataclass(frozen=True)
class Rule:
    """One rule and its check.

    Attributes:
        rule_id: Stable id, e.g. "rule_042".
        title: The rule's name in the style guide.
        details: The style guide's guidance for it.
        priority: 0, 100, 1000 or 2000; lower is reported first.
        llm: What the model does for the rule.
        patterns: Regexes for wrong forms, or for text the model should judge.
        correct_forms: Correct spellings: any other casing is a match, except a capital first letter at
            the start of a sentence or line.
        matcher: A named built-in matcher instead of patterns: "american_spelling" or "long_sentences".
        max_words: Sentence length limit for the long_sentences matcher.
        first_only: Send only the first match of each distinct text to the model (for first-use rules).
    """

    rule_id: str
    title: str
    details: str
    priority: int
    llm: LlmKind
    patterns: tuple[re.Pattern, ...] = ()
    correct_forms: tuple[CorrectForm, ...] = ()
    matcher: str | None = None
    max_words: int = 25
    first_only: bool = False


def _correct_form(form: str, whole_word: bool) -> CorrectForm:
    body = re.escape(form)
    if whole_word:
        start = r"\b" if re.match(r"\w", form) else ""
        end = r"s?\b" if re.search(r"\w$", form) else ""
        body = start + body + end
    return CorrectForm(form, re.compile(body, re.IGNORECASE))


def parse_rule(raw: dict) -> Rule:
    """Build a Rule from one rule_mapping.json entry.

    Raises:
        ValueError: The entry's priority or check is invalid.
    """
    rule_id = raw["rule_id"]
    check = raw.get("check")
    if raw.get("priority") not in PRIORITIES:
        raise ValueError(f"{rule_id}: priority must be one of {PRIORITIES}")
    if not isinstance(check, dict):
        raise ValueError(f"{rule_id}: missing check")
    llm = LlmKind(check.get("llm"))
    flags = 0 if check.get("case_sensitive") else re.IGNORECASE
    patterns = tuple(re.compile(p, flags) for p in check.get("patterns", []))
    whole_word = check.get("whole_word", True)
    correct_forms = tuple(_correct_form(f, whole_word) for f in check.get("correct_forms", []))
    matcher = check.get("matcher")
    if matcher is not None and matcher not in MATCHERS:
        raise ValueError(f"{rule_id}: unknown matcher {matcher!r}")
    has_trigger = bool(patterns or correct_forms or matcher)
    if llm in (LlmKind.NONE, LlmKind.MATCHES) and not has_trigger and not raw.get("redundant"):
        raise ValueError(f"{rule_id}: llm {llm.value!r} needs patterns, correct_forms or a matcher")
    if llm in (LlmKind.CHUNKS, LlmKind.DOCUMENT) and has_trigger:
        raise ValueError(f"{rule_id}: llm {llm.value!r} takes no patterns")
    return Rule(
        rule_id=rule_id,
        title=raw["rule"],
        details=raw["details"],
        priority=raw["priority"],
        llm=llm,
        patterns=patterns,
        correct_forms=correct_forms,
        matcher=matcher,
        max_words=check.get("max_words", 25),
        first_only=check.get("first_only", False),
    )


def load_rules(path: Path = RULES_FILE) -> list[Rule]:
    """Load the rules in file order, leaving out redundant ones."""
    raw_rules = json.loads(path.read_text(encoding="utf-8"))
    return [parse_rule(raw) for raw in raw_rules if not raw.get("redundant")]


RULES = load_rules()
RULES_BY_ID = {rule.rule_id: rule for rule in RULES}


def rules_for(thoroughness: StyleGuideThoroughness, rules: list[Rule] = RULES) -> list[Rule]:
    """The rules a check at this thoroughness runs."""
    return [rule for rule in rules if rule.priority <= MAX_PRIORITY[thoroughness]]
