"""The rule data in rule_mapping.json and how it is loaded."""

import json

import pytest

from app.config import StyleGuideThoroughness
from app.style_guide.rules import PRIORITIES, RULES, RULES_FILE, LlmKind, parse_rule, rules_for

pytestmark = [pytest.mark.unit]

RAW_RULES = json.loads(RULES_FILE.read_text(encoding="utf-8"))


def raw_rule(**check) -> dict:
    return {"rule_id": "rule_x", "rule": "Example", "priority": 100, "details": "Guidance.", "check": check}


def test_every_rule_has_a_valid_priority():
    bad = [r["rule_id"] for r in RAW_RULES if r.get("priority") not in PRIORITIES]
    assert bad == []


def test_rule_ids_are_unique():
    ids = [r["rule_id"] for r in RAW_RULES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("raw", RAW_RULES, ids=lambda r: r["rule_id"])
def test_every_rule_check_is_valid(raw):
    parse_rule(raw)


def test_redundant_rules_are_left_out():
    redundant = {r["rule_id"] for r in RAW_RULES if r.get("redundant")}
    assert redundant
    assert not redundant & {rule.rule_id for rule in RULES}


def test_every_kind_is_used():
    assert {rule.llm for rule in RULES} == set(LlmKind)


def test_patterns_are_case_insensitive_unless_marked():
    insensitive = parse_rule(raw_rule(patterns=["abc"], llm="none"))
    sensitive = parse_rule(raw_rule(patterns=["abc"], case_sensitive=True, llm="none"))

    assert insensitive.patterns[0].search("ABC")
    assert not sensitive.patterns[0].search("ABC")


@pytest.mark.parametrize(
    "check, message",
    [
        ({"llm": "matches"}, "needs patterns"),
        ({"patterns": ["x"], "llm": "chunks"}, "takes no patterns"),
        ({"matcher": "unknown", "llm": "none"}, "unknown matcher"),
    ],
)
def test_invalid_checks_are_rejected(check, message):
    with pytest.raises(ValueError, match=message):
        parse_rule(raw_rule(**check))


def test_invalid_priority_is_rejected():
    raw = raw_rule(patterns=["x"], llm="none") | {"priority": 50}

    with pytest.raises(ValueError, match="priority"):
        parse_rule(raw)


@pytest.mark.parametrize(
    ("thoroughness", "priorities"),
    [
        (StyleGuideThoroughness.lite, {0, 100}),
        (StyleGuideThoroughness.balanced, {0, 100, 1000}),
        (StyleGuideThoroughness.thorough, {0, 100, 1000, 2000}),
    ],
)
def test_thoroughness_runs_rules_up_to_its_priority(thoroughness, priorities):
    assert {rule.priority for rule in rules_for(thoroughness)} == priorities


def test_thorough_runs_every_rule():
    assert rules_for(StyleGuideThoroughness.thorough) == RULES
