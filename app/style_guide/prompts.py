"""Prompt text for the style guide checker's LLM calls (engine.py)."""

from app.style_guide.matching import Match, match_context
from app.style_guide.rules import Rule

SYSTEM_PROMPT = """You check UK government content against the GOV.UK style guide. You are given some of the guide's \
rules and some or all of a piece of content, and you report where the content clearly breaks those rules.

- Judge only against the rules given, as their guidance describes them. Don't add requirements the guidance doesn't \
state, and don't report problems that belong to other rules.
- Report only clear breaches. If a reasonable reading makes the text fine, leave it out. Proper nouns, the names of \
organisations, schemes and documents, quoted text, and uses the guidance allows are not breaches.
- The content was extracted from a document or pasted by a user, so links, italics and bold may have been lost. Don't \
report anything that depends on formatting you can't see.
- Context from the assistant, when given, says what the content is or what to focus on. Use it to judge scope; it is \
not part of the content.
- For each possible breach, give a reason in one short sentence saying what the guidance requires, then \
breaks_rule: true if the text breaks the rule, false if on reflection it doesn't."""

MATCHES_INSTRUCTIONS = """Each item below is text that a pattern found for one of these rules. Decide for each item \
whether that use breaks its rule, given its context. Many items are correct uses: list only the ones that break their \
rule, by item id.

An item's occurrence is its place among all the matches for that rule in the whole content, with the text of the \
first match. Use it for rules about first use and later mentions: for example, an abbreviation needs explaining at its \
first occurrence, not afterwards."""

CHUNK_INSTRUCTIONS = """Read the content below for breaches of these rules. Quote each breach exactly as it appears in \
the content, character for character: the words that break the rule, with only enough around them to find them.

Other checks already cover capitalisation, block capitals, punctuation, abbreviations, spelling, numbers, dates, \
specific words to avoid and the A to Z of terms. Report only what these rules themselves are about."""

DOCUMENT_INSTRUCTIONS = """These rules are about the content as a whole: its title, summary and headings. If the \
content has no title or no summary, for example because it is a passage or a message rather than a page, report \
nothing for the rule concerned. Quote each breach exactly as it appears in the content, character for character."""

MATCH_VERDICTS_SCHEMA = {
    "type": "object",
    "properties": {
        "violations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": "integer"},
                    "reason": {"type": "string"},
                    "breaks_rule": {"type": "boolean"},
                },
                "required": ["item", "reason", "breaks_rule"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["violations"],
    "additionalProperties": False,
}


def findings_schema(rule_ids: list[str]) -> dict:
    """The output schema for chunk and document calls, limited to the rules in the call."""
    return {
        "type": "object",
        "properties": {
            "violations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rule_id": {"type": "string", "enum": rule_ids},
                        "quote": {"type": "string"},
                        "reason": {"type": "string"},
                        "breaks_rule": {"type": "boolean"},
                    },
                    "required": ["rule_id", "quote", "reason", "breaks_rule"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["violations"],
        "additionalProperties": False,
    }


def _rules_block(rules: list[Rule]) -> str:
    lines = [f'<rule id="{rule.rule_id}" name="{rule.title}">{rule.details}</rule>' for rule in rules]
    return "<rules>\n" + "\n".join(lines) + "\n</rules>"


def _assistant_context(checker_context: str | None) -> str:
    if not checker_context or not checker_context.strip():
        return ""
    return f"\n<assistant-context>\n{checker_context.strip()}\n</assistant-context>"


def matches_prompt(text: str, rules: list[Rule], items: list[Match], checker_context: str | None) -> str:
    """The user message for judging pattern matches in context.

    Args:
        text: The whole content, for each match's context.
        rules: The rules the items belong to.
        items: The matches to judge; item ids are their 1-based positions in this list.
        checker_context: Guidance from the chat model, or None.
    """
    parts = []
    for item_id, match in enumerate(items, 1):
        occurrence = f"{match.index} of {match.total}"
        if match.index > 1:
            occurrence += f", first: {match.first_text!r}"
        lines = [
            f'<item id="{item_id}" rule="{match.rule_id}" occurrence="{occurrence}">',
            f"<match>{match.text}</match>",
        ]
        if match.suggestion:
            lines.append(f"<expected-form>{match.suggestion}</expected-form>")
        lines.append(f"<context>{match_context(text, match.start, match.end)}</context>")
        lines.append("</item>")
        parts.append("\n".join(lines))
    return (
        f"{MATCHES_INSTRUCTIONS}\n\n{_rules_block(rules)}{_assistant_context(checker_context)}\n\n"
        "<items>\n" + "\n".join(parts) + "\n</items>"
    )


def chunk_prompt(chunk: str, part: int, parts: int, rules: list[Rule], checker_context: str | None) -> str:
    """The user message for reading one chunk of the content for the open-ended rules."""
    where = "the whole content" if parts == 1 else f"part {part} of {parts} of the content"
    return (
        f"{CHUNK_INSTRUCTIONS}\n\n{_rules_block(rules)}{_assistant_context(checker_context)}\n\n"
        f'<content scope="{where}">\n{chunk}\n</content>'
    )


def document_prompt(digest: str, is_whole: bool, rules: list[Rule], checker_context: str | None) -> str:
    """The user message for the rules judged once from the whole content or its outline."""
    scope = "the whole content" if is_whole else "the start of the content, then its other short lines"
    return (
        f"{DOCUMENT_INSTRUCTIONS}\n\n{_rules_block(rules)}{_assistant_context(checker_context)}\n\n"
        f'<content scope="{scope}">\n{digest}\n</content>'
    )
