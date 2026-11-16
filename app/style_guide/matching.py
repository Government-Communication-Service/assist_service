"""Find where the rules' patterns match in a text, and split text into chunks for the model."""

import re
from dataclasses import dataclass

from breame.spelling import american_spelling_exists, get_british_spelling
from uwotm8.convert import CONVERSION_IGNORE_LIST

from app.style_guide.rules import Rule

_SENTENCE_END = ".!?"
_LIST_MARKER = re.compile(r"(?:[-•*]|\d+[.)])[ \t]+$")
_WORD = re.compile(r"\b[a-zA-Z]+\b")
_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.MULTILINE)
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_PLAIN_QUOTES = str.maketrans("‘’“”–—", "''\"\"--")


@dataclass(frozen=True)
class Match:
    """One place a rule's check matched.

    Attributes:
        rule_id: The rule.
        start: Offset of the match in the text.
        end: Offset just after it.
        text: The matched text.
        suggestion: A replacement the matcher knows, such as the correct form or British spelling.
        index: Position among the rule's matches, from 1, in text order.
        total: How many matches the rule has in the whole text.
        first_text: The rule's first match.
    """

    rule_id: str
    start: int
    end: int
    text: str
    suggestion: str | None = None
    index: int = 1
    total: int = 1
    first_text: str = ""


def is_sentence_start(text: str, position: int) -> bool:
    """Whether a position starts a sentence, a line or a list item."""
    line_start = text.rfind("\n", 0, position) + 1
    before_on_line = text[line_start:position]
    if not before_on_line.strip() or _LIST_MARKER.fullmatch(before_on_line.lstrip()):
        return True
    preceding = text[:position].rstrip()
    return not preceding or preceding[-1] in _SENTENCE_END


def sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """The sentence (or line) around a span, as offsets, without surrounding whitespace."""
    left = start
    while left > 0 and text[left - 1] not in _SENTENCE_END + "\n":
        left -= 1
    right = end
    while right < len(text) and text[right] not in _SENTENCE_END + "\n":
        right += 1
    if right < len(text) and text[right] in _SENTENCE_END:
        right += 1
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    return left, right


def match_context(text: str, start: int, end: int) -> str:
    """The sentence holding a span, with the sentence before it."""
    left, right = sentence_bounds(text, start, end)
    previous = text[:left].rstrip()
    if previous:
        prev_left, _ = sentence_bounds(text, len(previous) - 1, len(previous))
        left = prev_left
    return text[left:right]


def _case_matches(text: str, rule: Rule) -> list[Match]:
    matches = []
    for correct in rule.correct_forms:
        capitalised = correct.form[:1].upper() + correct.form[1:]
        for m in correct.pattern.finditer(text):
            found = m.group()
            base = found[: len(correct.form)]
            if base == correct.form:
                continue
            if base == capitalised and is_sentence_start(text, m.start()):
                continue
            matches.append(Match(rule.rule_id, m.start(), m.end(), found, suggestion=correct.form))
    return matches


def _american_spellings(text: str, rule: Rule) -> list[Match]:
    matches = []
    for m in _WORD.finditer(text):
        word = m.group().lower()
        if word in CONVERSION_IGNORE_LIST or not american_spelling_exists(word):
            continue
        try:
            british = get_british_spelling(word)
        except Exception:
            british = None
        matches.append(Match(rule.rule_id, m.start(), m.end(), m.group(), suggestion=british))
    return matches


def _long_sentences(text: str, rule: Rule) -> list[Match]:
    matches = []
    for m in _SENTENCE.finditer(text):
        sentence = m.group()
        if len(re.findall(r"\b\w+\b", sentence)) <= rule.max_words:
            continue
        stripped = sentence.strip()
        start = m.start() + sentence.index(stripped)
        matches.append(Match(rule.rule_id, start, start + len(stripped), stripped))
    return matches


def rule_matches(text: str, rule: Rule) -> list[Match]:
    """Every match of one rule's check, in text order and without duplicate starts."""
    if rule.matcher == "american_spelling":
        found = _american_spellings(text, rule)
    elif rule.matcher == "long_sentences":
        found = _long_sentences(text, rule)
    else:
        found = [
            Match(rule.rule_id, m.start(), m.end(), m.group())
            for pattern in rule.patterns
            for m in pattern.finditer(text)
            if m.group()
        ]
        found += _case_matches(text, rule)
    unique: dict[int, Match] = {}
    for match in sorted(found, key=lambda m: (m.start, -m.end)):
        unique.setdefault(match.start, match)
    ordered = list(unique.values())
    first_text = ordered[0].text if ordered else ""
    return [
        Match(m.rule_id, m.start, m.end, m.text, m.suggestion, i, len(ordered), first_text)
        for i, m in enumerate(ordered, 1)
    ]


def find_matches(text: str, rules: list[Rule]) -> dict[str, list[Match]]:
    """Run every rule's check over the whole text.

    Returns:
        Matches by rule id, each list in text order and tagged with its position among the rule's
        matches. Rules without matches are left out.
    """
    by_rule = {}
    for rule in rules:
        matches = rule_matches(text, rule)
        if matches:
            by_rule[rule.rule_id] = matches
    return by_rule


def _split_long(text: str, offset: int, max_chars: int) -> list[tuple[int, str]]:
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind(ch, start, end) for ch in _SENTENCE_END + "\n")
            if boundary > start:
                end = boundary + 1
        pieces.append((offset + start, text[start:end]))
        start = end
    return pieces


def split_into_chunks(text: str, max_chars: int) -> list[tuple[int, str]]:
    """Split text into chunks of about max_chars, at paragraph breaks where possible.

    A paragraph longer than max_chars is split at sentence ends, or cut if it has none.

    Returns:
        (offset, chunk) pairs in order; the chunks join back into the text, without the whitespace
        between paragraphs.
    """
    paragraphs = []
    position = 0
    for brk in _PARAGRAPH_BREAK.finditer(text):
        paragraphs.append((position, text[position : brk.start()]))
        position = brk.end()
    paragraphs.append((position, text[position:]))

    chunks: list[tuple[int, str]] = []
    current_start, current_end = None, None
    for start, paragraph in paragraphs:
        if not paragraph.strip():
            continue
        end = start + len(paragraph)
        if len(paragraph) > max_chars:
            if current_start is not None:
                chunks.append((current_start, text[current_start:current_end]))
                current_start = None
            chunks.extend(_split_long(paragraph, start, max_chars))
            continue
        if current_start is not None and end - current_start > max_chars:
            chunks.append((current_start, text[current_start:current_end]))
            current_start = None
        if current_start is None:
            current_start = start
        current_end = end
    if current_start is not None:
        chunks.append((current_start, text[current_start:current_end]))
    return chunks


def locate_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Find a model's quote in the text, ignoring differences in whitespace and case.

    Returns:
        The span of the first occurrence, or None if the quote isn't in the text.
    """
    quote = quote.strip().translate(_PLAIN_QUOTES)
    if not quote:
        return None
    text = text.translate(_PLAIN_QUOTES)
    exact = text.find(quote)
    if exact >= 0:
        return exact, exact + len(quote)
    words = [re.escape(w) for w in quote.split()]
    m = re.search(r"\s+".join(words), text, re.IGNORECASE)
    return (m.start(), m.end()) if m else None
