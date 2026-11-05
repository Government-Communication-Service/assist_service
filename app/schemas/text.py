"""Validators that clean free-text request fields.

Add one after `StringConstraints`; before-validators run ahead of the core checks, so length limits apply to the
cleaned value:

    ProjectName = Annotated[str, StringConstraints(min_length=1, max_length=255), CleanSingleLine]
"""

import unicodedata
from typing import Any

from pydantic import BeforeValidator

ZERO_WIDTH_JOINERS = "\u200c\u200d"
BLACK_FLAG = "\U0001f3f4"
FLAG_TAG_RANGE = range(0xE0020, 0xE0080)
# Removed from the start and end of values, in addition to Unicode whitespace: zero-width characters, BOM and
# blank-looking letters/symbols (Hangul fillers, Braille blank).
INVISIBLE_EDGE_CHARACTERS = "\u200b\u200c\u200d\u2060\ufeff\u115f\u1160\u3164\uffa0\u2800"
# Removed anywhere in single-line text: soft hyphen and directional marks, which come along with pasted text.
SINGLE_LINE_REMOVED_CHARACTERS = dict.fromkeys(map(ord, "\u00ad\u200e\u200f\u061c"))
SINGLE_LINE_DISALLOWED_CATEGORIES = ("Cc", "Cf", "Zl", "Zp")
# Windows-1252 quotes and dashes decoded as Latin-1 arrive as C1 control characters; convert them to ASCII.
MISDECODED_PUNCTUATION = {0x91: "'", 0x92: "'", 0x93: '"', 0x94: '"', 0x96: "-", 0x97: "-"}
# Converted to "\n" in multiline text.
MULTILINE_LINE_BREAKS = dict.fromkeys(map(ord, "\r\v\f\x85\u2028\u2029"), "\n")
# Removed anywhere in multiline text: zero-width space, word joiner, BOM, bidi embeddings/overrides/isolates.
MULTILINE_REMOVED_CHARACTERS = dict.fromkeys(
    map(ord, "\u200b\u2060\ufeff\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")
)
MULTILINE_ALLOWED_CONTROL_CHARACTERS = "\n\t"


def _strip_edges(value: str) -> str:
    previous = None
    while previous != value:
        previous = value
        value = value.strip().strip(INVISIBLE_EDGE_CHARACTERS)
    return value


def _is_joiner_between_non_ascii(value: str, index: int) -> bool:
    return 0 < index < len(value) - 1 and not value[index - 1].isascii() and not value[index + 1].isascii()


def _is_flag_tag(value: str, index: int) -> bool:
    """Whether the tag character at `index` is part of a flag sequence such as England's: U+1F3F4 then tags."""
    start = index
    while start > 0 and ord(value[start - 1]) in FLAG_TAG_RANGE:
        start -= 1
    return start > 0 and value[start - 1] == BLACK_FLAG


def _single_line_disallowed_indexes(value: str) -> list[int]:
    disallowed = []
    for index, character in enumerate(value):
        if unicodedata.category(character) not in SINGLE_LINE_DISALLOWED_CATEGORIES:
            continue
        if character in ZERO_WIDTH_JOINERS and _is_joiner_between_non_ascii(value, index):
            continue
        if ord(character) in FLAG_TAG_RANGE and _is_flag_tag(value, index):
            continue
        disallowed.append(index)
    return disallowed


def _normalise_single_line(value: str) -> str:
    normalised = unicodedata.normalize("NFC", value).translate(MISDECODED_PUNCTUATION)
    return _strip_edges(normalised.translate(SINGLE_LINE_REMOVED_CHARACTERS))


def clean_single_line(value: Any) -> Any:
    """Normalises a single-line value and rejects control, formatting and line-separator characters.

    Misdecoded Windows-1252 quotes and dashes become ASCII, and soft hyphens and directional marks are removed.
    ZWJ/ZWNJ are allowed only between two non-ASCII characters, as in emoji sequences and some scripts, and tag
    characters only within a flag sequence.

    Args:
        value: The raw input. Non-string values are returned unchanged for pydantic to validate.

    Returns:
        The NFC-normalised value with whitespace and invisible characters stripped from both ends.

    Raises:
        ValueError: If the value contains a disallowed Cc, Cf, Zl or Zp character.
    """
    if not isinstance(value, str):
        return value
    cleaned = _normalise_single_line(value)
    if _single_line_disallowed_indexes(cleaned):
        raise ValueError("Text must be a single line without control or invisible formatting characters")
    return cleaned


def strip_to_single_line(value: str) -> str:
    """Cleans a value as `clean_single_line` does, but removes disallowed characters instead of rejecting them.

    For text that can't be rejected with a 422, such as uploaded filenames.

    Args:
        value: The raw text.

    Returns:
        The cleaned value, which may be empty.
    """
    normalised = _normalise_single_line(value)
    disallowed = set(_single_line_disallowed_indexes(normalised))
    return _strip_edges("".join(c for i, c in enumerate(normalised) if i not in disallowed))


def clean_multiline(value: Any) -> Any:
    """Normalises multiline text, converting line breaks and removing invisible and control characters.

    Args:
        value: The raw input. Non-string values are returned unchanged for pydantic to validate.

    Returns:
        The NFC-normalised value with `\\r\\n`, `\\r`, `\\v`, `\\f`, NEL and Unicode line/paragraph separators
        converted to `\\n`, misdecoded Windows-1252 quotes and dashes converted to ASCII, zero-width spaces, BOMs,
        bidi controls and other control characters (apart from `\\n` and `\\t`) removed, and whitespace stripped
        from both ends.
    """
    if not isinstance(value, str):
        return value
    normalised = unicodedata.normalize("NFC", value).replace("\r\n", "\n").translate(MULTILINE_LINE_BREAKS)
    removed = normalised.translate(MISDECODED_PUNCTUATION).translate(MULTILINE_REMOVED_CHARACTERS)
    cleaned = "".join(
        c for c in removed if unicodedata.category(c) != "Cc" or c in MULTILINE_ALLOWED_CONTROL_CHARACTERS
    )
    return _strip_edges(cleaned)


CleanSingleLine = BeforeValidator(clean_single_line)
CleanMultiline = BeforeValidator(clean_multiline)
