from typing import Annotated, Optional

import pytest
from pydantic import BaseModel, StringConstraints, ValidationError

from app.schemas.text import CleanMultiline, CleanSingleLine, strip_to_single_line

pytestmark = [pytest.mark.unit]

ENGLAND_FLAG = "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f"
WOMAN_TECHNOLOGIST = "\U0001f469\u200d\U0001f4bb"


class Model(BaseModel):
    name: Optional[Annotated[str, StringConstraints(min_length=1, max_length=10), CleanSingleLine]] = None
    text: Optional[Annotated[str, StringConstraints(max_length=40), CleanMultiline]] = None


class TestCleanSingleLine:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  Campaign  ", "Campaign"),
            ("\u200bCampaign\ufeff", "Campaign"),
            ("\u00a0Campaign\u3000", "Campaign"),
            ("Cafe\u0301", "Caf\u00e9"),
            (f"{WOMAN_TECHNOLOGIST} Dev", f"{WOMAN_TECHNOLOGIST} Dev"),
            (f"UK {ENGLAND_FLAG}", f"UK {ENGLAND_FLAG}"),
            ("soft\u00adhyph", "softhyph"),
            ("don\x92t\x96x", "don't-x"),
            ("\u200f\u0645\u0631\u062d\u0628\u0627\u200e", "\u0645\u0631\u062d\u0628\u0627"),
        ],
    )
    def test_cleans(self, raw, expected):
        assert Model(name=raw).name == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "a\nb",
            "a\tb",
            "a\x00b",
            "a\x1bb",
            "abc\u202edef",
            "Camp\u200baign",
            "Camp\u200daign",
            "a\u2028b",
            "a\u2029b",
            "a\U000e0067b",
        ],
    )
    def test_rejects_control_formatting_and_separator_characters(self, raw):
        with pytest.raises(ValidationError, match="single line"):
            Model(name=raw)

    @pytest.mark.parametrize("raw", ["\u200b\u200b \u200d", "\u3164", "\u2800 \u2800", "\u00ad\u200e"])
    def test_invisible_only_value_fails_min_length(self, raw):
        with pytest.raises(ValidationError, match="at least 1 character"):
            Model(name=raw)

    def test_length_checked_after_cleaning(self):
        assert Model(name="   " + "x" * 10 + "   ").name == "x" * 10

    def test_none_passes_through(self):
        assert Model(name=None).name is None


class TestStripToSingleLine:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("report\u202etxt.exe.txt", "reporttxt.exe.txt"),
            ("two\nlines.pdf", "twolines.pdf"),
            ("\u200b\n", ""),
            (f"Team {ENGLAND_FLAG}.pdf", f"Team {ENGLAND_FLAG}.pdf"),
            ("Caf\u00e9 notes.docx", "Caf\u00e9 notes.docx"),
        ],
    )
    def test_strips(self, raw, expected):
        assert strip_to_single_line(raw) == expected


class TestCleanMultiline:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  line one\r\nline two\r  ", "line one\nline two"),
            ("a\tb", "a\tb"),
            ("page\fbreak\vline\x85nel\u2028ls\u2029ps", "page\nbreak\nline\nnel\nls\nps"),
            ("\ufeffpasted\u200b text", "pasted text"),
            ("abc\u202edef\u2066", "abcdef"),
            ("a\x00b\x1bc\x7fde", "abcde"),
            ("don\x92t \x93quote\x94 \x91x\x92 a\x96b\x97c", "don't \"quote\" 'x' a-b-c"),
            ("soft\u00adhyphen", "soft\u00adhyphen"),
            (WOMAN_TECHNOLOGIST, WOMAN_TECHNOLOGIST),
        ],
    )
    def test_cleans(self, raw, expected):
        assert Model(text=raw).text == expected
