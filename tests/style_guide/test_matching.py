"""Finding pattern matches, splitting text into chunks and locating quotes."""

import pytest

from app.style_guide.matching import (
    find_matches,
    is_sentence_start,
    locate_quote,
    match_context,
    rule_matches,
    split_into_chunks,
)
from app.style_guide.rules import parse_rule

pytestmark = [pytest.mark.unit]


def make_rule(rule_id: str = "rule_x", **check):
    check.setdefault("llm", "none")
    return parse_rule({"rule_id": rule_id, "rule": "Example", "priority": 100, "details": "Guidance.", "check": check})


class TestCorrectForms:
    """A case rule matches any casing of its correct form except the correct one."""

    def test_other_casing_matches_with_the_correct_form_as_suggestion(self):
        rule = make_rule(correct_forms=["internet"])

        [match] = rule_matches("We use the Internet daily.", rule)

        assert (match.text, match.suggestion) == ("Internet", "internet")

    def test_correct_form_and_plural_do_not_match(self):
        rule = make_rule(correct_forms=["council"])

        assert rule_matches("The council and other councils met.", rule) == []

    @pytest.mark.parametrize(
        "text",
        [
            "Internet use is up.",
            "It rose. Internet use is up.",
            "Heading\nInternet use",
            "- Internet use",
            "1. Internet",
        ],
    )
    def test_capital_first_letter_is_allowed_at_the_start_of_a_sentence_line_or_list_item(self, text):
        assert rule_matches(text, make_rule(correct_forms=["internet"])) == []

    def test_capitalised_proper_name_matches_mid_sentence(self):
        rule = make_rule(correct_forms=["Child Benefit"])

        [match] = rule_matches("You can claim child benefit now.", rule)

        assert match.text == "child benefit"

    def test_not_whole_word_matches_inside_words(self):
        rule = make_rule(correct_forms=["net"], whole_word=False)

        assert [m.text for m in rule_matches("the interNET", rule)] == ["NET"]


class TestPatterns:
    def test_matches_are_tagged_with_their_order(self):
        rule = make_rule(patterns=[r"\bPIP\b", r"\bESA\b"], case_sensitive=True)

        matches = rule_matches("ESA first, then PIP, then ESA again.", rule)

        assert [(m.text, m.index, m.total, m.first_text) for m in matches] == [
            ("ESA", 1, 3, "ESA"),
            ("PIP", 2, 3, "ESA"),
            ("ESA", 3, 3, "ESA"),
        ]

    def test_overlapping_patterns_give_one_match_per_start(self):
        rule = make_rule(patterns=["click", "click here"])

        [match] = rule_matches("Please click here.", rule)

        assert match.text == "click here"

    def test_find_matches_leaves_out_rules_without_matches(self):
        hit = make_rule("rule_a", patterns=["etc"])
        miss = make_rule("rule_b", patterns=["zzz"])

        assert list(find_matches("Apples, pears etc.", [hit, miss])) == ["rule_a"]


class TestMatchers:
    def test_long_sentences(self):
        rule = make_rule(matcher="long_sentences", max_words=5)
        text = "Short one. This sentence has far more than five words in it.\nAnother short line"

        [match] = rule_matches(text, rule)

        assert match.text == "This sentence has far more than five words in it."
        assert text[match.start : match.end] == match.text

    def test_american_spelling_suggests_british(self):
        rule = make_rule(matcher="american_spelling", llm="matches")

        [match] = rule_matches("We organize events.", rule)

        assert (match.text, match.suggestion) == ("organize", "organise")


class TestContext:
    def test_sentence_start(self):
        text = "One. Two\n- three"
        assert is_sentence_start(text, 0)
        assert is_sentence_start(text, text.index("Two"))
        assert is_sentence_start(text, text.index("three"))
        assert not is_sentence_start("one two", 4)

    def test_match_context_is_the_sentence_and_the_one_before(self):
        text = "First sentence. Second has etc in it. Third."
        start = text.index("etc")

        assert match_context(text, start, start + 3) == "First sentence. Second has etc in it."


class TestChunks:
    def test_short_text_is_one_chunk(self):
        assert split_into_chunks("Just this.", 100) == [(0, "Just this.")]

    def test_chunks_follow_paragraphs_and_keep_offsets(self):
        paragraphs = [f"Paragraph {i} " + "word " * 20 for i in range(6)]
        text = "\n\n".join(paragraphs)

        chunks = split_into_chunks(text, 300)

        assert len(chunks) > 1
        for offset, chunk in chunks:
            assert text[offset : offset + len(chunk)] == chunk
            assert len(chunk) <= 300
        assert " ".join(c for _, c in chunks).split() == text.split()

    def test_long_paragraph_is_split_at_sentence_ends(self):
        text = "A sentence here. " * 30

        chunks = split_into_chunks(text, 100)

        assert all(len(chunk) <= 100 for _, chunk in chunks)
        assert all(chunk.rstrip().endswith(".") for _, chunk in chunks[:-1])
        assert "".join(chunk for _, chunk in chunks) == text


class TestLocateQuote:
    def test_exact(self):
        assert locate_quote("one two three", "two") == (4, 7)

    def test_whitespace_and_case_differences(self):
        text = "Visit the online\n  Hub today"

        start, end = locate_quote(text, "online hub")

        assert text[start:end] == "online\n  Hub"

    def test_curly_quotes(self):
        text = "It’s the ‘Hub’."

        start, end = locate_quote(text, "It's the 'Hub'")

        assert text[start:end] == "It’s the ‘Hub’"

    def test_missing_quote(self):
        assert locate_quote("one two", "three") is None
        assert locate_quote("one two", "  ") is None
