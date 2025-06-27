import re

# American to British spellings dictionary
US_TO_UK_WORDS = {
    "color": "colour",
    "favorite": "favourite",
    "honor": "honour",
    "analyze": "analyse",
    "center": "centre",
    "defense": "defence",
    "organize": "organise",
    "realize": "realise",
    "traveler": "traveller",
    "meter": "metre",
    "real-time": "real-time",  # Example of hyphenated word
    "emphasizes": "emphasises",
    "organizations": "organisations",
    "organizational": "organisational",
    "emphasize": "emphasise",
    "apologize": "apologise",
    "organization": "organisation",
    "emphasizing": "emphasising",
    "programs": "programmes",
    "behavior": "behaviour",
    "program": "programme",
    "behavioral": "behavioural",
    "optimization": "optimisation",
    "licensing": "licencing",
    "prioritize": "prioritise",
    "summarize": "summarise",
    "emphasized": "emphasised",
    "summarizing": "summarising",
}

# Compile regex pattern for US word boundaries
US_WORDS_PATTERN = re.compile(r"\b(" + "|".join(re.escape(k) for k in US_TO_UK_WORDS.keys()) + r")\b")

# Regex to detect the last partial word or hyphenated partial at chunk boundaries
PARTIAL_WORD_PATTERN = re.compile(r"(\b\w+[-]?)$")


def replace_american_words(text: str) -> str:
    """Replace American spellings with British spellings."""
    return US_WORDS_PATTERN.sub(lambda match: US_TO_UK_WORDS[match.group(0)], text)
