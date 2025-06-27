import pytest

from app.bedrock.american_word_swap import replace_american_words


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("My favorite color is blue.", "My favourite colour is blue."),
        ("My favorite", "My favourite"),
        ("The best color.", "The best colour."),
        ("behavioral economics", "behavioural economics"),
        ("Hello world!", "Hello world!"),
    ],
)
def test_replace_american_words(input_text, expected_output):
    """Test American-to-British spelling conversion."""
    assert replace_american_words(input_text) == expected_output
