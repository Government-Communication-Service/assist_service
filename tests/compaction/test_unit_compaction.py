import logging

from app.compaction.service import (
    estimate_message_tokens,
)

logger = logging.getLogger(__name__)


def test_estimate_message_tokens_empty_content():
    result = estimate_message_tokens("")
    assert result == 0


def test_estimate_message_tokens_none_content():
    result = estimate_message_tokens(None)
    assert result == 0


def test_estimate_message_tokens_simple_content():
    content = "Hello world"
    result = estimate_message_tokens(content)
    # "Hello world" is 11 characters, 11/3.5 = 3.14, int(3.14) = 3
    assert result == 3


def test_estimate_message_tokens_longer_content():
    content = "This is a longer message with more content to test token estimation"
    result = estimate_message_tokens(content)
    # 67 characters, 67/3.5 = 19.14, int(19.14) = 19
    assert result == 19


def test_estimate_message_tokens_various_lengths():
    test_cases = [
        ("", 0),
        ("a", 0),  # 1/3.5 = 0.28, int(0.28) = 0
        ("ab", 0),  # 2/3.5 = 0.57, int(0.57) = 0
        ("abc", 0),  # 3/3.5 = 0.85, int(0.85) = 0
        ("abcd", 1),  # 4/3.5 = 1.14, int(1.14) = 1
        ("abcdef", 1),  # 6/3.5 = 1.71, int(1.71) = 1
        ("abcdefg", 2),  # 7/3.5 = 2.0, int(2.0) = 2
        ("abcdefgh", 2),  # 8/3.5 = 2.28, int(2.28) = 2
        ("abcdefghi", 2),  # 9/3.5 = 2.57, int(2.57) = 2
        ("abcdefghij", 2),  # 10/3.5 = 2.85, int(2.85) = 2
    ]

    for content, expected in test_cases:
        result = estimate_message_tokens(content)
        assert result == expected, f"For content '{content}' expected {expected}, got {result}"
