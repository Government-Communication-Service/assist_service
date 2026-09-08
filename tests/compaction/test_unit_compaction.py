import logging
from types import SimpleNamespace
from unittest.mock import patch

from app.compaction.service import (
    estimate_message_tokens,
    estimate_prefix_tokens,
    find_last_assistant_message_id,
    messages_since_compaction,
    should_compact,
)
from app.config import settings

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


# --- estimate_prefix_tokens ----------------------------------------------------------------


def test_estimate_prefix_tokens_sums_across_messages():
    formatted_messages = [
        {"role": "user", "content": "abcd"},  # 4/3.5 -> 1
        {"role": "assistant", "content": "abcdefg"},  # 7/3.5 -> 2
    ]
    assert estimate_prefix_tokens(formatted_messages) == 3


def test_estimate_prefix_tokens_empty_list():
    assert estimate_prefix_tokens([]) == 0


# --- find_last_assistant_message_id ---------------------------------------------------------


def _msg(message_id, role):
    return SimpleNamespace(id=message_id, role=role)


def test_find_last_assistant_message_id_finds_the_last_one():
    messages = [_msg(1, "user"), _msg(2, "assistant"), _msg(3, "user"), _msg(4, "assistant"), _msg(5, "user")]
    assert find_last_assistant_message_id(messages) == 4


def test_find_last_assistant_message_id_none_when_no_assistant_message():
    messages = [_msg(1, "user")]
    assert find_last_assistant_message_id(messages) is None


def test_find_last_assistant_message_id_empty_list():
    assert find_last_assistant_message_id([]) is None


# --- messages_since_compaction -------------------------------------------------------------


def test_messages_since_compaction_counts_everything_when_never_compacted():
    messages = [_msg(1, "user"), _msg(2, "assistant")]
    assert messages_since_compaction(messages, None) == 2


def test_messages_since_compaction_counts_only_messages_after_the_cut_point():
    messages = [_msg(1, "user"), _msg(2, "assistant"), _msg(3, "user"), _msg(4, "assistant")]
    compaction = SimpleNamespace(up_to_message_id=2)
    assert messages_since_compaction(messages, compaction) == 2


# --- should_compact -----------------------------------------------------------------------


def test_should_compact_below_threshold():
    with (
        patch.object(settings, "compaction_token_threshold", 1000),
        patch.object(settings, "compaction_min_messages", 1),
    ):
        assert should_compact(999, 6) is False


def test_should_compact_at_threshold():
    with (
        patch.object(settings, "compaction_token_threshold", 1000),
        patch.object(settings, "compaction_min_messages", 1),
    ):
        assert should_compact(1000, 6) is True


def test_should_compact_respects_feature_flag():
    with patch.object(settings, "compaction_enabled", False):
        with patch.object(settings, "compaction_token_threshold", 1000):
            assert should_compact(50000, 6) is False


def test_should_compact_respects_min_messages_even_over_threshold():
    """A chat with very few messages since the last compaction, but one huge one, can cross
    the token threshold without there being much worth summarising yet."""
    with (
        patch.object(settings, "compaction_token_threshold", 1000),
        patch.object(settings, "compaction_min_messages", 6),
    ):
        assert should_compact(50000, 2) is False


def test_should_compact_at_min_messages():
    with (
        patch.object(settings, "compaction_token_threshold", 1000),
        patch.object(settings, "compaction_min_messages", 6),
    ):
        assert should_compact(50000, 6) is True
