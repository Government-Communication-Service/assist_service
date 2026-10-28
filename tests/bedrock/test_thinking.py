"""Tests for the thinking_kwargs helper and ThinkingLevel enum."""

import pytest

from app.bedrock.thinking import ThinkingLevel, thinking_kwargs


def test_disabled_returns_between_tools():
    """Sonnet 5.5 rejects thinking.type=disabled; between_tools is its lowest setting and takes no effort."""
    result = thinking_kwargs(ThinkingLevel.disabled)
    assert result == {"thinking": {"type": "between_tools"}}


@pytest.mark.parametrize(
    "level",
    [
        ThinkingLevel.low,
        ThinkingLevel.medium,
        ThinkingLevel.high,
        ThinkingLevel.xhigh,
        ThinkingLevel.max,
    ],
)
def test_effort_levels_return_output_config(level):
    result = thinking_kwargs(level)
    assert result == {"output_config": {"effort": level.value}}
    assert "thinking" not in result


def test_effort_low_value():
    assert thinking_kwargs(ThinkingLevel.low) == {"output_config": {"effort": "low"}}


def test_effort_max_value():
    assert thinking_kwargs(ThinkingLevel.max) == {"output_config": {"effort": "max"}}


def test_thinking_level_from_string():
    """Pydantic and pydantic-settings will coerce strings to enum members."""
    assert ThinkingLevel("disabled") == ThinkingLevel.disabled
    assert ThinkingLevel("high") == ThinkingLevel.high
    assert ThinkingLevel("max") == ThinkingLevel.max


def test_invalid_thinking_level_raises():
    with pytest.raises(ValueError):
        ThinkingLevel("ultra")


def test_thinking_level_values_are_strings():
    """Members are str instances so they serialise cleanly in API responses."""
    for member in ThinkingLevel:
        assert isinstance(member.value, str)
