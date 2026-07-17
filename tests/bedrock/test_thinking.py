"""Tests for the thinking_kwargs helper and ThinkingLevel enum."""

from unittest.mock import MagicMock, patch

import pytest

from app.bedrock import BedrockHandler, RunMode
from app.bedrock.thinking import ThinkingLevel, thinking_kwargs


def test_disabled_returns_thinking_disabled():
    result = thinking_kwargs(ThinkingLevel.disabled)
    assert result == {"thinking": {"type": "disabled"}}


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


@patch("app.bedrock.bedrock.bedrock_stream")
def test_stream_on_complete_not_in_extra_api_kwargs(mock_bedrock_stream):
    """Regression: on_complete must not leak into extra_api_kwargs and reach the SDK."""
    llm = MagicMock()
    llm.model = "gpt-4o"  # non-cross-region so model id is unchanged
    llm.max_tokens = 1000
    handler = BedrockHandler(llm=llm, mode=RunMode.ASYNC)

    on_complete_cb = MagicMock()

    handler._stream(
        messages=[],
        system="",
        on_complete=on_complete_cb,
        output_config={"effort": "high"},
    )

    stream_input = mock_bedrock_stream.call_args[0][0]
    assert stream_input.on_complete is on_complete_cb
    assert "on_complete" not in stream_input.extra_api_kwargs
    assert stream_input.extra_api_kwargs == {"output_config": {"effort": "high"}}
