"""Tests for ToolArgs / ToolDef spec generation and validation."""

from typing import Literal

import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
from pydantic import Field, ValidationError

from app.bedrock.tool_schema import LenientList, ToolArgs, ToolDef, ToolResponseError


class Item(ToolArgs):
    """An item."""

    title: str = Field(description="The item's title")
    score: int


class SampleArgs(ToolArgs):
    """Do a sample thing."""

    query: str = Field(description="What to search for")
    flag: bool
    kind: Literal["a", "b"]
    note: str | None
    items: LenientList[Item]


SAMPLE = ToolDef("sample_tool", SampleArgs)

VALID_INPUT = {"query": "q", "flag": True, "kind": "a", "note": None, "items": [{"title": "t", "score": 1}]}


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _object_nodes(schema):
    return [n for n in _walk(schema) if n.get("type") == "object"]


def _message(content) -> Message:
    return Message(
        id="msg_1",
        type="message",
        role="assistant",
        model="m",
        content=content,
        stop_reason="tool_use",
        stop_sequence=None,
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def _tool_use(name: str, tool_input: dict, block_id: str = "toolu_1") -> ToolUseBlock:
    return ToolUseBlock(id=block_id, type="tool_use", name=name, input=tool_input)


# --- spec ---


def test_spec_top_level_fields():
    spec = SAMPLE.spec
    assert spec["name"] == "sample_tool"
    assert spec["description"] == "Do a sample thing."
    assert spec["strict"] is True
    assert "description" not in spec["input_schema"]


def test_spec_non_strict_omits_strict_key():
    assert "strict" not in ToolDef("t", SampleArgs, strict=False).spec


def test_spec_additional_properties_false_on_every_object_including_defs():
    schema = SAMPLE.spec["input_schema"]
    assert "$defs" in schema
    objects = _object_nodes(schema)
    assert len(objects) >= 2
    assert all(o.get("additionalProperties") is False for o in objects)


def test_spec_required_lists_all_fields():
    schema = SAMPLE.spec["input_schema"]
    assert set(schema["required"]) == {"query", "flag", "kind", "note", "items"}
    assert set(schema["$defs"]["Item"]["required"]) == {"title", "score"}


def test_spec_no_auto_titles_but_title_property_survives():
    schema = SAMPLE.spec["input_schema"]
    assert all("title" not in n or isinstance(n["title"], dict) for n in _walk(schema))
    assert "title" in schema["$defs"]["Item"]["properties"]


def test_spec_field_descriptions_kept():
    assert SAMPLE.spec["input_schema"]["properties"]["query"]["description"] == "What to search for"


def test_spec_is_stable_across_accesses_and_instances():
    assert SAMPLE.spec == ToolDef("sample_tool", SampleArgs).spec
    assert SAMPLE.spec is SAMPLE.spec


def test_forced_choice_disables_parallel_tool_use():
    assert SAMPLE.forced_choice == {"type": "tool", "name": "sample_tool", "disable_parallel_tool_use": True}


# --- parse ---


def test_parse_valid():
    args = SAMPLE.parse(VALID_INPUT)
    assert isinstance(args, SampleArgs)
    assert args.items[0].title == "t"


def test_parse_stringified_list():
    args = SAMPLE.parse({**VALID_INPUT, "items": '[{"title": "t", "score": 2}]'})
    assert args.items[0].score == 2


def test_parse_ignores_extra_keys():
    args = SAMPLE.parse({**VALID_INPUT, "unexpected": 1, "items": [{"title": "t", "score": 1, "x": 2}]})
    assert not hasattr(args, "unexpected")


def test_parse_coerces_string_bool_and_int():
    args = SAMPLE.parse({**VALID_INPUT, "flag": "true", "items": [{"title": "t", "score": "3"}]})
    assert args.flag is True
    assert args.items[0].score == 3


@pytest.mark.parametrize(
    "bad",
    [
        {**VALID_INPUT, "items": "not json"},
        {**VALID_INPUT, "items": "{}"},
        {**VALID_INPUT, "kind": "c"},
        {k: v for k, v in VALID_INPUT.items() if k != "query"},
        {**VALID_INPUT, "flag": "maybe"},
    ],
)
def test_parse_garbage_raises_validation_error(bad):
    with pytest.raises(ValidationError):
        SAMPLE.parse(bad)


# --- parse_response ---


def test_parse_response_single_block():
    response = _message([TextBlock(type="text", text="hi"), _tool_use("sample_tool", VALID_INPUT)])
    assert SAMPLE.parse_response(response).query == "q"


@pytest.mark.parametrize(
    "content",
    [
        [TextBlock(type="text", text="no tool")],
        [_tool_use("other_tool", VALID_INPUT)],
        [_tool_use("sample_tool", VALID_INPUT, "toolu_1"), _tool_use("sample_tool", VALID_INPUT, "toolu_2")],
    ],
)
def test_parse_response_wrong_block_count_raises(content):
    with pytest.raises(ToolResponseError):
        SAMPLE.parse_response(_message(content))


def test_parse_response_invalid_input_raises_validation_error():
    with pytest.raises(ValidationError):
        SAMPLE.parse_response(_message([_tool_use("sample_tool", {"query": "q"})]))


def test_field_names_are_the_names_the_model_uses():
    assert SAMPLE.field_names == ["query", "flag", "kind", "note", "items"]


def test_unknown_fields_lists_top_level_keys_the_args_model_ignores():
    assert SAMPLE.unknown_fields({**VALID_INPUT, "extra": 1, "other": 2}) == ["extra", "other"]
    assert SAMPLE.unknown_fields(VALID_INPUT) == []
    assert SAMPLE.unknown_fields({**VALID_INPUT, "items": [{"title": "t", "score": 1, "extra": 1}]}) == []
