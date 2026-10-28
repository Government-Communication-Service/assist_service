"""Tool definitions generated from Pydantic models.

One `ToolArgs` model per tool is the source of truth for both the `input_schema` sent to the
LLM and the validation of the `tool_use.input` that comes back.
"""

import json
from dataclasses import dataclass
from functools import cached_property
from typing import Annotated, Any, Generic, TypeVar

from anthropic.types import Message, ToolUseBlock
from pydantic import BaseModel, BeforeValidator, ConfigDict

ArgsT = TypeVar("ArgsT", bound="ToolArgs")
T = TypeVar("T")

REMOVE_DUPLICATES_DESCRIPTION = (
    "Leave out results already returned earlier in this conversation. Only set true when those earlier "
    "results are still visible to you in full; leave false to see them again, for example when they were "
    "summarised or are far back in the conversation. Default false."
)


class ToolResponseError(Exception):
    """Raised when a response doesn't contain exactly one tool_use block for the expected tool."""


class ToolArgs(BaseModel):
    """Base for tool argument models.

    Advertises `additionalProperties: false` in the schema (required by strict mode) but
    ignores unexpected keys on validation.
    """

    model_config = ConfigDict(extra="ignore", json_schema_extra={"additionalProperties": False})


def _strip_titles(node: object, in_properties: bool = False) -> None:
    """Remove pydantic's auto-generated "title" keys in place, preserving properties named "title"."""
    if isinstance(node, dict):
        if not in_properties:
            node.pop("title", None)
        for key, value in node.items():
            _strip_titles(value, in_properties=(key == "properties" and not in_properties))
    elif isinstance(node, list):
        for value in node:
            _strip_titles(value)


def _parse_json_string(value: object) -> object:
    """Parse a JSON-encoded string, if the LLM wrapped a structured value in one."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError as e:
            raise ValueError(f"Cannot parse stringified JSON {value!r}") from e
    return value


LenientList = Annotated[list[T], BeforeValidator(_parse_json_string)]


@dataclass(frozen=True)
class ToolDef(Generic[ArgsT]):
    """A tool's name and argument model.

    Args:
        name: Tool name sent to the LLM.
        args_model: ToolArgs subclass; its docstring is the tool description.
        strict: Send `strict: true` so the API enforces the schema. When False the key is
            omitted: Sonnet 5 on Bedrock rejects `strict` in any form (checked 2026-09-28; not re-checked on 5.5).
    """

    name: str
    args_model: type[ArgsT]
    strict: bool = True

    @cached_property
    def spec(self) -> dict[str, Any]:
        """The tool definition dict for the `tools=[...]` request param. Treat as read-only."""
        schema = self.args_model.model_json_schema()
        description = schema.pop("description", "")
        _strip_titles(schema)
        spec = {"name": self.name, "description": description, "input_schema": schema}
        if self.strict:
            spec["strict"] = True
        return spec

    @property
    def forced_choice(self) -> dict[str, Any]:
        """A tool_choice that forces exactly one call to this tool."""
        return {"type": "tool", "name": self.name, "disable_parallel_tool_use": True}

    @property
    def field_names(self) -> list[str]:
        """The top-level input field names the model can use, as sent to it in the spec."""
        return [field.alias or name for name, field in self.args_model.model_fields.items()]

    def unknown_fields(self, tool_input: dict) -> list[str]:
        """Top-level keys of a tool_use input that the args model has no field for, and so ignores."""
        accepted = set(self.field_names)
        if self.args_model.model_config.get("populate_by_name"):
            accepted.update(self.args_model.model_fields)
        return [key for key in tool_input if key not in accepted]

    def parse(self, tool_input: dict) -> ArgsT:
        """Validate one tool_use block's input.

        Raises:
            pydantic.ValidationError: The input didn't validate.
        """
        return self.args_model.model_validate(tool_input)

    def parse_response(self, response: Message) -> ArgsT:
        """Validate the single tool_use block for this tool in a forced-tool response.

        Raises:
            ToolResponseError: Zero or several matching tool_use blocks.
            pydantic.ValidationError: The input didn't validate.
        """
        matches = [b.input for b in response.content if isinstance(b, ToolUseBlock) and b.name == self.name]
        if len(matches) != 1:
            raise ToolResponseError(f"Expected exactly one {self.name!r} tool_use block, found {len(matches)}")
        return self.parse(matches[0])
