from pydantic import Field, model_validator

from app.bedrock.tool_schema import ToolArgs, ToolDef


class CheckStyleGuideArgs(ToolArgs):
    """Check content against the GOV.UK style guide (A to Z and related rules). Returns the issues found,
    each with the rule, its guidance and where it occurs. It does not return a corrected version. Use it
    when the user asks for a style guide check, or for content to be checked or made compliant with
    GOV.UK style. Give exactly one of text or document_id. To check text from the user's message or the
    conversation, copy just that text into text, leaving out any request around it.

    By default, say briefly what was checked, then go through the issues grouped by rule, quoting the
    original wording and the correction. If the user asked for something narrower, such as only the
    issues or one section, answer that instead. Issues detected by pattern match were found without
    reading the surrounding text, so leave out any that are clearly fine there."""

    text: str | None = Field(default=None, description="The text to check, copied exactly.")
    document_id: str | None = Field(
        default=None, description="The id of one attached document to check, as listed in the context."
    )
    context: str | None = Field(
        default=None,
        description=(
            "Optional guidance for the checker: what the content is, who it is for, or what the user wants "
            "checked. Not itself checked."
        ),
    )

    @model_validator(mode="after")
    def _check_one_source(self) -> "CheckStyleGuideArgs":
        if (self.text is None) == (self.document_id is None):
            raise ValueError("Give exactly one of 'text' or 'document_id'")
        return self


CHECK_STYLE_GUIDE = ToolDef("check_style_guide", CheckStyleGuideArgs, strict=False)
